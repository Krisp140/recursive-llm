"""Core RLM implementation."""

import asyncio
import re
from typing import Optional, Dict, Any, List

import litellm

from .types import Message
from .repl import REPLExecutor, REPLError
from .exceptions import FinalAnswer
from .prompts import build_system_prompt
from .parser import parse_response, is_final
from .partitions import partition_text, Partition
from .retrieval import PartitionRetriever


class RLMError(Exception):
    """Base error for RLM."""
    pass


class MaxIterationsError(RLMError):
    """Max iterations exceeded."""
    pass


class MaxDepthError(RLMError):
    """Max recursion depth exceeded."""
    pass


class RLM:
    """Recursive Language Model."""

    def __init__(
        self,
        model: str,
        recursive_model: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        max_depth: int = 5,
        max_iterations: int = 30,
        _current_depth: int = 0,
        # Phase 1: Partitioning and retrieval config
        partition_strategy: Optional[str] = None,
        retrieval_method: Optional[str] = None,
        parallel_subqueries: bool = False,
        max_parallel_subqueries: int = 5,
        max_partition_tokens: int = 4000,
        partition_overlap_tokens: int = 200,
        **llm_kwargs: Any
    ):
        """
        Initialize RLM.

        Args:
            model: Model name (e.g., "gpt-4o", "claude-sonnet-4", "ollama/llama3.2")
            recursive_model: Optional cheaper model for recursive calls
            api_base: Optional API base URL
            api_key: Optional API key
            max_depth: Maximum recursion depth
            max_iterations: Maximum REPL iterations per call
            _current_depth: Internal current depth tracker
            partition_strategy: How to partition context ("token", "structural", "semantic", "learned")
            retrieval_method: How to retrieve partitions ("regex", "embedding", "unfiltered")
            parallel_subqueries: Whether to run recursive calls in parallel
            max_parallel_subqueries: Maximum number of parallel recursive calls
            max_partition_tokens: Maximum tokens per partition
            partition_overlap_tokens: Token overlap between partitions
            **llm_kwargs: Additional LiteLLM parameters
        """
        self.model = model
        self.recursive_model = recursive_model or model
        self.api_base = api_base
        self.api_key = api_key
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self._current_depth = _current_depth
        self.llm_kwargs = llm_kwargs

        # Phase 1: Partitioning config
        self.partition_strategy = partition_strategy
        self.retrieval_method = retrieval_method
        self.parallel_subqueries = parallel_subqueries
        self.max_parallel_subqueries = max_parallel_subqueries
        self.max_partition_tokens = max_partition_tokens
        self.partition_overlap_tokens = partition_overlap_tokens

        self.repl = REPLExecutor()

        # Stats
        self._llm_calls = 0
        self._child_llm_calls = 0
        self._iterations = 0

    def completion(
        self,
        query: str = "",
        context: str = "",
        **kwargs: Any
    ) -> str:
        """
        Sync wrapper for acompletion.

        Args:
            query: User query (optional if query is in context)
            context: Context to process (optional, can pass query here)
            **kwargs: Additional LiteLLM parameters

        Returns:
            Final answer string

        Examples:
            # Standard usage
            rlm.completion(query="Summarize this", context=document)

            # Query in context (RLM will extract task)
            rlm.completion(context="Summarize this document: ...")

            # Single string (treat as context)
            rlm.completion("Process this text and extract dates")
        """
        # If only one argument provided, treat it as context
        if query and not context:
            context = query
            query = ""

        return asyncio.run(self.acompletion(query, context, **kwargs))

    async def acompletion(
        self,
        query: str = "",
        context: str = "",
        **kwargs: Any
    ) -> str:
        """
        Main async completion method.

        Args:
            query: User query (optional if query is in context)
            context: Context to process (optional, can pass query here)
            **kwargs: Additional LiteLLM parameters

        Returns:
            Final answer string

        Raises:
            MaxIterationsError: If max iterations exceeded
            MaxDepthError: If max recursion depth exceeded

        Examples:
            # Explicit query and context
            await rlm.acompletion(query="What is this?", context=doc)

            # Query embedded in context
            await rlm.acompletion(context="Extract all dates from: ...")

            # LLM will figure out the task
            await rlm.acompletion(context=document_with_instructions)
        """
        # If only query provided, treat it as context
        if query and not context:
            context = query
            query = ""
        if self._current_depth >= self.max_depth:
            raise MaxDepthError(f"Max recursion depth ({self.max_depth}) exceeded")

        # Phase 1: Partition and delegate if configured (only at root level)
        if self.partition_strategy is not None and self._current_depth == 0:
            return await self._acompletion_with_partitions(query, context, **kwargs)

        # Original behavior (no partitioning)
        # Initialize REPL environment
        repl_env = self._build_repl_env(query, context)

        # Build initial messages
        system_prompt = build_system_prompt(len(context), self._current_depth)
        messages: List[Message] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        # Main loop
        for iteration in range(self.max_iterations):
            self._iterations = iteration + 1

            # Call LLM
            response = await self._call_llm(messages, **kwargs)

            # Execute code in REPL
            try:
                exec_result = self.repl.execute(response, repl_env)
            except FinalAnswer as e:
                return str(e.value)
            except REPLError as e:
                exec_result = f"Error: {str(e)}"
            except Exception as e:
                exec_result = f"Unexpected error: {str(e)}"

            # Check for FINAL (fallback for non-code responses)
            if is_final(response):
                answer = parse_response(response, repl_env)
                if answer is not None:
                    return answer

            # Add to conversation
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": exec_result})

        raise MaxIterationsError(
            f"Max iterations ({self.max_iterations}) exceeded without FINAL()"
        )

    async def _acompletion_with_partitions(
        self,
        query: str,
        context: str,
        **kwargs: Any
    ) -> str:
        """
        Completion using partitioning strategy.

        Args:
            query: User query
            context: Full context to partition
            **kwargs: Additional LiteLLM parameters

        Returns:
            Final answer after processing partitions
        """
        # Partition the context
        partitions = partition_text(
            text=context,
            strategy=self.partition_strategy,
            max_tokens=self.max_partition_tokens,
            overlap_tokens=self.partition_overlap_tokens,
            model_name=self.model,
            api_key=self.api_key  # For semantic partitioning
        )

        # Phase 3: Apply retrieval if configured
        if self.retrieval_method is not None:
            retriever = PartitionRetriever(
                method=self.retrieval_method,
                top_k=self.max_parallel_subqueries,
                api_key=self.api_key
            )
            partitions = retriever.retrieve(query, partitions)
        else:
            # If no retrieval configured, limit to max_parallel_subqueries
            partitions = partitions[:self.max_parallel_subqueries]

        # Create child RLM with increased depth (to prevent infinite partitioning)
        child_rlm = RLM(
            model=self.recursive_model,
            recursive_model=self.recursive_model,
            api_base=self.api_base,
            api_key=self.api_key,
            max_depth=self.max_depth,
            max_iterations=self.max_iterations,
            _current_depth=self._current_depth + 1,
            # Don't partition at child level (partitioning only happens at root)
            partition_strategy=None,
            retrieval_method=None,
            parallel_subqueries=False,
            max_parallel_subqueries=self.max_parallel_subqueries,
            max_partition_tokens=self.max_partition_tokens,
            partition_overlap_tokens=self.partition_overlap_tokens,
            **self.llm_kwargs
        )

        # Process partitions (Phase 4: parallel or sequential)
        partial_answers: List[str] = []

        if self.parallel_subqueries:
            # Parallel execution using asyncio.gather
            async def process_partition(partition: Any) -> str:
                """Process a single partition and return its answer."""
                try:
                    answer = await child_rlm.acompletion(query, partition.text, **kwargs)
                    return answer
                except Exception as e:
                    return f"[Error processing partition {partition.index}: {str(e)}]"

            # Create tasks for all partitions
            tasks = [process_partition(partition) for partition in partitions]

            # Execute in parallel
            partial_answers = await asyncio.gather(*tasks)
        else:
            # Sequential execution (original behavior)
            for partition in partitions:
                # Make recursive call on this partition
                try:
                    answer = await child_rlm.acompletion(query, partition.text, **kwargs)
                    partial_answers.append(answer)
                except Exception as e:
                    # Record error but continue with other partitions
                    partial_answers.append(f"[Error processing partition {partition.index}: {str(e)}]")

        # Track child RLM calls (after all partitions are processed)
        self._child_llm_calls += child_rlm.stats['llm_calls']

        # Stitch answers together
        # For Phase 1, use simple concatenation
        # (more sophisticated stitching can be added later)
        if len(partial_answers) == 1:
            return partial_answers[0]

        # Multiple answers - ask root LM to synthesize
        stitched_answer = await self._stitch_answers(query, partial_answers, **kwargs)
        return stitched_answer

    async def _stitch_answers(
        self,
        query: str,
        partial_answers: List[str],
        **kwargs: Any
    ) -> str:
        """
        Stitch together partial answers from multiple partitions.

        Args:
            query: Original query
            partial_answers: Answers from each partition
            **kwargs: Additional LiteLLM parameters

        Returns:
            Final synthesized answer
        """
        # Build stitching prompt
        answers_text = "\n\n".join([
            f"Answer from partition {i+1}:\n{answer}"
            for i, answer in enumerate(partial_answers)
        ])

        stitching_prompt = f"""You received partial answers from different parts of a document. Synthesize them into a single final answer.

Original question: {query}

{answers_text}

Instructions:
1. If the answers are partial counts or lists from different sections, combine them (e.g. sum them up or concatenate lists).
2. If they are attempts to answer the same question from different contexts, synthesize the key information.
3. Answer the Original Question directly. Do NOT provide a meta-summary of what each partition said.
4. If the answer is a number, provide just the number or the number with units."""

        # Call LLM to synthesize
        messages: List[Message] = [
            {"role": "system", "content": "You are a helpful assistant that synthesizes information from multiple sources into a direct answer."},
            {"role": "user", "content": stitching_prompt}
        ]

        response = await self._call_llm(messages, **kwargs)
        return response

    async def _call_llm(
        self,
        messages: List[Message],
        **kwargs: Any
    ) -> str:
        """
        Call LLM API.

        Args:
            messages: Conversation messages
            **kwargs: Additional parameters (can override model here)

        Returns:
            LLM response text
        """
        self._llm_calls += 1

        # Choose model based on depth
        default_model = self.model if self._current_depth == 0 else self.recursive_model

        # Allow override via kwargs
        model = kwargs.pop('model', default_model)

        # Merge kwargs
        call_kwargs = {**self.llm_kwargs, **kwargs}
        if self.api_base:
            call_kwargs['api_base'] = self.api_base
        if self.api_key:
            call_kwargs['api_key'] = self.api_key

        # Call LiteLLM
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            **call_kwargs
        )

        # Extract text
        return response.choices[0].message.content

    def _build_repl_env(self, query: str, context: str) -> Dict[str, Any]:
        """
        Build REPL environment.

        Args:
            query: User query
            context: Context string

        Returns:
            Environment dict
        """
        def final_func(answer: Any) -> None:
            """Capture final answer."""
            raise FinalAnswer(answer)

        env: Dict[str, Any] = {
            'context': context,
            'query': query,
            'recursive_llm': self._make_recursive_fn(),
            're': re,  # Whitelist re module
            'FINAL': final_func,
            'FINAL_VAR': final_func,
        }
        return env

    def _make_recursive_fn(self) -> Any:
        """
        Create recursive LLM function for REPL.

        Returns:
            Async function that can be called from REPL
        """
        async def recursive_llm(sub_query: str, sub_context: str) -> str:
            """
            Recursively process sub-context.

            Args:
                sub_query: Query for sub-context
                sub_context: Sub-context to process

            Returns:
                Answer from recursive call
            """
            if self._current_depth + 1 >= self.max_depth:
                return f"Max recursion depth ({self.max_depth}) reached"

            # Create sub-RLM with increased depth
            sub_rlm = RLM(
                model=self.recursive_model,
                recursive_model=self.recursive_model,
                api_base=self.api_base,
                api_key=self.api_key,
                max_depth=self.max_depth,
                max_iterations=self.max_iterations,
                _current_depth=self._current_depth + 1,
                # Pass partitioning config
                partition_strategy=self.partition_strategy,
                retrieval_method=self.retrieval_method,
                parallel_subqueries=self.parallel_subqueries,
                max_parallel_subqueries=self.max_parallel_subqueries,
                max_partition_tokens=self.max_partition_tokens,
                partition_overlap_tokens=self.partition_overlap_tokens,
                **self.llm_kwargs
            )

            return await sub_rlm.acompletion(sub_query, sub_context)

        # Wrap in sync function for REPL compatibility
        def sync_recursive_llm(sub_query: str, sub_context: str) -> str:
            """Sync wrapper for recursive_llm."""
            # Check if we're in an async context
            try:
                loop = asyncio.get_running_loop()
                # We're in async context, but REPL is sync
                # Create a new thread to run async code
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        recursive_llm(sub_query, sub_context)
                    )
                    return future.result()
            except RuntimeError:
                # No running loop, safe to use asyncio.run
                return asyncio.run(recursive_llm(sub_query, sub_context))

        return sync_recursive_llm

    @property
    def stats(self) -> Dict[str, int]:
        """Get execution statistics."""
        return {
            'llm_calls': self._llm_calls,
            'child_llm_calls': self._child_llm_calls,
            'iterations': self._iterations,
            'depth': self._current_depth,
        }
