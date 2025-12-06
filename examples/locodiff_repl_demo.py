"""
Demo script to show RLM REPL interactions on a LoCoDiff example.

This loads a real LoCoDiff example and shows how RLM processes it step-by-step.
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rlm.core import RLM as RLMCore
from rlm.parser import parse_response, is_final
from rlm.repl import REPLError

# Load environment variables
load_dotenv()


class VerboseRLM(RLMCore):
    """RLM with verbose output showing REPL interactions."""

    def _build_custom_system_prompt(self, context_size: int, depth: int = 0) -> str:
        """Short and sweet system prompt for LoCoDiff."""
        return f"""You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment that can recursively query sub-LLMs, which you are strongly encouraged to use as much as possible. You will be queried iteratively until you provide a final answer.

The REPL environment is initialized with:
1. A `context` variable that contains extremely important information about your query. You should check the content of the `context` variable to understand what you are working with. Make sure you look through it sufficiently as you answer your query.
2. A `llm_query` function that allows you to query an LLM (that can handle around 500K chars) inside your REPL environment.
3. The ability to use `print()` statements to view the output of your REPL code and continue your reasoning.

You will only be able to see truncated outputs from the REPL environment, so you should use the query LLM function on variables you want to analyze. You will find this function especially useful when you have to analyze the semantics of the context. Use these variables as buffers to build up your final answer.
Make sure to explicitly look through the entire context in REPL before answering your query. An example strategy is to first look at the context and figure out a chunking strategy, then break up the context into smart chunks, and query an LLM per chunk with a particular question and save the answers to a buffer, then query an LLM with all the buffers to produce your final answer.

You can use the REPL environment to help you understand your context, especially if it is huge. Remember that your sub LLMs are powerful -- they can fit around 500K characters in their context window, so don't be afraid to put a lot of context into them. For example, a viable strategy is to feed 10 documents per sub-LLM query. Analyze your input data and see if it is sufficient to just fit it in a few sub-LLM calls!

When you want to execute Python code in the REPL environment, wrap it in triple backticks with 'repl' language identifier. For example, say we want our recursive model to search for the magic number in the context (assuming the context is a string), and the context is very long, so we want to chunk it:
```python
chunk = context[:10000]
answer = llm_query(f"What is the magic number in the context? Here is the chunk: {{chunk}}")
print(answer)
```

As an example, after analyzing the context and realizing its separated by Markdown headers, we can maintain state through buffers by chunking the context by headers, and iteratively querying an LLM over it:
```python
# After finding out the context is separated by Markdown headers, we can chunk, summarize, and answer
import re
sections = re.split(r'### (.+)', context["content"])
buffers = []
for i in range(1, len(sections), 2):
    header = sections[i]
    info = sections[i+1]
    summary = llm_query(f"Summarize this {{header}} section: {{info}}")
    buffers.append(f"{{header}}: {{summary}}")
final_answer = llm_query(f"Based on these summaries, answer the original query: {{query}}\\n\\nSummaries:\\n" + "\\n".join(buffers))
```
In the next step, we can return FINAL_VAR(final_answer).

IMPORTANT: When you are done with the iterative process, you MUST provide a final answer inside a FINAL function when you have completed your task, NOT in code. Do not use these tags unless you have completed your task. You have two options:
1. Use FINAL(your final answer here) to provide the answer directly
2. Use FINAL_VAR(variable_name) to return a variable you have created in the REPL environment as your final output

Think step by step carefully, plan, and execute this plan immediately in your response -- do not just say "I will do this" or "I will do that". Output to the REPL environment and recursive LLMs as much as possible. Remember to explicitly answer the original query in your final answer.


Depth: {depth}"""

    async def acompletion(self, query: str = "", context: str = "", **kwargs):
        """Override to add verbose output."""
        # If only query provided, treat it as context
        if query and not context:
            context = query
            query = ""

        if self._current_depth >= self.max_depth:
            from rlm.core import MaxDepthError
            raise MaxDepthError(f"Max recursion depth ({self.max_depth}) exceeded")

        # Check if partitioning is enabled
        if self.partition_strategy is not None and self._current_depth == 0:
            print(f"\n{'='*80}")
            print(f"USING PARTITIONING: {self.partition_strategy}")
            print(f"{'='*80}\n")
            return await self._acompletion_with_partitions(query, context, **kwargs)

        # Original behavior (no partitioning) - WITH VERBOSE OUTPUT
        from rlm.repl import REPLError

        # Initialize REPL environment
        repl_env = self._build_repl_env(query, context)

        # Build initial messages with CUSTOM system prompt
        system_prompt = self._build_custom_system_prompt(len(context), self._current_depth)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        print(f"\n{'='*80}")
        print(f"RLM REPL SESSION (depth={self._current_depth})")
        print(f"{'='*80}")
        print(f"Query: {query}")
        print(f"Context size: {len(context):,} characters")
        print(f"Max iterations: {self.max_iterations}")
        print(f"{'='*80}\n")

        # Main loop
        for iteration in range(self.max_iterations):
            self._iterations = iteration + 1

            print(f"\n{'─'*80}")
            print(f"ITERATION {iteration + 1}/{self.max_iterations}")
            print(f"{'─'*80}")

            # Call LLM
            response = await self._call_llm(messages, **kwargs)

            print(f"\n[LLM Response]:")
            print(response)
            print()

            # Check for FINAL
            if is_final(response):
                answer = parse_response(response, repl_env)
                if answer is not None:
                    print(f"\n{'='*80}")
                    print(f"FINAL ANSWER (iteration {iteration + 1}):")
                    print(f"{'='*80}")
                    print(answer[:500])  # Print first 500 chars
                    if len(answer) > 500:
                        print(f"\n[... {len(answer) - 500} more characters ...]")
                    print(f"\n{'='*80}")
                    return answer

            # Execute code in REPL
            try:
                exec_result = self.repl.execute(response, repl_env)
                print(f"[REPL Output]:")
                print(exec_result)
            except REPLError as e:
                exec_result = f"Error: {str(e)}"
                print(f"[REPL Error]:")
                print(exec_result)
            except Exception as e:
                exec_result = f"Unexpected error: {str(e)}"
                print(f"[Unexpected Error]:")
                print(exec_result)

            # Add to conversation
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": exec_result})

        from rlm.core import MaxIterationsError
        raise MaxIterationsError(
            f"Max iterations ({self.max_iterations}) exceeded without FINAL()"
        )


def load_shortest_locodiff_example():
    """Load the shortest LoCoDiff example for testing."""
    dataset_dir = Path("locodiff/locodiff_data/locodiff-250425/prompts")

    if not dataset_dir.exists():
        print(f"❌ LoCoDiff dataset not found at {dataset_dir}")
        print("\nPlease run: bash locodiff/scripts/download_dataset.sh")
        sys.exit(1)

    # Find all prompt files
    prompt_files = list(dataset_dir.glob("*_prompt.txt"))

    # Find shortest by file size
    shortest = min(prompt_files, key=lambda f: f.stat().st_size)

    # Load prompt and expected output
    base_name = shortest.stem.replace("_prompt", "")
    expected_file = dataset_dir / f"{base_name}_expectedoutput.txt"

    with open(shortest, 'r', encoding='utf-8') as f:
        prompt = f.read()

    with open(expected_file, 'r', encoding='utf-8') as f:
        expected_output = f.read()

    return {
        'id': base_name,
        'prompt': prompt,
        'expected_output': expected_output,
        'file': shortest.name
    }


async def main():
    """Run demo with LoCoDiff example."""

    print(f"\n{'='*80}")
    print("RLM VERBOSE DEMO - LoCoDiff Example")
    print(f"{'='*80}\n")

    # Load shortest example
    print("Loading shortest LoCoDiff example...")
    example = load_shortest_locodiff_example()

    print(f"Example ID: {example['id']}")
    print(f"Prompt size: {len(example['prompt']):,} characters")
    print(f"Expected output size: {len(example['expected_output']):,} characters")

    # Convert LoCoDiff prompt to RLM format
    # LoCoDiff prompts have instructions + git history
    parts = example['prompt'].split("# File History", 1)
    if len(parts) == 2:
        import re
        git_log_match = re.search(r'> git log .* -- (.+)', parts[1])
        filename = git_log_match.group(1) if git_log_match else "the file"
        query = f"Reconstruct the current state of {filename} based on the git history provided."
        context = parts[1].strip()
    else:
        query = "Reconstruct the current state of the file based on the git history."
        context = example['prompt']

    print(f"\nQuery: {query}")
    print(f"Context size: {len(context):,} characters\n")

    # Initialize verbose RLM
    rlm = VerboseRLM(
        model="gpt-4o-mini",  # Use a fast, cheap model
        max_iterations=15,
        temperature=0.0  # Deterministic for demo
    )

    try:
        result = await rlm.acompletion(query, context)

        print(f"\n{'='*80}")
        print("EXECUTION STATS")
        print(f"{'='*80}")
        print(f"Total LLM calls: {rlm.stats['llm_calls']}")
        print(f"Total iterations: {rlm.stats['iterations']}")
        print(f"Recursion depth: {rlm.stats['depth']}")

        # Check if correct
        result_normalized = result.strip()
        expected_normalized = example['expected_output'].strip()
        exact_match = result_normalized == expected_normalized

        print(f"\n{'='*80}")
        print("EVALUATION")
        print(f"{'='*80}")
        print(f"Exact match: {'✅ YES' if exact_match else '❌ NO'}")

        if not exact_match:
            # Show similarity
            import difflib
            similarity = difflib.SequenceMatcher(None, result_normalized, expected_normalized).ratio()
            print(f"Similarity: {similarity:.1%}")

            # Show first difference
            diff = list(difflib.unified_diff(
                expected_normalized.splitlines(keepends=True)[:20],
                result_normalized.splitlines(keepends=True)[:20],
                fromfile='expected',
                tofile='predicted',
                lineterm=''
            ))
            if diff:
                print(f"\nFirst 20 lines of diff:")
                print(''.join(diff[:30]))

        print(f"{'='*80}\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found!")
        print("\nPlease set up your API key in .env file")
        sys.exit(1)

    asyncio.run(main())
