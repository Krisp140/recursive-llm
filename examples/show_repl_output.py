"""
Demo script to show RLM REPL interactions.

This script runs RLM with verbose output showing each iteration of the REPL loop.
"""

import asyncio
import os
from dotenv import load_dotenv
from rlm import RLM
from rlm.core import RLM as RLMCore

# Load environment variables
load_dotenv()


class VerboseRLM(RLMCore):
    """RLM with verbose output showing REPL interactions."""

    def _build_custom_system_prompt(self, context_size: int, depth: int = 0) -> str:
        """Custom system prompt with more careful instructions."""
        return f"""You are a Recursive Language Model. You interact with context through a Python REPL environment.

The context is stored in variable `context` (not in this prompt). Size: {context_size:,} characters.

Available in environment:
- context: str (the document to analyze)
- query: str (the question: "{"{"}query{"}"}")
- recursive_llm(sub_query, sub_context) -> str (recursively process sub-context)
- re: already imported regex module (use re.findall, re.search, etc.)

IMPORTANT INSTRUCTIONS:
1. First, explore the context to understand what you're working with
2. Extract the relevant information carefully using regex or string operations
3. Verify your answer makes sense before calling FINAL()
4. ALWAYS print intermediate results to check your work
5. When extracting numbers, use re.search() to capture the actual number, not len()

Write Python code to answer the query. The last expression or print() output will be shown to you.

Examples:
- print(context[:200])  # See first 200 chars to understand structure
- match = re.search(r'Alice.*?(\d+)\s+commits', context)  # Extract number
- if match: print(f"Found: {{match.group(1)}}")  # Verify extraction

When you have the VERIFIED answer, use FINAL("answer") - this is NOT a function, just write it as text.

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
        from rlm.parser import parse_response, is_final
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
                    print(answer)
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


async def main():
    """Run demo with verbose output."""

    # Simple example document
    document = """
    Project Status Report - Q4 2024

    Team: Engineering
    Date: December 15, 2024

    Bugs Fixed: 127
    Features Added: 23
    Tests Written: 456
    Code Review Comments: 89

    Critical Issues:
    - Memory leak in authentication module (FIXED)
    - Database connection timeout (IN PROGRESS)
    - API rate limiting bug (OPEN)

    Team Members:
    - Alice (Backend): 45 commits
    - Bob (Frontend): 67 commits
    - Carol (DevOps): 23 commits
    """

    query = "Alice's commits?"

    print(f"\n{'='*80}")
    print("RLM VERBOSE DEMO - Watch the REPL in action!")
    print(f"{'='*80}\n")

    # Initialize verbose RLM
    rlm = VerboseRLM(
        model="gpt-4o-mini",  # Use a fast, cheap model
        max_iterations=20,
        temperature=0.0  # Deterministic for demo
    )

    try:
        result = await rlm.acompletion(query, document)

        print(f"\n{'='*80}")
        print("EXECUTION STATS")
        print(f"{'='*80}")
        print(f"Total LLM calls: {rlm.stats['llm_calls']}")
        print(f"Total iterations: {rlm.stats['iterations']}")
        print(f"Recursion depth: {rlm.stats['depth']}")
        print(f"{'='*80}\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found!")
        print("\nPlease set up your API key in .env file")
        exit(1)

    asyncio.run(main())
