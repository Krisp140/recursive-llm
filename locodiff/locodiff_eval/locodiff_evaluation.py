"""
LoCoDiff Evaluation Script for RLM with Partition Strategies

This script evaluates RLM on the LoCoDiff benchmark dataset, which tests
LLMs' ability to understand git history and reconstruct code from diffs.
"""

import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import difflib

from dotenv import load_dotenv

# Add parent directory to path for imports when run as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from rlm import RLM
import litellm


class LoCoDiffEvaluator:
    """Evaluator for running RLM on LoCoDiff benchmark."""

    def __init__(
        self,
        model: str = "gpt-5-mini",
        recursive_model: str = "gpt-5-mini",
        output_dir: str = "locodiff/locodiff_results",
        dataset_dir: str = "locodiff/locodiff_data/locodiff-250425"
    ):
        """
        Initialize the evaluator.

        Args:
            model: Root LLM model name
            recursive_model: Recursive LLM model name
            output_dir: Directory to save results
            dataset_dir: Directory containing LoCoDiff dataset
        """
        self.model = model
        self.recursive_model = recursive_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_dir = Path(dataset_dir)

        # Create plots subdirectory
        (self.output_dir / "plots").mkdir(parents=True, exist_ok=True)

    def load_locodiff_dataset(self) -> List[Dict[str, Any]]:
        """
        Load LoCoDiff dataset from prompts directory.

        Returns:
            List of examples, each containing prompt, expected_output, and metadata
        """
        prompts_dir = self.dataset_dir / "prompts"

        if not prompts_dir.exists():
            raise FileNotFoundError(
                f"LoCoDiff dataset not found at {prompts_dir}. "
                f"Please download the dataset first."
            )

        print(f"Loading LoCoDiff dataset from {prompts_dir}...")

        examples = []
        prompt_files = sorted(prompts_dir.glob("*_prompt.txt"))

        for prompt_file in prompt_files:
            # Get corresponding expected output file
            base_name = prompt_file.stem.replace("_prompt", "")
            expected_file = prompts_dir / f"{base_name}_expectedoutput.txt"

            if not expected_file.exists():
                print(f"Warning: Missing expected output for {prompt_file.name}")
                continue

            # Read files
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompt = f.read()

            with open(expected_file, 'r', encoding='utf-8') as f:
                expected_output = f.read()

            # Extract metadata from filename
            # Format: repo_filepath_prompt.txt
            parts = base_name.split('_')
            repo = parts[0] if parts else "unknown"

            # Determine language from file extension
            language = self._extract_language(base_name)

            # Count tokens in prompt
            prompt_tokens = self._count_tokens(prompt)

            examples.append({
                'id': base_name,
                'prompt': prompt,
                'expected_output': expected_output,
                'repo': repo,
                'language': language,
                'prompt_tokens': prompt_tokens,
                'filepath': base_name
            })

        print(f"✓ Loaded {len(examples)} examples")

        # Print distribution
        self._print_dataset_stats(examples)

        return examples

    def _extract_language(self, filename: str) -> str:
        """Extract programming language from filename."""
        ext_map = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.tsx': 'TypeScript',
            '.rs': 'Rust',
            '.zig': 'Zig',
            '.go': 'Go',
            '.java': 'Java',
            '.c': 'C',
            '.cpp': 'C++',
            '.rb': 'Ruby',
        }

        for ext, lang in ext_map.items():
            if ext in filename:
                return lang

        return 'Unknown'

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            # Fallback: rough estimate
            return len(text.split()) * 1.3

    def _print_dataset_stats(self, examples: List[Dict[str, Any]]):
        """Print dataset statistics."""
        from collections import Counter

        # Count by repo
        repos = Counter(ex['repo'] for ex in examples)
        print(f"\nDataset distribution by repository:")
        for repo, count in repos.most_common():
            print(f"  {repo}: {count}")

        # Count by language
        languages = Counter(ex['language'] for ex in examples)
        print(f"\nDataset distribution by language:")
        for lang, count in languages.most_common():
            print(f"  {lang}: {count}")

        # Token distribution
        token_counts = [ex['prompt_tokens'] for ex in examples]
        print(f"\nPrompt token statistics:")
        print(f"  Min: {min(token_counts):,}")
        print(f"  Max: {max(token_counts):,}")
        print(f"  Avg: {sum(token_counts)/len(token_counts):,.0f}")

        # Bucket by token count
        buckets = {
            '0-10k': sum(1 for t in token_counts if t < 10000),
            '10k-25k': sum(1 for t in token_counts if 10000 <= t < 25000),
            '25k-50k': sum(1 for t in token_counts if 25000 <= t < 50000),
            '50k+': sum(1 for t in token_counts if t >= 50000),
        }
        print(f"\nPrompt token buckets:")
        for bucket, count in buckets.items():
            print(f"  {bucket}: {count}")

    def _convert_prompt_to_rlm_format(self, prompt: str) -> Tuple[str, str]:
        """
        Convert LoCoDiff prompt to RLM (query, context) format.

        Args:
            prompt: LoCoDiff prompt with instructions + git log

        Returns:
            Tuple of (query, context)
        """
        # Split instructions from file history
        parts = prompt.split("# File History", 1)

        if len(parts) == 2:
            # Extract filename from git log line
            git_log_match = re.search(r'> git log .* -- (.+)', parts[1])
            filename = git_log_match.group(1) if git_log_match else "the file"

            query = f"Reconstruct the current state of {filename} based on the git history provided."
            context = parts[1].strip()
        else:
            # Fallback if format is different
            query = "Reconstruct the current state of the file based on the git history below."
            context = prompt

        return query, context

    def _extract_code_from_response(self, response: str) -> str:
        """
        Extract code from model response (removes markdown code blocks).

        Args:
            response: Model response potentially containing code blocks

        Returns:
            Extracted code content
        """
        # Look for code blocks with triple backticks
        pattern = r'```(?:\w+)?\n(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            # Return the last code block (most likely the final answer)
            return matches[-1].strip()

        # If no code blocks found, return the whole response
        return response.strip()

    def _evaluate_exact_match(
        self,
        predicted: str,
        expected: str
    ) -> Dict[str, Any]:
        """
        Evaluate exact match between predicted and expected output.

        Args:
            predicted: Model's predicted output
            expected: Expected ground truth output

        Returns:
            Dictionary with evaluation metrics
        """
        # Normalize: strip whitespace
        pred_normalized = predicted.strip()
        exp_normalized = expected.strip()

        # Check exact match
        exact_match = pred_normalized == exp_normalized

        # Compute similarity score (for analysis)
        similarity = difflib.SequenceMatcher(
            None,
            pred_normalized,
            exp_normalized
        ).ratio()

        # Generate diff for debugging
        diff = list(difflib.unified_diff(
            exp_normalized.splitlines(keepends=True),
            pred_normalized.splitlines(keepends=True),
            fromfile='expected',
            tofile='predicted',
            lineterm=''
        ))

        # Count line-level differences
        added_lines = sum(1 for line in diff if line.startswith('+') and not line.startswith('+++'))
        removed_lines = sum(1 for line in diff if line.startswith('-') and not line.startswith('---'))

        return {
            'exact_match': exact_match,
            'similarity': similarity,
            'diff_lines': len(diff),
            'added_lines': added_lines,
            'removed_lines': removed_lines,
            'diff_preview': ''.join(diff[:50]) if diff else None  # First 50 lines
        }

    async def evaluate_single_example_rlm(
        self,
        example: Dict[str, Any],
        partition_strategy: Optional[str] = None,
        retrieval_method: Optional[str] = None,
        parallel_subqueries: bool = False,
        max_partition_tokens: int = 4000
    ) -> Dict[str, Any]:
        """
        Evaluate a single example using RLM.

        Args:
            example: Dataset example
            partition_strategy: Partition strategy to use (None = no partitioning)
            retrieval_method: Retrieval method to use
            parallel_subqueries: Whether to use parallel subqueries
            max_partition_tokens: Max tokens per partition

        Returns:
            Results dictionary with metrics
        """
        # Convert to RLM format
        query, context = self._convert_prompt_to_rlm_format(example['prompt'])

        # Initialize RLM
        rlm_kwargs = {
            'model': self.model,
            'recursive_model': self.recursive_model,
            'max_iterations': 30,
            'max_depth': 5,
            'temperature': 1.0  # Deterministic
        }

        # Add partition/retrieval config if specified
        if partition_strategy:
            rlm_kwargs['partition_strategy'] = partition_strategy
            rlm_kwargs['retrieval_method'] = retrieval_method or 'unfiltered'
            rlm_kwargs['parallel_subqueries'] = parallel_subqueries
            rlm_kwargs['max_partition_tokens'] = max_partition_tokens

        rlm = RLM(**rlm_kwargs)

        # Measure time
        start_time = time.time()

        try:
            # Run RLM completion
            answer = await rlm.acompletion(query=query, context=context)

            elapsed_time = time.time() - start_time

            # Extract code from response
            predicted_code = self._extract_code_from_response(answer)

            # Evaluate
            eval_result = self._evaluate_exact_match(
                predicted_code,
                example['expected_output']
            )

            # Get stats
            stats = rlm.stats

            return {
                'success': True,
                'id': example['id'],
                'exact_match': eval_result['exact_match'],
                'similarity': eval_result['similarity'],
                'diff_lines': eval_result['diff_lines'],
                'predicted_output': predicted_code,
                'evaluation': eval_result,
                'llm_calls': stats['llm_calls'],
                'iterations': stats['iterations'],
                'depth': stats['depth'],
                'elapsed_time': elapsed_time,
                'repo': example['repo'],
                'language': example['language'],
                'prompt_tokens': example['prompt_tokens'],
                'partition_strategy': partition_strategy or 'none',
                'retrieval_method': retrieval_method or 'none',
                'parallel_subqueries': parallel_subqueries,
                'method': 'rlm'
            }

        except Exception as e:
            return {
                'success': False,
                'id': example['id'],
                'error': str(e),
                'repo': example['repo'],
                'language': example['language'],
                'prompt_tokens': example['prompt_tokens'],
                'partition_strategy': partition_strategy or 'none',
                'retrieval_method': retrieval_method or 'none',
                'parallel_subqueries': parallel_subqueries,
                'method': 'rlm'
            }

    async def evaluate_single_example_baseline(
        self,
        example: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate a single example using baseline (direct LLM call).

        Args:
            example: Dataset example

        Returns:
            Results dictionary with metrics
        """
        # Measure time
        start_time = time.time()

        try:
            # Call LLM directly with the full prompt
            response = await litellm.acompletion(
                model=self.model,
                messages=[
                    {"role": "user", "content": example['prompt']}
                ],
                temperature=1.0
            )

            elapsed_time = time.time() - start_time

            # Extract answer
            answer = response.choices[0].message.content

            # Extract code from response
            predicted_code = self._extract_code_from_response(answer)

            # Evaluate
            eval_result = self._evaluate_exact_match(
                predicted_code,
                example['expected_output']
            )

            return {
                'success': True,
                'id': example['id'],
                'exact_match': eval_result['exact_match'],
                'similarity': eval_result['similarity'],
                'diff_lines': eval_result['diff_lines'],
                'predicted_output': predicted_code,
                'evaluation': eval_result,
                'llm_calls': 1,
                'elapsed_time': elapsed_time,
                'repo': example['repo'],
                'language': example['language'],
                'prompt_tokens': example['prompt_tokens'],
                'partition_strategy': 'none',
                'retrieval_method': 'none',
                'parallel_subqueries': False,
                'method': 'baseline'
            }

        except Exception as e:
            return {
                'success': False,
                'id': example['id'],
                'error': str(e),
                'repo': example['repo'],
                'language': example['language'],
                'prompt_tokens': example['prompt_tokens'],
                'partition_strategy': 'none',
                'retrieval_method': 'none',
                'parallel_subqueries': False,
                'method': 'baseline'
            }

    async def evaluate_dataset(
        self,
        examples: List[Dict[str, Any]],
        partition_strategies: List[Optional[str]],
        retrieval_methods: List[str],
        parallel_options: List[bool],
        include_baseline: bool = True,
        max_examples: Optional[int] = None
    ):
        """
        Evaluate RLM on LoCoDiff dataset with different configurations.

        Args:
            examples: List of dataset examples
            partition_strategies: List of partition strategies to test (None = no partitioning)
            retrieval_methods: List of retrieval methods to test
            parallel_options: List of parallel subquery options
            include_baseline: Whether to include baseline (direct LLM) comparison
            max_examples: Maximum number of examples to evaluate (None = all)
        """
        if max_examples:
            examples = examples[:max_examples]

        print(f"\n{'='*60}")
        print(f"Evaluating on {len(examples)} examples")
        print(f"{'='*60}\n")

        all_results = []

        # Baseline evaluation (if enabled)
        if include_baseline:
            print(f"\n{'='*60}")
            print(f"Configuration: BASELINE (Direct LLM)")
            print(f"{'='*60}")

            baseline_results = []

            for i, example in enumerate(examples, 1):
                print(f"\nExample {i}/{len(examples)}: {example['id']}")

                result = await self.evaluate_single_example_baseline(example)
                baseline_results.append(result)

                if result['success']:
                    match_status = "✓ EXACT MATCH" if result['exact_match'] else f"✗ NO MATCH (similarity: {result['similarity']:.2%})"
                    print(f"  {match_status} - Time: {result['elapsed_time']:.2f}s")
                else:
                    print(f"  ✗ Failed - {result['error']}")

            # Save baseline results
            self.save_results("baseline", baseline_results)
            all_results.extend(baseline_results)

        # RLM evaluation with different configurations
        for partition_strategy in partition_strategies:
            for retrieval_method in retrieval_methods:
                for parallel in parallel_options:

                    # Skip invalid combinations
                    if partition_strategy is None and retrieval_method != 'none':
                        continue

                    config_name = f"rlm_{partition_strategy or 'none'}_{retrieval_method}_parallel={parallel}"
                    print(f"\n{'='*60}")
                    print(f"Configuration: {config_name}")
                    print(f"{'='*60}")

                    config_results = []

                    for i, example in enumerate(examples, 1):
                        print(f"\nExample {i}/{len(examples)}: {example['id']}")

                        result = await self.evaluate_single_example_rlm(
                            example,
                            partition_strategy,
                            retrieval_method if partition_strategy else None,
                            parallel
                        )

                        config_results.append(result)

                        if result['success']:
                            match_status = "✓ EXACT MATCH" if result['exact_match'] else f"✗ NO MATCH (similarity: {result['similarity']:.2%})"
                            print(f"  {match_status} - Time: {result['elapsed_time']:.2f}s, "
                                  f"LLM calls: {result['llm_calls']}")
                        else:
                            print(f"  ✗ Failed - {result['error']}")

                    # Save configuration results
                    self.save_results(config_name, config_results)
                    all_results.extend(config_results)

        # Generate summary report
        self.generate_summary_report(all_results)

        return all_results

    def save_results(self, config_name: str, results: List[Dict[str, Any]]):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.output_dir / f"{config_name}_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Saved results to {output_file}")

    def generate_summary_report(self, all_results: List[Dict[str, Any]]):
        """Generate and save summary statistics."""
        # Group results by configuration
        configs = {}
        for result in all_results:
            config_key = (
                result.get('partition_strategy', 'none'),
                result.get('retrieval_method', 'none'),
                result.get('parallel_subqueries', False),
                result.get('method', 'rlm')
            )

            if config_key not in configs:
                configs[config_key] = {
                    'results': [],
                    'exact_matches': 0,
                    'total_time': 0,
                    'total_llm_calls': 0
                }

            configs[config_key]['results'].append(result)

            if result.get('success'):
                if result.get('exact_match'):
                    configs[config_key]['exact_matches'] += 1
                configs[config_key]['total_time'] += result.get('elapsed_time', 0)
                configs[config_key]['total_llm_calls'] += result.get('llm_calls', 0)

        # Create summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'model': self.model,
            'recursive_model': self.recursive_model,
            'configurations': {}
        }

        for config_key, data in configs.items():
            partition_strategy, retrieval_method, parallel, method = config_key
            config_name = f"{method}_{partition_strategy}_{retrieval_method}_parallel={parallel}"

            num_examples = len(data['results'])
            successful = sum(1 for r in data['results'] if r.get('success'))

            summary['configurations'][config_name] = {
                'method': method,
                'partition_strategy': partition_strategy,
                'retrieval_method': retrieval_method,
                'parallel_subqueries': parallel,
                'num_examples': num_examples,
                'successful': successful,
                'exact_match_count': data['exact_matches'],
                'exact_match_accuracy': data['exact_matches'] / num_examples if num_examples > 0 else 0,
                'avg_time': data['total_time'] / successful if successful > 0 else 0,
                'avg_llm_calls': data['total_llm_calls'] / successful if successful > 0 else 0,
                'total_time': data['total_time']
            }

        # Save summary
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = self.output_dir / f"summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*60}")
        print("SUMMARY REPORT")
        print(f"{'='*60}\n")

        for config_name, metrics in sorted(summary['configurations'].items()):
            print(f"{config_name}:")
            print(f"  Exact Match Accuracy: {metrics['exact_match_accuracy']:.1%} ({metrics['exact_match_count']}/{metrics['num_examples']})")
            print(f"  Avg Time: {metrics['avg_time']:.2f}s")
            print(f"  Avg LLM Calls: {metrics['avg_llm_calls']:.1f}")
            print()

        print(f"✓ Saved summary to {summary_file}")


async def main():
    """Main evaluation function."""

    # Initialize evaluator
    # Change models based on your API keys (OpenAI, Anthropic, or Gemini)
    evaluator = LoCoDiffEvaluator(
        model="gpt-5-mini",                    # Or "claude-sonnet-4", "gemini/gemini-2.5-pro"
        recursive_model="gpt-5-mini",     # Or "claude-haiku-4", "gemini/gemini-2.5-flash"
        output_dir="locodiff_results",
        dataset_dir="locodiff/locodiff_data/locodiff-250425"
    )

    # Load LoCoDiff dataset
    examples = evaluator.load_locodiff_dataset()

    # Define configurations to test
    partition_strategies = [
        None,  # No partitioning (traditional RLM)
        "token",
        "structural",
        "semantic"
    ]

    retrieval_methods = [
        "none",
        "unfiltered",
        "regex",
        "embedding"
    ]

    parallel_options = [
        False,  # Sequential
        # True  # Parallel (uncomment when ready to test)
    ]

    # Run evaluation
    # Start with a small subset for testing
    results = await evaluator.evaluate_dataset(
        examples,
        partition_strategies,
        retrieval_methods,
        parallel_options,
        include_baseline=True,
        max_examples=5  # Change to None for full evaluation
    )

    print("\n✓ Evaluation complete!")
    print(f"Results saved to {evaluator.output_dir}")


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    # Check for API key
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("GEMINI_API_KEY")):
        print("❌ Error: No API key found!")
        print("\nPlease set your API key in .env file:")
        print("  OPENAI_API_KEY=your-key-here")
        print("  or")
        print("  ANTHROPIC_API_KEY=your-key-here")
        print("  or")
        print("  GEMINI_API_KEY=your-key-here")
        exit(1)

    asyncio.run(main())
