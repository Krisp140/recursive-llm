"""
Quick baseline test for LoCoDiff evaluation.

This script runs a simple test on 1-2 examples to verify the setup works
before running a full evaluation.
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path so we can import locodiff_eval
sys.path.insert(0, str(Path(__file__).parent.parent))

from locodiff_eval.locodiff_evaluation import LoCoDiffEvaluator


async def main():
    """Run a quick baseline test."""
    print("\n" + "="*60)
    print("LOCODIFF BASELINE TEST")
    print("="*60 + "\n")

    # Initialize evaluator
    # Use OpenAI by default (change to gemini if you have that configured)
    evaluator = LoCoDiffEvaluator(
        model="gpt-5-mini",  # Use faster/cheaper model for testing
        recursive_model="gpt-5-mini",
        output_dir="locodiff/locodiff_results/test"
    )

    # Load dataset
    print("Loading dataset...")
    examples = evaluator.load_locodiff_dataset()

    # Select a simple example (shortest prompt)
    examples_sorted = sorted(examples, key=lambda x: x['prompt_tokens'])
    test_examples = examples_sorted[:2]  # Take 2 shortest examples

    print(f"\n{'='*60}")
    print(f"Testing on {len(test_examples)} examples (shortest prompts)")
    print(f"{'='*60}\n")

    for i, example in enumerate(test_examples, 1):
        print(f"\nExample {i}/{len(test_examples)}")
        print(f"  ID: {example['id']}")
        print(f"  Repo: {example['repo']}")
        print(f"  Language: {example['language']}")
        print(f"  Prompt tokens: {example['prompt_tokens']:,}")

    # Test baseline (direct LLM)
    print(f"\n{'='*60}")
    print("Testing BASELINE (Direct LLM)")
    print(f"{'='*60}\n")

    baseline_results = []
    for i, example in enumerate(test_examples, 1):
        print(f"\nExample {i}/{len(test_examples)}: {example['id']}")

        result = await evaluator.evaluate_single_example_baseline(example)
        baseline_results.append(result)

        if result['success']:
            match_status = "✓ EXACT MATCH" if result['exact_match'] else f"✗ NO MATCH (similarity: {result['similarity']:.2%})"
            print(f"  {match_status}")
            print(f"  Time: {result['elapsed_time']:.2f}s")
            print(f"  LLM Calls: {result['llm_calls']}")

            if not result['exact_match']:
                print(f"  Diff lines: {result['diff_lines']}")
                if result['evaluation']['diff_preview']:
                    print(f"  Diff preview:")
                    for line in result['evaluation']['diff_preview'].split('\n')[:10]:
                        print(f"    {line}")
        else:
            print(f"  ✗ Failed - {result['error']}")

    # Test RLM (no partitioning)
    print(f"\n{'='*60}")
    print("Testing RLM (No Partitioning)")
    print(f"{'='*60}\n")

    rlm_results = []
    for i, example in enumerate(test_examples, 1):
        print(f"\nExample {i}/{len(test_examples)}: {example['id']}")

        result = await evaluator.evaluate_single_example_rlm(
            example,
            partition_strategy=None,
            retrieval_method=None,
            parallel_subqueries=False
        )
        rlm_results.append(result)

        if result['success']:
            match_status = "✓ EXACT MATCH" if result['exact_match'] else f"✗ NO MATCH (similarity: {result['similarity']:.2%})"
            print(f"  {match_status}")
            print(f"  Time: {result['elapsed_time']:.2f}s")
            print(f"  LLM Calls: {result['llm_calls']}")
            print(f"  Iterations: {result['iterations']}")
        else:
            print(f"  ✗ Failed - {result['error']}")

    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}\n")

    baseline_matches = sum(1 for r in baseline_results if r.get('success') and r.get('exact_match'))
    rlm_matches = sum(1 for r in rlm_results if r.get('success') and r.get('exact_match'))

    print(f"Baseline:")
    print(f"  Exact matches: {baseline_matches}/{len(test_examples)}")
    baseline_successful = [r for r in baseline_results if r.get('success')]
    if baseline_successful:
        avg_time = sum(r['elapsed_time'] for r in baseline_successful) / len(baseline_successful)
        print(f"  Avg time: {avg_time:.2f}s")
    else:
        print(f"  No successful completions")

    print(f"\nRLM (No Partitioning):")
    print(f"  Exact matches: {rlm_matches}/{len(test_examples)}")
    rlm_successful = [r for r in rlm_results if r.get('success')]
    if rlm_successful:
        avg_time = sum(r['elapsed_time'] for r in rlm_successful) / len(rlm_successful)
        avg_calls = sum(r['llm_calls'] for r in rlm_successful) / len(rlm_successful)
        print(f"  Avg time: {avg_time:.2f}s")
        print(f"  Avg LLM calls: {avg_calls:.1f}")
    else:
        print(f"  No successful completions")

    print(f"\n{'='*60}")
    print("✓ Baseline test complete!")
    print(f"{'='*60}\n")

    if baseline_matches > 0 or rlm_matches > 0:
        print("✓ At least one exact match found! Setup is working correctly.")
        print("\nNext steps:")
        print("  - Run full evaluation: python locodiff_eval/locodiff_evaluation.py")
        print("  - Increase max_examples in main() to evaluate more cases")
    else:
        print("⚠ No exact matches found. This is expected for short tests.")
        print("  The models may need more context or different prompts.")
        print("  Try running on more examples or checking the prompt format.")


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    # Check for API key
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("GEMINI_API_KEY")):
        print("❌ Error: No API key found!")
        print("\nPlease set your API key:")
        print("  1. Create a .env file in the project root with:")
        print("     OPENAI_API_KEY=your-key-here")
        print("     or")
        print("     ANTHROPIC_API_KEY=your-key-here")
        print("     or")
        print("     GEMINI_API_KEY=your-key-here")
        exit(1)

    asyncio.run(main())
