"""
Test RLM vs Baseline on longer context examples.

This script tests examples from different token length buckets to see
where RLM partitioning starts to help vs baseline.
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
    """Run tests on longer context examples."""
    print("\n" + "="*60)
    print("LOCODIFF LONG CONTEXT TEST")
    print("="*60 + "\n")

    # Initialize evaluator
    evaluator = LoCoDiffEvaluator(
        model="gpt-4o-mini",
        recursive_model="gpt-4o-mini",
        output_dir="../../locodiff_results/test"  # Relative to locodiff/locodiff_eval/
    )

    # Load dataset
    print("Loading dataset...")
    examples = evaluator.load_locodiff_dataset()

    # Select examples from different token buckets
    test_examples = []

    # Get 1 example from each bucket
    buckets = {
        '10-25k': (10000, 25000),
        '25-50k': (25000, 50000),
        '50k+': (50000, float('inf'))
    }

    for bucket_name, (min_tokens, max_tokens) in buckets.items():
        candidates = [
            ex for ex in examples
            if min_tokens <= ex['prompt_tokens'] < max_tokens
        ]

        if candidates:
            # Take the shortest example in this bucket (for faster testing)
            selected = sorted(candidates, key=lambda x: x['prompt_tokens'])[0]
            test_examples.append((bucket_name, selected))
            print(f"✓ Selected from {bucket_name}: {selected['id']} ({selected['prompt_tokens']:,} tokens)")

    print(f"\n{'='*60}")
    print(f"Testing {len(test_examples)} examples across different lengths")
    print(f"{'='*60}\n")

    results = {}

    for bucket_name, example in test_examples:
        print(f"\n{'='*80}")
        print(f"BUCKET: {bucket_name} - {example['id']}")
        print(f"Tokens: {example['prompt_tokens']:,} | Repo: {example['repo']} | Language: {example['language']}")
        print(f"{'='*80}\n")

        # Test baseline
        print(f"Testing BASELINE...")
        baseline_result = await evaluator.evaluate_single_example_baseline(example)

        if baseline_result['success']:
            match_status = "✓ EXACT MATCH" if baseline_result['exact_match'] else f"✗ NO MATCH (similarity: {baseline_result['similarity']:.2%})"
            print(f"  {match_status}")
            print(f"  Time: {baseline_result['elapsed_time']:.2f}s")
            print(f"  LLM Calls: {baseline_result['llm_calls']}")
        else:
            print(f"  ✗ Failed - {baseline_result['error']}")

        # Test RLM without partitioning
        print(f"\nTesting RLM (No Partitioning)...")
        rlm_none_result = await evaluator.evaluate_single_example_rlm(
            example,
            partition_strategy=None
        )

        if rlm_none_result['success']:
            match_status = "✓ EXACT MATCH" if rlm_none_result['exact_match'] else f"✗ NO MATCH (similarity: {rlm_none_result['similarity']:.2%})"
            print(f"  {match_status}")
            print(f"  Time: {rlm_none_result['elapsed_time']:.2f}s")
            print(f"  LLM Calls: {rlm_none_result['llm_calls']}")
            print(f"  Iterations: {rlm_none_result['iterations']}")
        else:
            print(f"  ✗ Failed - {rlm_none_result['error']}")

        # Test RLM with structural partitioning
        print(f"\nTesting RLM (Structural + Embedding)...")
        rlm_structural_result = await evaluator.evaluate_single_example_rlm(
            example,
            partition_strategy="structural",
            retrieval_method="embedding"
        )

        if rlm_structural_result['success']:
            match_status = "✓ EXACT MATCH" if rlm_structural_result['exact_match'] else f"✗ NO MATCH (similarity: {rlm_structural_result['similarity']:.2%})"
            print(f"  {match_status}")
            print(f"  Time: {rlm_structural_result['elapsed_time']:.2f}s")
            print(f"  LLM Calls: {rlm_structural_result['llm_calls']}")
            print(f"  Iterations: {rlm_structural_result['iterations']}")
        else:
            print(f"  ✗ Failed - {rlm_structural_result['error']}")

        # Store results
        results[bucket_name] = {
            'example': example['id'],
            'tokens': example['prompt_tokens'],
            'baseline': baseline_result,
            'rlm_none': rlm_none_result,
            'rlm_structural': rlm_structural_result
        }

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY BY CONTEXT LENGTH")
    print(f"{'='*80}\n")

    print(f"{'Bucket':<12} {'Tokens':<10} {'Baseline':<15} {'RLM (None)':<15} {'RLM (Struct)':<15}")
    print("-" * 80)

    for bucket_name, data in results.items():
        baseline = data['baseline']
        rlm_none = data['rlm_none']
        rlm_struct = data['rlm_structural']

        baseline_status = "✓" if baseline.get('exact_match') else f"✗ ({baseline.get('similarity', 0):.0%})"
        rlm_none_status = "✓" if rlm_none.get('exact_match') else f"✗ ({rlm_none.get('similarity', 0):.0%})"
        rlm_struct_status = "✓" if rlm_struct.get('exact_match') else f"✗ ({rlm_struct.get('similarity', 0):.0%})"

        print(f"{bucket_name:<12} {data['tokens']:<10,} {baseline_status:<15} {rlm_none_status:<15} {rlm_struct_status:<15}")

    print(f"\n{'='*80}")
    print("KEY INSIGHTS")
    print(f"{'='*80}\n")

    # Check if there's a crossover point
    crossover_found = False
    for bucket_name, data in results.items():
        baseline_acc = 1 if data['baseline'].get('exact_match') else 0
        rlm_acc = 1 if data['rlm_structural'].get('exact_match') else 0

        if rlm_acc > baseline_acc:
            crossover_found = True
            print(f"✓ CROSSOVER FOUND at {bucket_name}:")
            print(f"  RLM (structural) outperforms baseline at {data['tokens']:,} tokens")
            break

    if not crossover_found:
        print("✗ No crossover point found in tested examples")
        print("  Baseline remains better or equal across all context lengths")
        print("\nPossible reasons:")
        print("  1. REPL overhead too high for exact reconstruction tasks")
        print("  2. Git diffs require sequential processing (partitioning breaks dependencies)")
        print("  3. Task mismatch: RLM optimized for exploration, not reconstruction")
        print("  4. Need to test even longer contexts (75k+) to see benefit")

    print(f"\n{'='*80}")
    print("✓ Long context test complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    # Check for API key
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("GEMINI_API_KEY")):
        print("❌ Error: No API key found!")
        print("\nPlease set your API key:")
        print("  OPENAI_API_KEY=your-key-here")
        exit(1)

    asyncio.run(main())
