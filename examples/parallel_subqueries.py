"""Example demonstrating parallel sub-query execution (Phase 4).

This example shows how to use parallel_subqueries to speed up processing
of partitioned long documents by making recursive calls in parallel.
"""

import time
from rlm import RLM


def generate_multi_section_document():
    """Generate a document with multiple independent sections."""
    sections = []

    # Create 5 independent sections with different topics
    topics = [
        ("Machine Learning", "neural networks, deep learning, training algorithms"),
        ("Climate Science", "global warming, carbon emissions, renewable energy"),
        ("Economics", "inflation, GDP growth, monetary policy"),
        ("Space Exploration", "Mars missions, satellite technology, astronaut training"),
        ("Medicine", "vaccine development, clinical trials, disease prevention"),
    ]

    for i, (topic, keywords) in enumerate(topics, 1):
        section = f"""
Section {i}: {topic}

This section covers {topic.lower()} in depth. Key areas include {keywords}.

Major findings:
- Finding A for {topic}: Important result with statistical significance
- Finding B for {topic}: Breakthrough discovery in the field
- Finding C for {topic}: Novel approach to longstanding problem

The research in {topic} has progressed significantly over the past decade.
Current studies focus on {keywords.split(',')[0]} and related areas.

Statistics:
- Growth rate: {i * 15}%
- Investment: ${i * 5} billion
- Research papers published: {i * 1000}

Future outlook:
The field of {topic} is expected to continue rapid advancement.
Key challenges include funding, regulation, and public acceptance.

""" + f"Additional detailed analysis of {topic}. " * 150
        sections.append(section)

    return "\n\n".join(sections)


def main():
    """Compare sequential vs parallel sub-query execution."""
    print("=" * 80)
    print("Phase 4: Parallel Sub-Queries Demonstration")
    print("=" * 80)
    print()

    # Generate document
    print("Generating multi-section document...")
    document = generate_multi_section_document()
    print(f"Document size: {len(document):,} characters (~{len(document) // 4:,} tokens)")
    print()

    query = "What are the growth rates mentioned across all sections?"

    # Test 1: Sequential execution (baseline)
    print("=" * 80)
    print("Test 1: Sequential Execution (parallel_subqueries=False)")
    print("=" * 80)

    rlm_sequential = RLM(
        model="gpt-5-mini",
        partition_strategy="structural",  # Split on section boundaries
        retrieval_method="unfiltered",     # Process all partitions
        parallel_subqueries=False,         # Sequential processing
        max_parallel_subqueries=5,
        max_partition_tokens=2000,
        temperature=0.3
    )

    start_time = time.time()
    try:
        result_sequential = rlm_sequential.completion(query, document)
        sequential_time = time.time() - start_time

        print(f"\nAnswer: {result_sequential}")
        print(f"\nPerformance:")
        print(f"  - Time: {sequential_time:.2f}s")
        print(f"  - LLM calls: {rlm_sequential.stats['llm_calls']}")
        print(f"  - Child LLM calls: {rlm_sequential.stats['child_llm_calls']}")
        print(f"  - Iterations: {rlm_sequential.stats['iterations']}")
    except Exception as e:
        print(f"Error: {e}")
        sequential_time = None

    print()

    # Test 2: Parallel execution
    print("=" * 80)
    print("Test 2: Parallel Execution (parallel_subqueries=True)")
    print("=" * 80)

    rlm_parallel = RLM(
        model="gpt-5-mini",
        partition_strategy="structural",
        retrieval_method="unfiltered",
        parallel_subqueries=True,          # Parallel processing!
        max_parallel_subqueries=5,
        max_partition_tokens=2000,
        temperature=0.3
    )

    start_time = time.time()
    try:
        result_parallel = rlm_parallel.completion(query, document)
        parallel_time = time.time() - start_time

        print(f"\nAnswer: {result_parallel}")
        print(f"\nPerformance:")
        print(f"  - Time: {parallel_time:.2f}s")
        print(f"  - LLM calls: {rlm_parallel.stats['llm_calls']}")
        print(f"  - Child LLM calls: {rlm_parallel.stats['child_llm_calls']}")
        print(f"  - Iterations: {rlm_parallel.stats['iterations']}")
    except Exception as e:
        print(f"Error: {e}")
        parallel_time = None

    print()

    # Compare results
    if sequential_time and parallel_time:
        print("=" * 80)
        print("Comparison")
        print("=" * 80)
        speedup = sequential_time / parallel_time
        print(f"Sequential time: {sequential_time:.2f}s")
        print(f"Parallel time:   {parallel_time:.2f}s")
        print(f"Speedup:         {speedup:.2f}x")
        print()

        if speedup > 1:
            print(f"✓ Parallel execution is {speedup:.1f}x faster!")
        else:
            print("Note: Speedup varies based on network latency and LLM response times.")

    print()
    print("=" * 80)
    print("Key Takeaways:")
    print("=" * 80)
    print("1. Parallel execution speeds up processing when multiple partitions are")
    print("   independent and can be processed simultaneously.")
    print("2. The speedup depends on:")
    print("   - Number of partitions processed in parallel")
    print("   - Network latency to LLM API")
    print("   - LLM response generation time")
    print("3. Both sequential and parallel produce the same results, but parallel")
    print("   is faster for long documents with multiple partitions.")
    print("4. Use parallel_subqueries=True when:")
    print("   - Processing long documents (>10k tokens)")
    print("   - Multiple partitions need to be analyzed")
    print("   - Latency is more important than token cost")


if __name__ == "__main__":
    # This example demonstrates Phase 4: Parallel Sub-Queries
    # which speeds up processing of partitioned long contexts
    main()
