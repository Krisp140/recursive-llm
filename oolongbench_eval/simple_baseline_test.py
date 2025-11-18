"""
Simple baseline test for RLM on OOLONGBench
Run this to quickly test base RLM performance on a few examples.
"""

import asyncio
import os
from dotenv import load_dotenv
from datasets import load_dataset
from rlm import RLM


async def test_baseline():
    """Test base RLM on a few OOLONGBench examples."""
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Check API key
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set GEMINI_API_KEY or OPENAI_API_KEY")
        print("\nOptions:")
        print("  1. Create a .env file with: GEMINI_API_KEY=your-key-here")
        print("  2. Or export in shell: export GEMINI_API_KEY='your-key-here'")
        return
    
    # Choose model based on available key
    if os.getenv("GEMINI_API_KEY"):
        model = "gemini/gemini-2.5-flash"  # Faster for testing
        print(f"Using Gemini: {model}")
    else:
        model = "gpt-4o-mini"
        print(f"Using OpenAI: {model}")
    
    print("\n1. Loading OOLONGBench dataset...")
    try:
        # OOLONGBench has configs: 'dnd' (full) or 'toy_dnd' (small)
        # Using toy_dnd for quick testing
        dataset = load_dataset("oolongbench/oolong-real", "toy_dnd", split="test", streaming=True)
        print("   ✓ Dataset loaded (toy_dnd config)")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print("   Trying 'dnd' config...")
        try:
            dataset = load_dataset("oolongbench/oolong-real", "dnd", split="test", streaming=True)
            print("   ✓ Dataset loaded (dnd config)")
        except Exception as e2:
            print(f"   ❌ Could not load dataset: {e2}")
            print("   Available configs: 'dnd', 'toy_dnd'")
            return
    
    print("\n2. Initializing RLM...")
    rlm = RLM(
        model=model,
        max_iterations=20,
        max_depth=5,
        temperature=0.0  # Deterministic
    )
    print("   ✓ RLM initialized")
    
    print("\n3. Running on 3 examples...")
    print("=" * 60)
    
    # Test on first 3 examples
    for i, example in enumerate(dataset, 1):
        if i > 3:  # Only 3 examples for quick test
            break
        
        # Extract fields - OOLONGBench uses: context_window_text, question, answer
        context = example.get('context_window_text', example.get('context', example.get('input', '')))
        question = example.get('question', example.get('query', ''))
        answer = example.get('answer', example.get('output', ''))
        
        if not context or not question:
            print(f"  ⚠ Skipping - missing data (context: {len(context)} chars, question: {len(question)} chars)")
            continue
        
        print(f"\nExample {i}:")
        print(f"  Context length: {len(context):,} chars")
        print(f"  Question: {question[:100]}...")
        
        try:
            import time
            start = time.time()
            
            result = await rlm.acompletion(query=question, context=context)
            
            elapsed = time.time() - start
            
            print(f"  ✓ Success!")
            print(f"    Answer: {result[:150]}...")
            print(f"    Time: {elapsed:.2f}s")
            print(f"    LLM calls: {rlm.stats['llm_calls']}")
            print(f"    Iterations: {rlm.stats['iterations']}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("✓ Baseline test complete!")
    print("\nNext steps:")
    print("  - Run full evaluation: python oolongbench_evaluation.py")
    print("  - Analyze results: python analyze_oolongbench_results.py")


if __name__ == "__main__":
    asyncio.run(test_baseline())

