"""
Test script to verify setup before running OOLONGBench evaluation.

This script checks:
1. API keys are configured
2. Dependencies are installed
3. OOLONGBench dataset is accessible
4. RLM can run a simple query
"""

import os
import sys
from dotenv import load_dotenv


def test_api_keys():
    """Test that API keys are configured."""
    print("\n1. Testing API Keys...")
    print("-" * 60)
    
    keys_to_check = [
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY"
    ]
    
    found_keys = []
    for key in keys_to_check:
        if os.getenv(key):
            found_keys.append(key)
            print(f"  ✓ {key} is set")
    
    if not found_keys:
        print("  ❌ No API keys found!")
        print("\n  Please set at least one API key:")
        print("    export GEMINI_API_KEY='your-key-here'")
        return False
    
    print(f"\n  ✓ Found {len(found_keys)} API key(s)")
    return True


def test_dependencies():
    """Test that required dependencies are installed."""
    print("\n2. Testing Dependencies...")
    print("-" * 60)
    
    required_packages = [
        ("datasets", "datasets"),
        ("huggingface_hub", "huggingface-hub"),
        ("matplotlib", "matplotlib"),
        ("rlm", "recursive-llm (pip install -e .)"),
    ]
    
    all_installed = True
    
    for module_name, package_name in required_packages:
        try:
            __import__(module_name)
            print(f"  ✓ {package_name} is installed")
        except ImportError:
            print(f"  ❌ {package_name} is NOT installed")
            print(f"     Install with: pip install {package_name}")
            all_installed = False
    
    if all_installed:
        print("\n  ✓ All required packages are installed")
    else:
        print("\n  ❌ Some packages are missing")
        print("     Install all with: pip install -r requirements_oolongbench.txt")
    
    return all_installed


def test_dataset_access():
    """Test that OOLONGBench dataset is accessible."""
    print("\n3. Testing OOLONGBench Dataset Access...")
    print("-" * 60)
    
    try:
        from datasets import load_dataset
        
        print("  Attempting to load OOLONGBench...")
        
        # Try loading just the dataset info (faster than full download)
        try:
            dataset = load_dataset("oolongbench/oolong-real", split="train", streaming=True)
            print("  ✓ Successfully accessed oolongbench/oolong-real")
            
            # Get first example to check structure
            first_example = next(iter(dataset))
            print(f"\n  Dataset structure:")
            for key in first_example.keys():
                value_preview = str(first_example[key])[:50]
                print(f"    - {key}: {value_preview}...")
            
            return True
            
        except Exception as e1:
            print(f"  ⚠ Could not load oolongbench/oolong-real: {e1}")
            print("  Trying alternative name...")
            
            try:
                dataset = load_dataset("oolongbench/OolongBench", split="train", streaming=True)
                print("  ✓ Successfully accessed oolongbench/OolongBench")
                return True
            except Exception as e2:
                print(f"  ❌ Could not load dataset: {e2}")
                print("\n  Possible solutions:")
                print("    1. Check dataset name on Hugging Face")
                print("    2. Verify internet connection")
                print("    3. Try: huggingface-cli login")
                return False
    
    except ImportError:
        print("  ❌ 'datasets' package not installed")
        print("     Install with: pip install datasets")
        return False


def test_rlm_basic():
    """Test that RLM can run a basic query."""
    print("\n4. Testing RLM Basic Functionality...")
    print("-" * 60)
    
    try:
        from rlm import RLM
        
        # Determine which model to use based on available API keys
        model = None
        if os.getenv("GEMINI_API_KEY"):
            model = "gemini/gemini-2.0-flash-exp"
            print(f"  Using Gemini model: {model}")
        elif os.getenv("OPENAI_API_KEY"):
            model = "gpt-3.5-turbo"
            print(f"  Using OpenAI model: {model}")
        elif os.getenv("ANTHROPIC_API_KEY"):
            model = "claude-3-haiku-20240307"
            print(f"  Using Anthropic model: {model}")
        else:
            print("  ⚠ No API key found, skipping RLM test")
            return True
        
        print("  Running simple test query...")
        
        rlm = RLM(model=model, max_iterations=5, temperature=0.0)
        
        test_context = """
        Sales Report Q1 2024:
        - Product A: $50,000
        - Product B: $75,000
        - Product C: $25,000
        Total: $150,000
        """
        
        test_query = "What was the total sales in Q1 2024?"
        
        result = rlm.completion(query=test_query, context=test_context)
        
        print(f"\n  Query: {test_query}")
        print(f"  Answer: {result}")
        print(f"  Stats: {rlm.stats}")
        
        if result:
            print("\n  ✓ RLM is working correctly!")
            return True
        else:
            print("\n  ⚠ RLM returned empty result")
            return False
    
    except ImportError:
        print("  ❌ Could not import RLM")
        print("     Install with: pip install -e .")
        return False
    except Exception as e:
        print(f"  ❌ Error running RLM: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all setup tests."""
    # Load environment variables from .env file
    load_dotenv()
    
    print("="*60)
    print("OOLONGBENCH EVALUATION SETUP TEST")
    print("="*60)
    
    results = {
        "API Keys": test_api_keys(),
        "Dependencies": test_dependencies(),
        "Dataset Access": test_dataset_access(),
        "RLM Functionality": test_rlm_basic()
    }
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    
    if all_passed:
        print("✓ All tests passed! You're ready to run the evaluation.")
        print("\nNext steps:")
        print("  1. python oolongbench_evaluation.py  # Run evaluation")
        print("  2. python analyze_oolongbench_results.py  # Analyze results")
    else:
        print("❌ Some tests failed. Please fix the issues above before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()

