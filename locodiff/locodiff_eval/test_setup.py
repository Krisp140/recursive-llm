"""
Setup verification script for LoCoDiff evaluation.

This script checks that:
- LoCoDiff dataset is downloaded and accessible
- API keys are configured
- RLM can be imported
- Dependencies are installed
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path so we can import locodiff_eval
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_dataset():
    """Check if LoCoDiff dataset is available."""
    print("\n" + "="*60)
    print("Checking LoCoDiff Dataset...")
    print("="*60)

    dataset_dir = Path("locodiff_data/locodiff-250425")
    prompts_dir = dataset_dir / "prompts"

    if not dataset_dir.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        print("\nTo download the dataset:")
        print("  1. Clone LoCoDiff-bench:")
        print("     git clone https://github.com/AbanteAI/LoCoDiff-bench external/LoCoDiff-bench")
        print("  2. Copy dataset:")
        print("     cp -r external/LoCoDiff-bench/locodiff-250425 locodiff_data/")
        return False

    if not prompts_dir.exists():
        print(f"❌ Prompts directory not found: {prompts_dir}")
        return False

    # Count prompt files
    prompt_files = list(prompts_dir.glob("*_prompt.txt"))
    expected_files = list(prompts_dir.glob("*_expectedoutput.txt"))

    print(f"✓ Dataset directory found: {dataset_dir}")
    print(f"✓ Found {len(prompt_files)} prompt files")
    print(f"✓ Found {len(expected_files)} expected output files")

    if len(prompt_files) != len(expected_files):
        print(f"⚠ Warning: Mismatch between prompt and expected output files")

    return True


def check_api_keys():
    """Check if API keys are configured."""
    print("\n" + "="*60)
    print("Checking API Keys...")
    print("="*60)

    load_dotenv()

    has_key = False

    # Check for OpenAI
    if os.getenv("OPENAI_API_KEY"):
        print("✓ OPENAI_API_KEY found")
        has_key = True
    else:
        print("❌ OPENAI_API_KEY not found")

    # Check for Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        print("✓ ANTHROPIC_API_KEY found")
        has_key = True
    else:
        print("❌ ANTHROPIC_API_KEY not found")

    # Check for Gemini
    if os.getenv("GEMINI_API_KEY"):
        print("✓ GEMINI_API_KEY found")
        has_key = True
    else:
        print("❌ GEMINI_API_KEY not found")

    if not has_key:
        print("\n❌ No API keys found!")
        print("\nPlease set at least one API key in your .env file:")
        print("  OPENAI_API_KEY=your-key-here")
        print("  or")
        print("  ANTHROPIC_API_KEY=your-key-here")
        print("  or")
        print("  GEMINI_API_KEY=your-key-here")
        return False

    print("\n✓ At least one API key is configured")
    return True


def check_dependencies():
    """Check if required dependencies are installed."""
    print("\n" + "="*60)
    print("Checking Dependencies...")
    print("="*60)

    required_packages = [
        ('rlm', 'RLM'),
        ('litellm', 'LiteLLM'),
        ('dotenv', 'python-dotenv'),
        ('tiktoken', 'tiktoken'),
    ]

    optional_packages = [
        ('matplotlib.pyplot', 'matplotlib (for plots)'),
    ]

    all_required_installed = True

    for module_name, package_name in required_packages:
        try:
            __import__(module_name.split('.')[0])
            print(f"✓ {package_name} installed")
        except ImportError:
            print(f"❌ {package_name} not installed")
            all_required_installed = False

    print("\nOptional dependencies:")
    for module_name, package_name in optional_packages:
        try:
            __import__(module_name.split('.')[0])
            print(f"✓ {package_name} installed")
        except ImportError:
            print(f"⚠ {package_name} not installed (optional)")

    if not all_required_installed:
        print("\n❌ Some required dependencies are missing!")
        print("\nInstall missing dependencies:")
        print("  pip install -e .")
        return False

    print("\n✓ All required dependencies installed")
    return True


def test_rlm_import():
    """Test that RLM can be imported and initialized."""
    print("\n" + "="*60)
    print("Testing RLM Import...")
    print("="*60)

    try:
        from rlm import RLM
        print("✓ RLM imported successfully")

        # Try to initialize (without calling)
        rlm = RLM(model="gpt-4o-mini", max_iterations=1)
        print("✓ RLM initialized successfully")

        return True
    except Exception as e:
        print(f"❌ Error with RLM: {e}")
        return False


def test_dataset_loading():
    """Test loading a sample from the dataset."""
    print("\n" + "="*60)
    print("Testing Dataset Loading...")
    print("="*60)

    try:
        from locodiff_eval.locodiff_evaluation import LoCoDiffEvaluator

        evaluator = LoCoDiffEvaluator()
        examples = evaluator.load_locodiff_dataset()

        if len(examples) > 0:
            print(f"\n✓ Successfully loaded {len(examples)} examples")

            # Show sample
            sample = examples[0]
            print(f"\nSample example:")
            print(f"  ID: {sample['id']}")
            print(f"  Repo: {sample['repo']}")
            print(f"  Language: {sample['language']}")
            print(f"  Prompt tokens: {sample['prompt_tokens']:,}")
            print(f"  Prompt preview: {sample['prompt'][:200]}...")

            return True
        else:
            print("❌ No examples loaded")
            return False

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all setup checks."""
    print("\n" + "="*60)
    print("LOCODIFF EVALUATION SETUP VERIFICATION")
    print("="*60)

    results = []

    # Run checks
    results.append(("Dataset", check_dataset()))
    results.append(("API Keys", check_api_keys()))
    results.append(("Dependencies", check_dependencies()))
    results.append(("RLM Import", test_rlm_import()))
    results.append(("Dataset Loading", test_dataset_loading()))

    # Print summary
    print("\n" + "="*60)
    print("SETUP VERIFICATION SUMMARY")
    print("="*60 + "\n")

    all_passed = True
    for check_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{check_name:<20} {status}")
        if not passed:
            all_passed = False

    print("\n" + "="*60)

    if all_passed:
        print("✓ All checks passed! You're ready to run the evaluation.")
        print("\nNext steps:")
        print("  1. Quick test: python locodiff_eval/baseline_test.py")
        print("  2. Full evaluation: python locodiff_eval/locodiff_evaluation.py")
        return 0
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
