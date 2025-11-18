"""
Quick script to check if API keys are properly loaded from .env file.
Run this to verify your .env setup before running the evaluation.
"""

import os
from dotenv import load_dotenv

print("Checking API key setup...\n")

# Load .env file
print("1. Loading .env file...")
load_dotenv()
print("   ✓ load_dotenv() called\n")

# Check for API keys
print("2. Checking environment variables:")

keys_to_check = {
    "GEMINI_API_KEY": "Gemini",
    "OPENAI_API_KEY": "OpenAI",
    "ANTHROPIC_API_KEY": "Anthropic"
}

found_keys = []

for env_var, service in keys_to_check.items():
    value = os.getenv(env_var)
    if value:
        # Show first/last few chars for verification
        masked = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
        print(f"   ✓ {env_var} = {masked}")
        found_keys.append(service)
    else:
        print(f"   ✗ {env_var} not found")

print()

if found_keys:
    print(f"✓ Success! Found API key(s) for: {', '.join(found_keys)}")
    print("\nYou can now run:")
    print("  python oolongbench_evaluation.py")
else:
    print("❌ No API keys found!")
    print("\nTo fix this:")
    print("  1. Create a .env file in the project root (C:\\Users\\rdavi\\recursive-llm\\.env)")
    print("  2. Add your API key:")
    print("     GEMINI_API_KEY=your-actual-key-here")
    print("  3. See env_example.txt for a template")
    print("  4. Run this script again to verify")

