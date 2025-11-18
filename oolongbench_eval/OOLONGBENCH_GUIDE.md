# OOLONGBench Evaluation Guide

This guide explains how to evaluate your RLM implementation on OOLONGBench with different partition strategies.

## Overview

Once your friend's implementation from `plan.md` is complete, you'll be able to:
1. Test different **partition strategies** (token, structural, semantic, learned)
2. Test different **retrieval methods** (regex, embedding, unfiltered)
3. Test **parallel vs sequential** recursive calls
4. Collect **metrics** (accuracy, latency, token usage)

## Prerequisites

### 1. Install Required Packages

```bash
# Install Hugging Face datasets library
pip install datasets huggingface-hub

# Ensure RLM dependencies are installed
pip install -e .
```

### 2. Set Up API Keys

```bash
# Set your Gemini API key
export GEMINI_API_KEY="your-key-here"

# Or add to .env file
echo "GEMINI_API_KEY=your-key-here" >> .env
```

## Step 1: Load OOLONGBench Dataset

OOLONGBench is a long-context benchmark dataset available on Hugging Face.

### Basic Loading

```python
from datasets import load_dataset

# Load the full dataset
dataset = load_dataset("oolongbench/oolong-real")

# Or load a specific subset (if available)
dataset = load_dataset("oolongbench/oolong-real", "dnd")
```

### Understanding the Dataset Structure

OOLONGBench typically contains:
- **context**: Long document text (can be 132k+ tokens)
- **question**: Query about the context
- **answer**: Ground truth answer

Example:
```python
example = dataset['test'][0]
print(f"Context length: {len(example['context'])} chars")
print(f"Question: {example['question']}")
print(f"Answer: {example['answer']}")
```

## Step 2: Wait for Implementation to Be Merged

Your friend is implementing the features in `plan.md`. Once merged, the `RLM` class will support:

```python
from rlm import RLM

rlm = RLM(
    model="gemini/gemini-2.5-pro",
    recursive_model="gemini/gemini-2.5-flash",
    
    # New parameters from plan.md:
    partition_strategy="semantic",      # token, structural, semantic, learned
    retrieval_method="embedding",       # regex, embedding, unfiltered
    parallel_subqueries=True,           # True/False
    max_parallel_subqueries=4           # Max concurrent recursive calls
)
```

## Step 3: Run Evaluation Script

I've created `oolongbench_evaluation.py` for you. Here's how to use it:

### Quick Start (5 examples)

```bash
python oolongbench_evaluation.py
```

This will:
- Load OOLONGBench from Hugging Face
- Test with baseline configuration
- Run on 5 examples
- Save results to `oolongbench_results/`

### Full Evaluation

Once the implementation is ready, edit `oolongbench_evaluation.py` and uncomment the strategies:

```python
partition_strategies = [
    "token",        # Baseline
    "structural",   # Uncomment when ready
    "semantic",     # Uncomment when ready
    "learned"       # Uncomment when ready
]

retrieval_methods = [
    "unfiltered",   # No retrieval
    "regex",        # Uncomment when ready
    "embedding"     # Uncomment when ready
]

parallel_options = [
    False,          # Sequential
    True            # Uncomment when ready
]
```

Then run on full dataset:

```python
# In main() function, change:
max_examples=None  # Run on all examples
```

## Step 4: Analyze Results

### Output Files

The script generates:

1. **Individual configuration results**: `{config_name}_{timestamp}.json`
   - Contains detailed results for each example
   - Includes answers, ground truth, timings, token counts

2. **Summary report**: `summary_{timestamp}.json`
   - Aggregated metrics across configurations
   - Average time, LLM calls, iterations per configuration

### Example Results Structure

```json
{
  "success": true,
  "answer": "Model's answer...",
  "ground_truth": "Correct answer...",
  "context_length": 150000,
  "llm_calls": 5,
  "iterations": 4,
  "elapsed_time": 12.5,
  "partition_strategy": "semantic",
  "retrieval_method": "embedding",
  "parallel_subqueries": true
}
```

### Summary Metrics

```json
{
  "configurations": {
    "semantic_embedding_parallel=True": {
      "num_examples": 100,
      "avg_time": 15.2,
      "avg_llm_calls": 6.3,
      "avg_iterations": 4.1,
      "total_time": 1520.0
    }
  }
}
```

## Step 5: Compare Strategies

### Key Metrics to Compare

1. **Accuracy**
   - Did the model produce correct answers?
   - Manual comparison with ground truth (or use LLM-as-judge)

2. **Efficiency**
   - Average time per query
   - Average LLM calls per query
   - Total token usage

3. **Scalability**
   - Performance on different context lengths
   - Impact of parallel processing

### Analysis Script

```python
import json
from pathlib import Path

def analyze_results(results_dir="oolongbench_results"):
    """Analyze and compare results across configurations."""
    
    results_dir = Path(results_dir)
    
    # Load all result files
    configs = {}
    for file in results_dir.glob("*.json"):
        if file.name.startswith("summary"):
            continue
        
        with open(file) as f:
            results = json.load(f)
        
        # Extract config name from filename
        config = file.stem.split('_')[0:3]
        config_name = '_'.join(config)
        
        if config_name not in configs:
            configs[config_name] = []
        configs[config_name].extend(results)
    
    # Compare metrics
    print("Configuration Comparison:")
    print("-" * 80)
    
    for config_name, results in configs.items():
        successful = [r for r in results if r['success']]
        
        if not successful:
            continue
        
        avg_time = sum(r['elapsed_time'] for r in successful) / len(successful)
        avg_calls = sum(r['llm_calls'] for r in successful) / len(successful)
        
        print(f"{config_name}:")
        print(f"  Success rate: {len(successful)}/{len(results)}")
        print(f"  Avg time: {avg_time:.2f}s")
        print(f"  Avg LLM calls: {avg_calls:.1f}")
        print()

# Run analysis
analyze_results()
```

## Expected Findings (From Plan.md)

Based on the project goals, you should investigate:

### Questions to Answer

1. **Does smarter partitioning improve accuracy at fixed cost?**
   - Compare token vs structural vs semantic vs learned

2. **Does embedding retrieval help compared to regex on long contexts?**
   - Compare retrieval methods

3. **How much speedup do we get from parallel sub-queries?**
   - Compare parallel=False vs parallel=True

4. **Does the learned partition strategy outperform fixed ones?**
   - If implemented, compare learned vs hand-designed strategies

## Troubleshooting

### Issue: Dataset Not Found

If `oolongbench/oolong-real` doesn't exist, try:
- `oolongbench/OolongBench`
- Check Hugging Face for the exact dataset name
- Look for alternative long-context benchmarks

### Issue: Rate Limits

Gemini has rate limits. To handle:
```python
# Add rate limiting in the evaluation script
import time

async def evaluate_with_rate_limit(self, example, ...):
    result = await self.evaluate_single_example(example, ...)
    await asyncio.sleep(1)  # Wait 1 second between calls
    return result
```

### Issue: Out of Memory

For very long contexts:
- Start with fewer examples (`max_examples=10`)
- Test one configuration at a time
- Monitor memory usage

## Advanced: Custom Evaluation Metrics

### Add Accuracy Scoring

```python
def compute_accuracy(prediction: str, ground_truth: str) -> float:
    """
    Compute accuracy score between prediction and ground truth.
    
    Options:
    1. Exact match
    2. F1 score (token overlap)
    3. LLM-as-judge
    """
    # Exact match
    if prediction.strip().lower() == ground_truth.strip().lower():
        return 1.0
    
    # Or use LLM-as-judge
    # judge_prompt = f"Does '{prediction}' correctly answer the question? Ground truth: '{ground_truth}'"
    # ...
    
    return 0.0
```

### Track Token Usage

If using OpenAI or other providers that return token counts:

```python
# In evaluate_single_example()
response = await litellm.acompletion(...)
tokens_used = response.usage.total_tokens

result['total_tokens'] = tokens_used
```

## Next Steps

1. **Wait for implementation**: Your friend completes the features in `plan.md`
2. **Pull from git**: Get the latest code
3. **Test baseline**: Run with `partition_strategy="token"` first
4. **Run full sweep**: Test all combinations
5. **Analyze results**: Compare metrics and generate plots
6. **Write report**: Document findings for your class project

## References

- **Plan Document**: See `plan.md` for implementation details
- **RLM Paper**: https://alexzhang13.github.io/blog/2025/rlm/
- **Hugging Face Datasets**: https://huggingface.co/docs/datasets/
- **LiteLLM Docs**: https://docs.litellm.ai/

Good luck with your evaluation! 🚀

