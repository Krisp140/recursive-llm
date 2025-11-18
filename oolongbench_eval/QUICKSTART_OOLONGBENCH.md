# Quick Start: OOLONGBench Evaluation

TL;DR for running your RLM on OOLONGBench with partition strategies.

## Prerequisites

```bash
# Install dependencies
pip install datasets huggingface-hub matplotlib

# Set API key
export GEMINI_API_KEY="your-key-here"
```

## Step 1: Wait for Implementation

Your friend is implementing the partition strategies from `plan.md`. Once merged:

```bash
git pull origin main
pip install -e .
```

## Step 2: Run Baseline Test (5 examples)

```bash
# Test with current implementation
python oolongbench_evaluation.py
```

Expected output:
```
Loading OOLONGBench dataset...
✓ Loaded OOLONGBench successfully

Evaluating on 5 examples
============================================================

Configuration: token_unfiltered_parallel=False
============================================================

Example 1/5
  ✓ Success - Time: 12.34s, LLM calls: 3

...

✓ Saved results to oolongbench_results/token_unfiltered_parallel=False_20250118_123456.json
✓ Evaluation complete!
```

## Step 3: Enable All Strategies

Once implementation is ready, edit `oolongbench_evaluation.py`:

```python
# Line ~215: Uncomment all strategies
partition_strategies = [
    "token",
    "structural",   # ← Uncomment
    "semantic",     # ← Uncomment
    "learned"       # ← Uncomment (if implemented)
]

retrieval_methods = [
    "unfiltered",
    "regex",        # ← Uncomment
    "embedding"     # ← Uncomment
]

parallel_options = [
    False,
    True            # ← Uncomment
]

# Line ~240: Run on full dataset
max_examples=None  # ← Change from 5 to None
```

## Step 4: Run Full Evaluation

```bash
# This will take longer - tests all combinations
python oolongbench_evaluation.py
```

This tests:
- 4 partition strategies × 3 retrieval methods × 2 parallel options = **24 configurations**
- On full OOLONGBench dataset

## Step 5: Analyze Results

```bash
# Generate comparison tables, metrics, and plots
python analyze_oolongbench_results.py
```

Output:
- **Console**: Comparison tables and detailed metrics
- **CSV**: `oolongbench_results/results.csv` - spreadsheet-friendly
- **Plot**: `oolongbench_results/comparison_plot.png` - visualizations

## Key Files

| File | Purpose |
|------|---------|
| `oolongbench_evaluation.py` | Main evaluation script |
| `analyze_oolongbench_results.py` | Results analysis and visualization |
| `OOLONGBENCH_GUIDE.md` | Detailed guide with troubleshooting |
| `oolongbench_results/` | Output directory (auto-created) |

## Expected Results Directory Structure

```
oolongbench_results/
├── token_unfiltered_parallel=False_20250118_123456.json
├── semantic_embedding_parallel=True_20250118_130000.json
├── ... (one file per configuration)
├── summary_20250118_140000.json
├── results.csv
└── comparison_plot.png
```

## Quick Commands

```bash
# Test with 10 examples (faster)
python oolongbench_evaluation.py  # Edit max_examples=10

# Run full evaluation overnight
nohup python oolongbench_evaluation.py > eval.log 2>&1 &

# Analyze results
python analyze_oolongbench_results.py

# Export to CSV only
python analyze_oolongbench_results.py  # CSV auto-generated
```

## Troubleshooting

### "GEMINI_API_KEY not found"
```bash
export GEMINI_API_KEY="your-key-here"
```

### "Dataset not found"
Try alternative names in `oolongbench_evaluation.py` line ~55:
```python
dataset = load_dataset("oolongbench/OolongBench")  # Alternative
```

### Rate limit errors
Add delay in evaluation loop (line ~160):
```python
await asyncio.sleep(1)  # Add this after each example
```

### Out of memory
Reduce concurrent examples:
```python
max_examples=10  # Start small
```

## What to Expect

### Performance Metrics

Based on `plan.md` goals, you should see:

1. **Partition Strategy Impact**
   - `semantic` may be slower but more accurate
   - `token` is fastest but may miss context
   - `structural` balances both

2. **Retrieval Method Impact**
   - `embedding` more accurate but slower
   - `regex` faster but less semantic
   - `unfiltered` slowest (baseline)

3. **Parallel Processing Impact**
   - `parallel=True` should reduce latency
   - Most benefit on multi-partition queries

### Sample Output

```
CONFIGURATION COMPARISON
====================================================================
Configuration                            Success    Avg Time    Avg Calls
--------------------------------------------------------------------
token_regex_parallel=True                5/5        8.2s        4.1
semantic_embedding_parallel=True         5/5        12.5s       5.3
structural_regex_parallel=False          5/5        15.1s       4.8
...
```

## Next Steps

1. ✅ Run baseline test (5 examples)
2. ⏳ Wait for partition implementation
3. ✅ Run full evaluation (all strategies)
4. ✅ Analyze results
5. 📊 Create plots and tables for your report
6. 📝 Write up findings

## Questions?

- See `OOLONGBENCH_GUIDE.md` for detailed instructions
- See `plan.md` for implementation details
- Check issues in the repo

Good luck! 🚀

