# LoCoDiff Benchmark Evaluation for RLM

This directory contains the evaluation infrastructure for benchmarking RLM (Recursive Language Model) on the **LoCoDiff dataset**, which tests LLMs' ability to understand git history and reconstruct code from diffs.

## Overview

LoCoDiff is a challenging benchmark that provides git commit histories (with diffs) and asks models to reconstruct the current state of a file. The key insight from LoCoDiff is that model accuracy degrades significantly as context length increases, dropping to <50% accuracy at just 25k tokens.

This evaluation suite tests whether RLM's partitioning and retrieval strategies can improve performance on long git histories.

## Dataset

The LoCoDiff dataset consists of:
- **200 files** from 5 real-world repositories
- **Repositories**: Aider (Python), Ghostty (Zig), tldraw (TypeScript), Qdrant (Rust), React (JavaScript)
- **Prompt lengths**: 0-75k tokens
- **Task**: Reconstruct exact file state from git history
- **Metric**: Exact match accuracy (no partial credit)

## Setup

### 1. Install Dependencies

```bash
# Install project dependencies
pip install -e .

# Optional: Install matplotlib for visualizations
pip install matplotlib
```

### 2. Download Dataset

**Important:** The dataset is NOT included in the repository (323MB). Use the download script:

```bash
# Run the download script
bash locodiff/scripts/download_dataset.sh
```

This will:
- Clone the LoCoDiff-bench repository to `external/`
- Copy the dataset (200 files, ~323MB) to `locodiff/locodiff_data/`
- Verify the download with statistics

**Manual download** (alternative):
```bash
# Clone LoCoDiff-bench repository
git clone https://github.com/AbanteAI/LoCoDiff-bench external/LoCoDiff-bench

# Copy dataset to project
cp -r external/LoCoDiff-bench/locodiff-250425 locodiff/locodiff_data/
```

### 3. Configure API Keys

Create a `.env` file in the project root:

```bash
# For OpenAI models
OPENAI_API_KEY=your-openai-key-here

# For Anthropic models
ANTHROPIC_API_KEY=your-anthropic-key-here

# For Google Gemini models
GEMINI_API_KEY=your-gemini-key-here
```

### 4. Verify Setup

Run the setup verification script:

```bash
python locodiff_eval/test_setup.py
```

This will check:
- ✓ Dataset is downloaded and accessible
- ✓ API keys are configured
- ✓ Dependencies are installed
- ✓ RLM can be imported
- ✓ Dataset can be loaded

## Quick Start

### Run a Quick Test

Test on 2 examples (shortest prompts) to verify everything works:

```bash
python locodiff_eval/baseline_test.py
```

This will:
1. Load the 2 shortest prompts from the dataset
2. Test baseline (direct LLM call)
3. Test RLM without partitioning
4. Show exact match results

### Run Full Evaluation

Edit `locodiff_eval/locodiff_evaluation.py` to configure:

```python
# In main() function:

# Choose models
evaluator = LoCoDiffEvaluator(
    model="gemini/gemini-2.5-pro",           # Or "gpt-4o", "claude-sonnet-4"
    recursive_model="gemini/gemini-2.5-flash"  # Cheaper model for recursive calls
)

# Choose configurations to test
partition_strategies = [
    None,          # No partitioning (traditional RLM)
    "token",       # Fixed-size token chunks
    "structural",  # Paragraph/heading boundaries
    "semantic"     # Topic-based (uses embeddings)
]

retrieval_methods = [
    "none",        # No retrieval
    "unfiltered",  # First k partitions
    "regex",       # Keyword matching
    "embedding"    # Semantic similarity
]

# Number of examples to evaluate
max_examples=10  # Change to None for full 200 examples
```

Then run:

```bash
python locodiff_eval/locodiff_evaluation.py
```

## Analyzing Results

After evaluation, analyze results:

```bash
python locodiff_eval/analyze_locodiff_results.py
```

This generates:
- **Comparison tables**: Exact match accuracy, time, LLM calls
- **RLM vs Baseline comparison**: Side-by-side metrics
- **By context length**: Accuracy degradation as context grows (key LoCoDiff insight)
- **By language**: Performance on Python, TypeScript, Rust, etc.
- **By repository**: Performance on different codebases
- **CSV export**: `locodiff_results/results.csv`
- **Visualizations**:
  - `locodiff_results/plots/accuracy_vs_context.png`
  - `locodiff_results/plots/comparison_plot.png`

## Results Structure

```
locodiff_results/
├── baseline_20250119_123456.json              # Baseline (direct LLM) results
├── rlm_token_unfiltered_parallel=False_20250119_123456.json
├── rlm_structural_embedding_parallel=False_20250119_123456.json
├── summary_20250119_123456.json               # Aggregated metrics
├── results.csv                                # CSV export
└── plots/
    ├── accuracy_vs_context.png                # Key LoCoDiff visualization
    └── comparison_plot.png                    # Configuration comparison
```

## Understanding the Metrics

### Primary Metric: Exact Match Accuracy

LoCoDiff uses **exact match** (no partial credit). A prediction is correct only if it matches the expected output character-for-character (after normalization).

```python
exact_match_accuracy = exact_matches / total_examples
```

### Secondary Metrics

- **Similarity Score**: Sequence similarity (0-1) for analysis
- **Avg Time**: Average processing time per example
- **Avg LLM Calls**: Average number of LLM API calls
- **Diff Lines**: Number of different lines (for debugging)

### Context Length Analysis

The most important LoCoDiff insight is how accuracy degrades with context length:

```
Context Length | Baseline Accuracy | RLM Accuracy | Improvement
0-10k tokens   | 95%              | 96%          | +1%
10k-25k tokens | 65%              | 75%          | +10%
25k-50k tokens | 40%              | 60%          | +20%
50k+ tokens    | 20%              | 45%          | +25%
```

*(Example numbers - run evaluation to get actual results)*

## Research Questions

This benchmark helps answer:

1. **Does RLM help on git history tasks?**
   - Compare RLM exact match % vs baseline

2. **Which partition strategy works best for code diffs?**
   - Structural vs semantic vs token on git commit data

3. **Does retrieval help with long git histories?**
   - Embedding vs regex vs unfiltered

4. **How does RLM handle long contexts?**
   - Accuracy at 10k, 25k, 50k+ tokens

5. **Language-specific performance**
   - Which languages benefit most from RLM partitioning?

## Configuration Options

### Partition Strategies

- **None**: No partitioning (traditional RLM REPL loop)
- **token**: Fixed-size chunks with overlap
- **structural**: Split on paragraph/heading boundaries
- **semantic**: Topic-based clustering using embeddings

### Retrieval Methods

- **none**: No retrieval (use all context)
- **unfiltered**: Take first k partitions
- **regex**: Keyword-based relevance scoring
- **embedding**: Semantic similarity to query

### Parameters

```python
# In RLM initialization
max_partition_tokens=4000       # Max tokens per partition
partition_overlap_tokens=200    # Overlap between partitions
top_k=5                        # Number of partitions to retrieve
parallel_subqueries=False      # Parallel processing (experimental)
```

## Example Output

```
============================================================
LOCODIFF RESULTS ANALYSIS
============================================================

Configuration                                      Success    Exact Match  Avg Time    Avg Calls
----------------------------------------------------------------------------------------------------
baseline                                          10/10      4/10 (40.0%) 12.34s      1.0
rlm_none_none_parallel=False                      10/10      5/10 (50.0%) 18.56s      3.2
rlm_token_unfiltered_parallel=False               10/10      6/10 (60.0%) 15.23s      2.8
rlm_structural_embedding_parallel=False           10/10      7/10 (70.0%) 16.78s      3.1

RLM vs BASELINE COMPARISON
----------------------------------------------------------------------------------------------------
BASELINE:
  Exact Match Accuracy: 40.0% (4/10)
  Avg Time: 12.34s
  Avg LLM Calls: 1.0

Configuration                                      Accuracy       vs Baseline    Avg Time    Avg Calls
rlm_structural_embedding_parallel=False           70.0%          +30.0%         16.78s      3.1
rlm_token_unfiltered_parallel=False               60.0%          +20.0%         15.23s      2.8
rlm_none_none_parallel=False                      50.0%          +10.0%         18.56s      3.2
```

## Troubleshooting

### Dataset Not Found

```bash
# Re-download dataset
git clone https://github.com/AbanteAI/LoCoDiff-bench external/LoCoDiff-bench
cp -r external/LoCoDiff-bench/locodiff-250425 locodiff_data/
```

### API Key Errors

```bash
# Verify API keys are set
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('OPENAI_API_KEY:', 'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET')"
```

### Import Errors

```bash
# Reinstall dependencies
pip install -e .
```

### Out of Memory

For very long contexts (50k+ tokens), you may need to:
- Reduce `max_partition_tokens`
- Use a model with larger context window
- Filter to shorter examples

## Advanced Usage

### Custom Model Configuration

```python
evaluator = LoCoDiffEvaluator(
    model="anthropic/claude-sonnet-4",  # Any LiteLLM-supported model
    recursive_model="anthropic/claude-haiku-4",
    output_dir="locodiff_results/custom"
)
```

### Filter by Context Length

```python
# Load dataset
examples = evaluator.load_locodiff_dataset()

# Filter to specific token range
filtered = [ex for ex in examples if 10000 <= ex['prompt_tokens'] < 25000]

# Evaluate on filtered set
results = await evaluator.evaluate_dataset(filtered, ...)
```

### Evaluate Specific Files

```python
# Load dataset
examples = evaluator.load_locodiff_dataset()

# Filter by repo or language
python_examples = [ex for ex in examples if ex['language'] == 'Python']
aider_examples = [ex for ex in examples if ex['repo'] == 'aider']

# Evaluate
results = await evaluator.evaluate_dataset(aider_examples, ...)
```

## Files in This Directory

- `locodiff_evaluation.py` - Main evaluation script
- `analyze_locodiff_results.py` - Results analysis and visualization
- `test_setup.py` - Setup verification
- `baseline_test.py` - Quick sanity check (2 examples)
- `README.md` - This file

## References

- **LoCoDiff Benchmark**: https://abanteai.github.io/LoCoDiff-bench/
- **LoCoDiff GitHub**: https://github.com/AbanteAI/LoCoDiff-bench
- **RLM Paper**: "Recursive Language Models" (Zhang & Khattab, MIT 2025)

## Support

For issues or questions:
1. Check this README
2. Run `python locodiff_eval/test_setup.py`
3. Try `python locodiff_eval/baseline_test.py` for a quick test
4. Check the main project README

---

**Happy Benchmarking! 🚀**
