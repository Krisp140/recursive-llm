# RLM Partition Strategy Evaluation on OOLONGBench

This directory contains scripts to evaluate different RLM partition strategies on the OOLONGBench long-context benchmark and generate comparison figures.

## What We're Doing

We're comparing different **context partitioning strategies** for Recursive Language Models (RLM) to understand which approach works best for long-context question answering. The evaluation tests:

### Partition Strategies
1. **Token** - Fixed-size chunks (baseline)
2. **Structural** - Splits on paragraph/heading boundaries
3. **Semantic** - Splits based on topic changes (using embeddings)

### Retrieval Methods
1. **Unfiltered** - Use all partitions (baseline)
2. **Regex** - Select partitions by keyword matching
3. **Embedding** - Select partitions by semantic similarity

This creates **9 total configurations** to compare (3 strategies × 3 retrieval methods).

### Metrics
- **F1 Score** - Token overlap between prediction and ground truth
- **Exact Match** - Exact string match accuracy
- **Latency** - Time per query
- **LLM Calls** - Number of API calls per query

## Files

- **`oolongbench_evaluation.py`** - Main evaluation script that runs RLM on OOLONGBench
- **`generate_oolongbench_figure.py`** - Generates comparison figures from results
- **`requirements_oolongbench.txt`** - Python dependencies
- **`oolongbench_results/`** - Output directory (auto-created)

## Setup

### 1. Install Dependencies

```bash
cd oolongbench_eval
pip install -r requirements_oolongbench.txt
```

### 2. Set API Key

Create a `.env` file in the **project root** (not in oolongbench_eval/):

```bash
# In C:\Users\rdavi\recursive-llm\.env
OPENAI_API_KEY=your-openai-api-key-here
```

Or export it in your shell:

```bash
export OPENAI_API_KEY="your-key-here"
```

## Usage

### Step 1: Run Evaluation

This runs all 9 partition/retrieval configurations on the OOLONGBench dataset:

```bash
python oolongbench_evaluation.py
```

**What it does:**
- Loads OOLONGBench toy_dnd dataset (480 test examples)
- Tests 5 examples per configuration (adjustable via `max_examples` parameter)
- Uses **GPT-5 Mini** for all LLM calls
- Saves detailed results to `oolongbench_results/*.json`

**Output:**
```
Loading OOLONGBench dataset...
✓ Loaded OOLONGBench successfully

============================================================
Configuration: token_unfiltered_parallel=False
============================================================
Example 1/5
  ✓ Success - Time: 240s, LLM calls: 12
...

✓ Evaluation complete!
Results saved to oolongbench_results
```

**Configuration:**
- Default: 5 examples, toy_dnd dataset, GPT-5 Mini
- To test more examples: Edit line 356, change `max_examples=5` to desired number
- To use full dataset: Edit line 309, change `config="toy_dnd"` to `config="dnd"`

### Step 2: Generate Figures

Once evaluation is complete, generate comparison plots:

```bash
python generate_oolongbench_figure.py
```

**Output:**
- `oolongbench_results/performance_comparison.png` - F1 Score and Exact Match by strategy
- `oolongbench_results/latency_comparison.png` - Average latency by strategy

## Understanding Results

### Result Files

Each configuration produces a JSON file with detailed metrics:

```json
{
  "success": true,
  "answer": "110",
  "ground_truth": "114",
  "f1_score": 0.85,
  "exact_match": false,
  "context_length": 188658,
  "question": "Total number of rolls in this episode?",
  "llm_calls": 12,
  "iterations": 12,
  "depth": 0,
  "elapsed_time": 244.23,
  "partition_strategy": "token",
  "retrieval_method": "unfiltered"
}
```

### Expected Findings

Based on the research plan, you should investigate:

1. **Does semantic partitioning improve accuracy?**
   - Compare F1 scores: semantic vs token vs structural

2. **Does smart retrieval reduce LLM calls?**
   - Compare LLM calls: embedding/regex vs unfiltered

3. **Which combination is best?**
   - Best accuracy: Highest F1 score
   - Most efficient: Lowest latency + LLM calls
   - Best balance: Good F1 with reasonable cost

## Dataset Information

**OOLONGBench** is a long-context benchmark using D&D campaign transcripts.

- **toy_dnd**: Smaller dataset (~480 test examples) - Default for quick testing
- **dnd**: Full dataset - For final evaluation

### Dataset Structure
- `context_window_text`: Long D&D transcript (~188K characters)
- `question`: Question about the transcript
- `answer`: Ground truth answer
- `question_type`: Type of question (counting, character info, etc.)

## Troubleshooting

### "OPENAI_API_KEY not found"
Create `.env` file in project root with your API key.

### "gpt-5-mini model not found"
Ensure you have access to GPT-5 Mini API. The model was released August 2025.

### Semantic partitioning fails
Semantic partitioning requires OpenAI embeddings API. If it fails, it automatically falls back to structural partitioning with a warning.

### Out of memory
Reduce `max_examples` in the evaluation script or use `toy_dnd` instead of `dnd`.

## Workflow Summary

```
1. Set up .env with API key
2. Run: python oolongbench_evaluation.py
   → Tests all 9 configurations
   → Saves results to oolongbench_results/
3. Run: python generate_oolongbench_figure.py
   → Generates comparison plots
4. Analyze figures and JSON files
   → Determine best partition strategy
5. Write up findings for your research
```

## Notes

- **Temperature**: GPT-5 models require `temperature=1.0` (only supported value)
- **Cost**: Each configuration tests 5 examples with ~10 LLM calls each = ~50 calls per config
  - 9 configs × 50 calls = ~450 total API calls for full evaluation
- **Time**: Expect ~15-30 minutes for full evaluation depending on context length
- **Partition Implementation**: Located in `../src/rlm/partitions.py` and `../src/rlm/retrieval.py`

## Research Context

This evaluation is part of the RLM-1 project investigating efficient long-context handling through:
- Automatic context partitioning
- Intelligent partition retrieval
- Recursive processing of selected partitions

The goal is to improve upon baseline RLM performance (no partitioning) and demonstrate that smarter partitioning strategies yield better accuracy and/or efficiency on long-context tasks.

For implementation details, see `../plan.md`.

## Citation

If you use this evaluation framework, please cite:

- **RLM Paper**: Zhang, Alex and Khattab, Omar. "Recursive Language Models." 2025.
- **OOLONGBench**: The OOLONGBench dataset (oolongbench/oolong-real on Hugging Face)

---

**Ready to evaluate?** Run `python oolongbench_evaluation.py` to start! 🚀

