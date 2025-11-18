# OOLONGBench Evaluation Suite

This folder contains everything needed to evaluate RLM on the OOLONGBench long-context benchmark with different partition strategies.

## 📁 Files Overview

### Scripts

| File | Purpose |
|------|---------|
| `oolongbench_evaluation.py` | Main evaluation script - runs RLM on OOLONGBench with different configurations |
| `analyze_oolongbench_results.py` | Analysis script - generates metrics, tables, plots from results |
| `test_setup.py` | Setup verification - checks API keys, dependencies, dataset access |

### Documentation

| File | Purpose |
|------|---------|
| `QUICKSTART_OOLONGBENCH.md` | Quick reference guide (start here!) |
| `OOLONGBENCH_GUIDE.md` | Comprehensive guide with detailed instructions and troubleshooting |
| `requirements_oolongbench.txt` | Python dependencies for evaluation |

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd oolongbench_eval
pip install -r requirements_oolongbench.txt
```

### 2. Set Up API Key

**Option 1: Using .env file (Recommended)**

Create a `.env` file in the **project root** directory (not in oolongbench_eval/):

```bash
# In C:\Users\rdavi\recursive-llm\.env
GEMINI_API_KEY=your-actual-key-here
```

See `env_example.txt` for a template.

**Option 2: Using environment variable**

```bash
export GEMINI_API_KEY="your-key-here"
```

### 3. Verify Setup

```bash
python test_setup.py
```

### 4. Run Evaluation

```bash
# Start with 5 examples from toy dataset (baseline test - fast)
python oolongbench_evaluation.py

# For full dataset: edit line 309 to use config="dnd" instead of "toy_dnd"
# Once partition strategies are implemented, uncomment all strategies in the script
```

**Note**: The script uses `toy_dnd` config by default for faster testing. See `DATASET_INFO.md` for details.

### 5. Analyze Results

```bash
python analyze_oolongbench_results.py
```

## 📊 Output

Results are saved to `oolongbench_results/` directory (created automatically):

```
oolongbench_results/
├── {config}_{timestamp}.json     # Detailed results per configuration
├── summary_{timestamp}.json      # Aggregated metrics
├── results.csv                   # Spreadsheet-friendly export
└── comparison_plot.png           # Visualization
```

## 📖 Workflow

### Phase 1: Baseline (Available Now)

Test current RLM implementation on OOLONGBench:

```bash
python test_setup.py              # Verify setup
python oolongbench_evaluation.py  # Run with baseline config (5 examples)
python analyze_oolongbench_results.py  # Analyze results
```

### Phase 2: Full Evaluation (After Implementation)

Once partition strategies from `../plan.md` are implemented:

1. Pull latest code: `git pull`
2. Edit `oolongbench_evaluation.py`:
   - Uncomment all partition strategies (token, structural, semantic, learned)
   - Uncomment all retrieval methods (regex, embedding, unfiltered)
   - Uncomment parallel options
   - Set `max_examples=None` for full dataset
3. Run: `python oolongbench_evaluation.py`
4. Analyze: `python analyze_oolongbench_results.py`

## 🎯 Research Questions

This evaluation suite helps answer (from `../plan.md`):

1. Does smarter partitioning improve accuracy?
2. Does embedding retrieval help vs regex?
3. How much speedup from parallel sub-queries?
4. Does learned strategy outperform fixed ones?

## 📚 Documentation

- **Quick Start**: Read `QUICKSTART_OOLONGBENCH.md` first
- **Detailed Guide**: See `OOLONGBENCH_GUIDE.md` for troubleshooting
- **Implementation Plan**: See `../plan.md` for partition strategy details

## 🔧 Configuration

The evaluation tests combinations of:

- **Partition Strategies**: token, structural, semantic, learned
- **Retrieval Methods**: regex, embedding, unfiltered
- **Parallel Processing**: False, True

This creates multiple configurations to compare (e.g., 4 × 3 × 2 = 24 configs).

## 📈 Example Results

After running the evaluation and analysis, you'll see:

```
CONFIGURATION COMPARISON
============================================================
Configuration                            Success    Avg Time    Avg Calls
------------------------------------------------------------
token_regex_parallel=True                5/5        8.2s        4.1
semantic_embedding_parallel=True         5/5        12.5s       5.3
...
```

## ⚠️ Important Notes

1. **API Keys**: Make sure to set your API key before running
2. **Rate Limits**: Start with small `max_examples` to avoid rate limits
3. **Memory**: Long contexts may require significant memory
4. **Time**: Full evaluation can take hours depending on dataset size

## 🆘 Troubleshooting

Run `test_setup.py` to diagnose common issues:
- Missing API keys
- Missing dependencies
- Dataset access problems
- RLM configuration issues

For detailed troubleshooting, see `OOLONGBENCH_GUIDE.md`.

## 🤝 Contributing

If you improve the evaluation scripts or add new analysis features, please update this README and the relevant documentation files.

---

**Ready to start?** Run `python test_setup.py` to verify your setup! 🚀

