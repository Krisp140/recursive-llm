# OOLONGBench Dataset Information

## Available Configs

The OOLONGBench dataset (`oolongbench/oolong-real`) has two configurations:

### 1. `toy_dnd` (Recommended for Quick Testing)
- **Purpose**: Smaller dataset for rapid testing and development
- **Use when**: You want to quickly verify your setup or test changes
- **Faster**: Downloads and runs much quicker

### 2. `dnd` (Full Dataset)
- **Purpose**: Complete D&D dataset for full evaluation
- **Use when**: Running final evaluation for results/paper
- **Larger**: Takes longer to download and process

## Usage

### In Python Code

```python
from datasets import load_dataset

# Quick testing (small dataset)
dataset = load_dataset("oolongbench/oolong-real", "toy_dnd")

# Full evaluation (complete dataset)
dataset = load_dataset("oolongbench/oolong-real", "dnd")
```

### In Our Scripts

**For quick testing** (default):
```python
# oolongbench_evaluation.py line 309
dataset = evaluator.load_oolongbench(config="toy_dnd")
```

**For full evaluation**:
```python
# oolongbench_evaluation.py line 309
dataset = evaluator.load_oolongbench(config="dnd")
```

## Dataset Structure

Both configs have the same structure but different sizes:

```python
DatasetDict({
    'validation': Dataset({
        features: ['id', 'context_window_id', 'context_window_text', 'question', 
                   'answer', 'question_type', 'episodes', 'campaign'],
        num_rows: N
    }),
    'test': Dataset({
        features: ['id', 'context_window_id', 'context_window_text', 'question', 
                   'answer', 'question_type', 'episodes', 'campaign'],
        num_rows: N
    })
})
```

### Key Fields

- **`context_window_text`**: The long context document (D&D campaign transcript)
- **`question`**: The question about the context
- **`answer`**: Ground truth answer
- **`question_type`**: Type of question being asked
- **`id`**: Unique example identifier
- **`episodes`**: Related episode information
- **`campaign`**: Campaign identifier

## Recommended Workflow

1. **Development & Testing**: Use `toy_dnd`
   ```bash
   # Edit oolongbench_evaluation.py line 309:
   dataset = evaluator.load_oolongbench(config="toy_dnd")
   
   # Run with few examples
   python oolongbench_evaluation.py
   ```

2. **Final Evaluation**: Use `dnd`
   ```bash
   # Edit oolongbench_evaluation.py line 309:
   dataset = evaluator.load_oolongbench(config="dnd")
   
   # Set max_examples=None for full dataset
   python oolongbench_evaluation.py
   ```

## Switching Configs

To switch between configs, edit `oolongbench_evaluation.py` line 309:

```python
# Quick testing
dataset = evaluator.load_oolongbench(config="toy_dnd")

# OR

# Full evaluation
dataset = evaluator.load_oolongbench(config="dnd")
```

## Quick Commands

```bash
# Test with toy dataset (fast)
cd oolongbench_eval
python simple_baseline_test.py  # Uses toy_dnd by default

# Full evaluation with complete dataset
# Edit oolongbench_evaluation.py first to use config="dnd"
python oolongbench_evaluation.py
```

