# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RLM (Recursive Language Model) is a Python implementation for processing unbounded context lengths. Instead of passing huge contexts directly to LLMs, RLM stores context as a Python variable and allows the LLM to recursively explore and partition it through a safe REPL environment.

Based on the paper by Alex Zhang and Omar Khattab (MIT, 2025).

## Common Commands

```bash
# Install in editable mode
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_parser.py -v

# Run tests with coverage
pytest tests/ -v --cov=src/rlm --cov-report=term-missing

# Type checking
mypy src/rlm

# Linting
ruff check src/rlm

# Format code
black src/rlm tests examples
```

## Architecture

```
src/rlm/
├── core.py       # Main RLM class with completion logic and partition orchestration
├── repl.py       # Safe REPL executor using RestrictedPython
├── prompts.py    # System prompt templates (standard + locodiff task-specific)
├── parser.py     # FINAL() and FINAL_VAR() extraction from LLM responses
├── partitions.py # Context partitioning strategies (token, structural, semantic)
├── retrieval.py  # Partition retrieval methods (regex, embedding, unfiltered)
└── types.py      # Type definitions
```

### Core Flow

1. **RLM.completion()** receives query + context
2. If partitioning enabled (root level only): partitions context, processes each partition recursively, stitches answers
3. Otherwise: builds REPL environment with `context`, `query`, `recursive_llm()`, `llm_query()` functions
4. LLM writes Python code to explore context, executed in RestrictedPython sandbox
5. Loop continues until LLM outputs `FINAL("answer")` or `FINAL_VAR(variable_name)`

### Key Components

- **REPLExecutor** (`repl.py`): Sandboxed code execution via RestrictedPython. Exposes safe builtins (re, json, math, datetime, Counter) but blocks file/network access.
- **Partitioning** (`partitions.py`): Three strategies - `token` (fixed chunks with overlap), `structural` (paragraph/heading boundaries), `semantic` (embedding-based topic splits)
- **Retrieval** (`retrieval.py`): Selects relevant partitions via `regex` (keyword matching), `embedding` (cosine similarity), or `unfiltered` (first k)
- **Parser** (`parser.py`): Extracts final answers from `FINAL("...")`, `FINAL('''...''')`, or `FINAL_VAR(varname)` patterns

## Benchmarks

Two evaluation suites in the repository:

- **locodiff/locodiff_eval/**: Git history reconstruction benchmark (LoCoDiff dataset). Tests file reconstruction from commit diffs.
- **oolongbench_eval/**: Long-context QA benchmark using D&D transcripts. Tests partition/retrieval strategy combinations.

Run evaluations:
```bash
# Download LoCoDiff dataset first
bash locodiff/scripts/download_dataset.sh

# Run LoCoDiff evaluation
python locodiff/locodiff_eval/locodiff_evaluation.py

# Run OOLONGBench evaluation
python oolongbench_eval/oolongbench_evaluation.py
```

## Key Configuration Parameters

```python
RLM(
    model="gpt-4o",              # Main model
    recursive_model="gpt-4o-mini",  # Cheaper model for recursive calls
    max_depth=5,                 # Max recursion depth
    max_iterations=30,           # Max REPL iterations per call
    partition_strategy="token",  # None, "token", "structural", "semantic"
    retrieval_method="embedding", # "regex", "embedding", "unfiltered"
    parallel_subqueries=False,   # Parallel partition processing
    max_partition_tokens=4000,   # Max tokens per partition
    task="locodiff"              # Task-specific prompts (optional)
)
```

## Dependencies

- **litellm**: Universal LLM API support (100+ providers)
- **RestrictedPython**: Safe code execution sandbox
- **tiktoken**: Token counting for partitioning
- **numpy**: Embedding operations
