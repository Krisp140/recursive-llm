# RLM-1 Project Plan: Partitioning, Retrieval, and Parallel Recursive Calls

**Repo:** https://github.com/ysz/recursive-llm  
**Project:** RLM-1 — Towards More Efficient Long-Context Handling  
**Owner(s):** TODO (fill in)  
**Last updated:** TODO

This document is a *plan* for extending `recursive-llm` with:

1. Multiple **partitioning strategies** for long contexts.
2. Pluggable **retrieval methods** over partitions.
3. **Parallel recursive calls** over independent partitions.
4. A stretch goal: a **learned partitioning strategy** that adapts chunking to the data.

The public API (`from rlm import RLM`) should remain intact. All changes are internal and additive.

---

## 0. Current State (Baseline)

The `recursive-llm` repo already implements Recursive Language Models (RLMs):

- Context is stored as variables inside a Python REPL (via RestrictedPython).
- The root LM only sees the **query + tool descriptions**, not the entire raw context.
- The LM can:
  - Run Python code in the REPL to inspect/transform `context`.
  - Make **recursive LLM calls** via the core engine.
  - Return final answers via `FINAL(...)`.

Typical usage today:

```python
from rlm import RLM

rlm = RLM(model="gpt-5-mini")

# Sync
out = rlm.completion(query="...", context=long_doc)

# Async
out = await rlm.acompletion(query="...", context=long_doc)
```

## 1. Goals & Non-Goals

### 1.1 Goals

#### 1. Partitioning strategies

Implement a module to split context into structured partitions, with a configurable strategy:

- **token** — Baseline, fixed-size chunks (roughly reproducing current behavior).
- **structural** — Chunk on natural boundaries (paragraphs, headings, code blocks).
- **semantic** — Chunk based on topic shifts (embedding similarity).
- **learned** — A model-guided strategy that chooses boundaries or hyperparameters to improve downstream RLM performance (stretch goal, see §3.4).

#### 2. Retrieval over partitions

Implement a retrieval layer that chooses which partitions to focus on for recursive calls:

- **regex** — Grep-style retrieval over partitions (keyword/regex matches).
- **embedding** — Embedding-based similarity between query and partitions.
- **unfiltered** — No intelligent filtering; diagnostic baseline (e.g., "first k partitions").

#### 3. Parallel recursive calls

When multiple partitions are selected and are independent, allow recursive calls to be issued in parallel using the existing async stack (acompletion + asyncio), then stitched into a final answer.

#### 4. Experimentability

Make it easy to sweep and log:

- `partition_strategy ∈ {token, structural, semantic, learned}`
- `retrieval_method ∈ {regex, embedding, unfiltered}`
- `parallel_subqueries ∈ {False, True}`

and record:

- Accuracy / task performance,
- Latency (wall-clock),
- Token usage (root + recursive calls).

These become the main knobs for the class project experiments.

### 1.2 Non-Goals

We explicitly will not:

- Change the public RLM import or basic usage pattern.
- Replace RestrictedPython or the underlying REPL security model.
- Add unrelated tools or generic "agent" frameworks.
- Overhaul logging/telemetry beyond what's needed for basic metrics (tokens, time, accuracy).

## 2. Architecture & Integration Points

All new logic will live under `src/rlm/` alongside the existing core modules.

### 2.1 New modules

**`src/rlm/partitions.py`**

- Defines a `Partition` type.
- Implements `partition_text(...)` for different strategies, including the learned strategy.

**`src/rlm/retrieval.py`**

- Implements `PartitionRetriever` with different retrieval methods.

### 2.2 Existing modules to extend

**`src/rlm/rlm.py` (or equivalent core RLM class)**

- Add configuration fields:
  - `partition_strategy`
  - `retrieval_method`
  - `parallel_subqueries`
  - `max_parallel_subqueries`
  - Any knobs needed for the learned partitioner.
- Call `partition_text(...)` and `PartitionRetriever(...)` as part of completion / acompletion.
- Implement the fan-out logic for parallel recursive calls.

**REPL / executor module**

- Optionally expose helper functions to access specific partitions from within REPL code.
- Keep backwards compatibility by continuing to expose the full `context` variable as today.

## 3. Partitioning Strategies (partitions.py)

### 3.1 Partition data structure

We represent chunks of the original context as `Partition` objects:

### 3.2 Hand-designed strategies

#### (a) Token-based ("token")

Reproduce current "naive" behavior as closely as possible.

- Chunk by tokens or characters (depending on tokenizer availability).
- Ensure partitions are under `max_tokens`.

This is the default strategy and acts as the main baseline.

#### (b) Structural ("structural")

Tailored for natural text (and potentially code later):

- Split on `\n\n` to approximate paragraphs.
- Optionally treat lines that look like headings (e.g., `#`, numbered sections) as hard boundaries.
- Ensure each partition respects `max_tokens` by splitting/merging as needed.
- Populate metadata, for example:
  - `{"kind": "paragraph"}` or `{"kind": "heading_block", "heading": "..."}`.

#### (c) Semantic ("semantic")

Goal: cut where topics change.

Rough steps:

1. Split the document into sentences or short spans.
2. Embed each span with a sentence embedding model.
3. Compute cosine similarity between adjacent spans.
4. Start a new partition when similarity < threshold.
5. Merge spans into partitions up to `max_tokens`.

Optionally store per-partition embedding(s) in metadata for reuse in retrieval, e.g.:

### 3.3 Learned partitioning ("learned") — Stretch goal

Interpretation of the teacher's suggestion: make the partitioner itself learned, not purely rule-based.

We'll scope this to something feasible for a class project:

#### Option A: Hyperparameter meta-learner (minimum viable "learned" strategy)

Keep token/structural/semantic strategies.

Learn which hyperparameters to use (chunk size, overlap, thresholds) rather than hard-coding them.

Approaches:

- Grid/Bayesian search over a held-out set to pick best parameters.
- Or a small model that predicts config from document features (length, number of headings, etc.).

Implementation sketch:

1. Define a small config space.
2. Run RLM on a validation set with different configs.
3. Cache the best config and use it when `partition_strategy="learned"`.

#### Option B: Boundary classifier over sentences (ambitious, if time allows)

- Represent document as a sequence of sentence embeddings.
- Train a simple classifier `P(boundary | sentence)`:
  - Supervision from heuristics (e.g., headings, section markers) or a small labeled dataset.
- Use classifier outputs to decide partition boundaries, respecting `max_tokens`.

Implement as:

## 4. Retrieval Methods

### 4.1 Methods

#### (a) Regex / keyword ("regex")

For each partition:

- Count regex hits or simple keyword matches between query tokens and `partition.text`.
- Score partitions by:
  - Number of matches, and/or
  - Earliest match index.
- Return top-k.

#### (b) Embedding-based ("embedding")

- Ensure each partition has an embedding:
  - Reuse from semantic partitioning if available (`metadata["embedding"]`),
  - Or compute embeddings here.
- For each query:
  - Embed the query.
  - Compute cosine similarity with each partition embedding.
  - Return top-k partitions by similarity.

#### (c) Unfiltered ("unfiltered")

- Return all partitions (or the first k) in order.
- Used as a no-retrieval baseline and for debugging.

## 5. Parallel Recursive Calls

### 5.1 New configuration

Extend the RLM constructor with:

### 5.2 Async fan-out

When multiple partitions are chosen and no ordering dependency exists, we can fan out recursive calls in parallel:

### 5.3 Stitching partial answers

Initial simple approach:

- Collect `partial_answers` as a list of strings.
- Ask the root LM to synthesize a final answer, e.g.:
  - "You have the following partial answers from different parts of the context: [list]. Combine them into a single final answer to the original question."

More advanced stitching strategies can be considered later if time permits.

## 6. Experiment Plan

High-level evaluation for the class project:

**Tasks:**

- Long-document question answering (e.g., based on long articles / multi-file contexts).
- Possibly synthetic tasks (e.g., "find the one sentence with X" buried deep in long text).

**Conditions to sweep:**

- `partition_strategy`: token vs structural vs semantic vs learned.
- `retrieval_method`: regex vs embedding vs unfiltered.
- `parallel_subqueries`: False vs True.

**Metrics:**

- Answer accuracy (or hit rate).
- Total tokens used.
- Latency (time to answer).

**Questions we want to answer:**

- Does smarter partitioning improve accuracy at fixed cost?
- Does embedding retrieval help compared to regex on long contexts?
- How much speedup do we get from parallel sub-queries?
- Does the learned partition strategy outperform fixed ones?

## 7. Milestones / Checklist

### Phase 1 – Baseline & hooks

- Run existing examples to confirm baseline RLM behavior.
- Add `partition_strategy` / `retrieval_method` arguments to `RLM.__init__` (default to current behavior).
- Implement `Partition` type and `partition_text(..., "token")` to mirror existing behavior.

### Phase 2 – Structural & Semantic Partitioning

- Implement structural strategy in `partitions.py`.
- Implement semantic strategy (with embeddings) in `partitions.py`.
- Add minimal tests / sanity checks for both strategies.

### Phase 3 – Retrieval Layer

- Implement `PartitionRetriever` in `retrieval.py`.
- Implement regex, embedding, and unfiltered retrieval methods.
- Wire retrieval into the recursion loop (child calls use selected partitions).
- Add tests comparing retrieval outputs on toy inputs.

### Phase 4 – Parallel Sub-Queries

- Add `parallel_subqueries` + `max_parallel_subqueries` config to RLM.
- Implement `_run_subqueries_parallel` with asyncio.
- Replace serial loops where partitions are processed independently.
- Verify same answers vs serial, measure latency improvements.

### Phase 5 – Learned Partitioning (Stretch)

- Add an example script (e.g. `examples/rlm1_ablation.py`) to run ablations.
- Collect metrics across configurations.
- Prepare plots/tables for the class project write-up.

## 8. Open Questions

- Which embedding model do we standardize on for semantic partitioning + retrieval?
- How much of the partition/retrieval logic (especially "learned") should be surfaced to the LM vs. kept in Python?
- What dataset(s) do we use for evaluation (real long documents vs synthetic)?
- How far do we go with the "learned partitioning" idea within the class timeline?
