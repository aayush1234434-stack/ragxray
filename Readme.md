# ragxray 🔬

> Automated failure case mining for RAG pipelines — LLM-as-judge evaluation, retrieval quality scoring, and W&B experiment tracking.

![Python](https://img.shields.io/badge/Python-3.10+-3B5998?style=flat-square)
![Claude API](https://img.shields.io/badge/Claude-API-C8501A?style=flat-square)
![W&B](https://img.shields.io/badge/Weights_%26_Biases-tracked-FFBE00?style=flat-square&logoColor=black)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?style=flat-square)

---

## What is this?

RAG pipelines fail in structured, repeatable ways. Instead of eyeballing outputs manually, **ragxray** runs your query set through a full evaluation pipeline:

1. Retrieves context using dense / BM25 / hybrid (RRF) retrieval
2. Generates answers via Claude with 3 prompt variants
3. Scores retrieval quality automatically (context relevance, faithfulness, chunk precision)
4. Classifies each failure using an **LLM-as-judge** into a 4-category taxonomy
5. Logs everything to **W&B Tables** with regression detection vs baseline runs
6. Surfaces patterns via a **Streamlit dashboard**
---

## Evaluation Metrics

<!-- SCREENSHOT: Crop the top metrics row from the dashboard -->


| Metric | Run: hybrid/faithful |
|---|---|
| Pass Rate | **65%** |
| Failure Rate | 35% |
| Context Relevance | 0.682 |
| Answer Faithfulness | 0.714 |
| Chunk Precision | 0.820 |
| Keyword Recall | 0.765 |

---

## Failure Taxonomy
| Category | Count | Description |
|---|---|---|
| ✅ PASS | 13 | Answer correct and grounded in context |
| 🟠 RETRIEVAL_DRIFT | 3 | Chunks retrieved but don't contain the answer |
| 🔴 HALLUCINATION | 2 | Model fabricated information not in context |
| 🟣 CONTEXT_DROP | 1 | Correct chunk retrieved but model ignored it |
| 🔵 REASONING_FAILURE | 1 | Info present but model failed to connect it |

---

## Prompt × Retriever Sweep

| Config | Pass Rate |
|---|---|
| hybrid / cot | **73%** |
| hybrid / faithful | 65% |
| dense / faithful | 60% |
| dense / cot | 55% |
| bm25 / faithful | 50% |
| dense / basic | 45% |

> **Finding:** Hybrid retrieval with chain-of-thought prompting achieves the highest pass rate. BM25-only retrieval underperforms on semantically-phrased queries where keyword overlap is low.

---

## Sample Per-Query Output

<!-- SCREENSHOT: Crop the results table from the dashboard -->

```json
{
  "query_id": "q020",
  "query": "How does chain-of-thought prompting help RAG?",
  "ground_truth": "Chain-of-thought prompting improves multi-hop reasoning...",
  "generated_answer": "...Studies show up to 30% improvement in complex QA tasks.",
  "failure_category": "HALLUCINATION",
  "judge_reasoning": "The model fabricated 'up to 30%' — not present in context.",
  "retrieval_scores": {
    "context_relevance": 0.667,
    "answer_faithfulness": 0.423,
    "chunk_precision": 0.33,
    "keyword_recall": 0.50
  }
}
```

---

## Architecture

```
ragxray/
│
├── run_eval.py               ← main CLI entry point
│
├── pipeline/
│   └── rag_pipeline.py       ← corpus loading, chunking, VectorStore,
│                                BM25Retriever, HybridRetriever (RRF), RAGPipeline
│
├── evaluator/
│   └── failure_classifier.py ← LLM-as-judge (Claude) → 4-category taxonomy
│
├── scorer/
│   └── retrieval_scorer.py   ← context relevance, faithfulness, chunk precision
│
├── tracker/
│   └── wandb_tracker.py      ← W&B Tables, metrics, charts, regression detection
│
├── dashboard/
│   └── app.py                ← Streamlit results dashboard
│
├── data/
│   ├── corpus.txt            ← document corpus
│   └── queries.json          ← evaluation queries + ground truth
│
└── results/                  ← JSON outputs per run
```

---

## Retrieval Strategies

| Strategy | Description |
|---|---|
| `dense` | Cosine similarity over `all-MiniLM-L6-v2` embeddings |
| `bm25` | Sparse keyword retrieval via BM25Okapi |
| `hybrid` | **Reciprocal Rank Fusion** of dense + BM25 |

---

## Prompt Variants

| Variant | Description |
|---|---|
| `basic` | Minimal context + question |
| `faithful` | Instructs model to answer only from context, acknowledge unknowns |
| `cot` | Chain-of-thought: think step by step, then answer |

---

## Setup

```bash
git clone https://github.com/aayush1234434-stack/ragxray
cd ragxray
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
```

---

## Usage

```bash
# Single run (hybrid retriever, faithful prompt)
python run_eval.py --retriever hybrid --prompt-variant faithful

# Full sweep across all retriever x prompt combinations
python run_eval.py --sweep

# Limit queries (faster iteration)
python run_eval.py --num-queries 10 --no-wandb

# Regression detection vs a baseline W&B run
python run_eval.py --baseline-run-id entity/project/run_id --regression-threshold 0.05

# Launch dashboard
streamlit run dashboard/app.py
```

---

## W&B Tracking

Each run logs:
- **Results Table** — per-query: query, answer, failure category, judge reasoning, all retrieval scores
- **Aggregate metrics** — pass rate, failure rate, mean retrieval quality scores
- **Failure distribution bar chart**
- **Regression alerts** — flags metric drops > threshold vs a baseline run

---

## Bring Your Own Data

**Corpus format** (`data/corpus.txt`):
```
===DOC_START===
TITLE: Your Document Title
Your document content here.
===DOC_END===
```

**Query format** (`data/queries.json`):
```json
[
  {
    "id": "q001",
    "query": "Your question?",
    "ground_truth": "Expected answer"
  }
]
```

---

## Stack

| Component | Library |
|---|---|
| Generation + Judge | Anthropic Claude API |
| Dense embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Sparse retrieval | `rank-bm25` |
| Experiment tracking | `wandb` |
| Dashboard | `streamlit` + `plotly` |

---

## Key CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--retriever` | `hybrid` | `dense`, `bm25`, or `hybrid` |
| `--prompt-variant` | `faithful` | `basic`, `faithful`, or `cot` |
| `--top-k` | `3` | Chunks to retrieve per query |
| `--chunk-size` | `200` | Chunk size in words |
| `--num-queries` | all | Limit evaluation set size |
| `--baseline-run-id` | None | W&B run ID for regression detection |
| `--regression-threshold` | `0.05` | Relative drop = regression flag |
| `--no-wandb` | False | Disable W&B tracking |
| `--sweep` | False | Run all retriever x prompt combos |
