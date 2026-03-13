"""
run_eval.py — Main CLI entry point for the RAG Failure Mining pipeline.

Usage:
    python run_eval.py --api-key YOUR_KEY
    python run_eval.py --api-key YOUR_KEY --retriever hybrid --prompt-variant cot --no-wandb
    python run_eval.py --help
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_eval")


def parse_args():
    parser = argparse.ArgumentParser(
        description="RAG Failure Case Mining & Error Analysis Framework"
    )
    parser.add_argument(
        "--api-key", type=str, default=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)"
    )
    parser.add_argument(
        "--corpus", type=str, default="data/corpus.txt",
        help="Path to corpus file"
    )
    parser.add_argument(
        "--queries", type=str, default="data/queries.json",
        help="Path to queries JSON file"
    )
    parser.add_argument(
        "--retriever", type=str, choices=["dense", "bm25", "hybrid"], default="hybrid",
        help="Retrieval strategy to use"
    )
    parser.add_argument(
        "--prompt-variant", type=str, choices=["basic", "faithful", "cot"], default="faithful",
        help="Prompt template variant"
    )
    parser.add_argument(
        "--top-k", type=int, default=3,
        help="Number of chunks to retrieve per query"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=200,
        help="Chunk size in words"
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=30,
        help="Chunk overlap in words"
    )
    parser.add_argument(
        "--num-queries", type=int, default=None,
        help="Limit number of queries to evaluate (default: all)"
    )
    parser.add_argument(
        "--wandb-project", type=str, default="rag-failure-miner",
        help="W&B project name"
    )
    parser.add_argument(
        "--wandb-entity", type=str, default=None,
        help="W&B entity/team"
    )
    parser.add_argument(
        "--baseline-run-id", type=str, default=None,
        help="W&B run ID to compare against for regression detection"
    )
    parser.add_argument(
        "--regression-threshold", type=float, default=0.05,
        help="Relative drop threshold for regression detection (default: 0.05 = 5%%)"
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable W&B tracking"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory to save JSON results"
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Run a full sweep across all prompt variants and retriever types"
    )
    return parser.parse_args()


def run_single_eval(
    api_key, corpus_path, queries, retriever_type, prompt_variant,
    top_k, chunk_size, chunk_overlap, wandb_project, wandb_entity,
    baseline_run_id, regression_threshold, use_wandb, output_dir,
):
    from pipeline.rag_pipeline import (
        load_corpus, chunk_documents, VectorStore, BM25Retriever,
        HybridRetriever, RAGPipeline,
    )
    from evaluator.failure_classifier import FailureClassifier
    from scorer.retrieval_scorer import RetrievalScorer
    from tracker.wandb_tracker import WandbTracker

    run_name = f"{retriever_type}_{prompt_variant}_{datetime.now().strftime('%m%d_%H%M')}"
    logger.info(f"\n{'='*60}")
    logger.info(f"Run: {run_name}")
    logger.info(f"  Retriever: {retriever_type} | Prompt: {prompt_variant} | top_k: {top_k}")
    logger.info(f"{'='*60}\n")

    # --- Build retriever ---
    docs = load_corpus(corpus_path)
    chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    vs = VectorStore()
    vs.build(chunks)

    if retriever_type == "dense":
        retriever = vs
    elif retriever_type == "bm25":
        retriever = BM25Retriever(chunks)
    else:
        bm25 = BM25Retriever(chunks)
        retriever = HybridRetriever(vs, bm25)

    # --- RAG pipeline ---
    pipeline = RAGPipeline(
        retriever=retriever,
        api_key=api_key,
        top_k=top_k,
        prompt_variant=prompt_variant,
    )

    # --- Run queries ---
    rag_results = []
    for q in queries:
        logger.info(f"Processing [{q['id']}]: {q['query'][:60]}...")
        result = pipeline.run(q["id"], q["query"], q["ground_truth"])
        rag_results.append(result)
        logger.info(f"  Answer: {result.generated_answer[:80]}...")

    # --- Score retrieval quality ---
    logger.info("\nScoring retrieval quality...")
    scorer = RetrievalScorer()
    retrieval_scores = scorer.score_batch(rag_results)
    agg_scores = scorer.aggregate(retrieval_scores)
    logger.info(f"Aggregate scores: {agg_scores}")

    # --- Classify failures ---
    logger.info("\nRunning LLM-as-judge failure classification...")
    classifier = FailureClassifier(api_key=api_key)
    failure_labels = classifier.classify_batch(rag_results)

    # --- Failure stats ---
    from collections import Counter
    category_counts = Counter(l.category for l in failure_labels)
    pass_rate = category_counts.get("PASS", 0) / len(failure_labels)
    failure_rate = 1 - pass_rate
    logger.info(f"\nFailure distribution: {dict(category_counts)}")
    logger.info(f"Pass rate: {pass_rate:.1%} | Failure rate: {failure_rate:.1%}")

    # --- W&B tracking ---
    tracker = WandbTracker(
        project=wandb_project,
        entity=wandb_entity,
        run_name=run_name,
        config={
            "retriever": retriever_type,
            "prompt_variant": prompt_variant,
            "top_k": top_k,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "num_queries": len(queries),
        },
        enabled=use_wandb,
    )

    metrics = {
        **agg_scores,
        "pass_rate": round(pass_rate, 4),
        "failure_rate": round(failure_rate, 4),
        **{f"failure_{cat.lower()}": count / len(failure_labels)
           for cat, count in category_counts.items()},
    }

    tracker.log_results_table(rag_results, failure_labels, retrieval_scores)
    tracker.log_aggregate_metrics(metrics)
    tracker.log_failure_distribution(failure_labels)

    if baseline_run_id:
        tracker.detect_regression(metrics, baseline_run_id, threshold=regression_threshold)

    tracker.finish()

    # --- Save JSON results ---
    Path(output_dir).mkdir(exist_ok=True)
    output_path = f"{output_dir}/{run_name}.json"

    results_data = {
        "run_name": run_name,
        "config": {
            "retriever": retriever_type,
            "prompt_variant": prompt_variant,
            "top_k": top_k,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        },
        "aggregate_metrics": metrics,
        "failure_distribution": dict(category_counts),
        "per_query": [
            {
                "query_id": r.query_id,
                "query": r.query,
                "ground_truth": r.ground_truth,
                "generated_answer": r.generated_answer,
                "prompt_used": r.prompt_used,
                "failure_category": next(
                    (l.category for l in failure_labels if l.query_id == r.query_id), "UNKNOWN"
                ),
                "judge_reasoning": next(
                    (l.reasoning for l in failure_labels if l.query_id == r.query_id), ""
                ),
                "retrieval_scores": next(
                    ({
                        "context_relevance": s.context_relevance,
                        "answer_faithfulness": s.answer_faithfulness,
                        "chunk_precision": s.chunk_precision,
                        "keyword_recall": s.keyword_recall,
                    } for s in retrieval_scores if s.query_id == r.query_id),
                    {},
                ),
            }
            for r in rag_results
        ],
    }

    with open(output_path, "w") as f:
        json.dump(results_data, f, indent=2)
    logger.info(f"\nResults saved to: {output_path}")

    return metrics, output_path


def main():
    args = parse_args()

    if not args.api_key:
        logger.error("No API key provided. Use --api-key or set ANTHROPIC_API_KEY.")
        sys.exit(1)

    with open(args.queries) as f:
        all_queries = json.load(f)

    if args.num_queries:
        all_queries = all_queries[:args.num_queries]

    if args.sweep:
        logger.info("Running full sweep across all retriever × prompt combinations...")
        sweep_configs = [
            ("dense", "basic"),
            ("dense", "faithful"),
            ("dense", "cot"),
            ("bm25", "faithful"),
            ("hybrid", "faithful"),
            ("hybrid", "cot"),
        ]
        all_metrics = []
        for retriever_type, prompt_variant in sweep_configs:
            metrics, path = run_single_eval(
                api_key=args.api_key,
                corpus_path=args.corpus,
                queries=all_queries,
                retriever_type=retriever_type,
                prompt_variant=prompt_variant,
                top_k=args.top_k,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                baseline_run_id=args.baseline_run_id,
                regression_threshold=args.regression_threshold,
                use_wandb=not args.no_wandb,
                output_dir=args.output_dir,
            )
            all_metrics.append({"config": f"{retriever_type}/{prompt_variant}", "metrics": metrics})

        logger.info("\n" + "="*60)
        logger.info("SWEEP SUMMARY")
        logger.info("="*60)
        for entry in sorted(all_metrics, key=lambda x: x["metrics"].get("pass_rate", 0), reverse=True):
            logger.info(
                f"{entry['config']:30s} | "
                f"pass_rate={entry['metrics'].get('pass_rate', 0):.1%} | "
                f"faithfulness={entry['metrics'].get('mean_answer_faithfulness', 0):.3f}"
            )
    else:
        run_single_eval(
            api_key=args.api_key,
            corpus_path=args.corpus,
            queries=all_queries,
            retriever_type=args.retriever,
            prompt_variant=args.prompt_variant,
            top_k=args.top_k,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            baseline_run_id=args.baseline_run_id,
            regression_threshold=args.regression_threshold,
            use_wandb=not args.no_wandb,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()