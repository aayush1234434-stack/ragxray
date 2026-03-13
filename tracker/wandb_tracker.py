"""
tracker/wandb_tracker.py

Logs all RAG eval results to Weights & Biases:
  - Per-query rows to W&B Tables
  - Aggregate metrics as run summary
  - Failure category distribution
  - Regression detection vs a baseline run
"""

import logging
from typing import List, Optional, Dict
from dataclasses import asdict

logger = logging.getLogger(__name__)


class WandbTracker:
    def __init__(
        self,
        project: str = "rag-failure-miner",
        entity: Optional[str] = None,
        run_name: Optional[str] = None,
        config: Optional[Dict] = None,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.run = None

        if not enabled:
            logger.info("W&B tracking disabled.")
            return

        try:
            import wandb
            self.wandb = wandb
            self.run = wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                config=config or {},
            )
            logger.info(f"W&B run initialised: {self.run.url}")
        except ImportError:
            logger.warning("wandb not installed. Run: pip install wandb --break-system-packages")
            self.enabled = False
        except Exception as e:
            logger.warning(f"W&B init failed: {e}. Tracking disabled.")
            self.enabled = False

    # ------------------------------------------------------------------
    # Log per-query results table
    # ------------------------------------------------------------------

    def log_results_table(self, rag_results, failure_labels, retrieval_scores):
        if not self.enabled:
            return

        label_map = {l.query_id: l for l in failure_labels}
        score_map = {s.query_id: s for s in retrieval_scores}

        columns = [
            "query_id", "query", "ground_truth", "generated_answer",
            "prompt_variant", "failure_category", "confidence", "judge_reasoning",
            "context_relevance", "answer_faithfulness", "chunk_precision", "keyword_recall",
            "num_chunks_retrieved", "top_chunk_score",
        ]
        table = self.wandb.Table(columns=columns)

        for r in rag_results:
            label = label_map.get(r.query_id)
            score = score_map.get(r.query_id)
            table.add_data(
                r.query_id,
                r.query,
                r.ground_truth,
                r.generated_answer,
                r.prompt_used,
                label.category if label else "UNKNOWN",
                label.confidence if label else 0.0,
                label.reasoning if label else "",
                score.context_relevance if score else 0.0,
                score.answer_faithfulness if score else 0.0,
                score.chunk_precision if score else 0.0,
                score.keyword_recall if score else 0.0,
                len(r.retrieved_chunks),
                r.retrieved_chunks[0].score if r.retrieved_chunks else 0.0,
            )

        self.run.log({"eval/results_table": table})
        logger.info("Logged results table to W&B.")

    # ------------------------------------------------------------------
    # Log aggregate metrics
    # ------------------------------------------------------------------

    def log_aggregate_metrics(self, metrics: Dict):
        if not self.enabled:
            return
        self.run.log({f"metrics/{k}": v for k, v in metrics.items()})
        logger.info(f"Logged aggregate metrics: {metrics}")

    # ------------------------------------------------------------------
    # Log failure distribution
    # ------------------------------------------------------------------

    def log_failure_distribution(self, failure_labels):
        if not self.enabled:
            return

        from collections import Counter
        counts = Counter(l.category for l in failure_labels)
        total = len(failure_labels)

        dist = {f"failures/{cat}": count for cat, count in counts.items()}
        dist["failures/pass_rate"] = round(counts.get("PASS", 0) / total, 4) if total else 0
        dist["failures/failure_rate"] = round(1 - dist["failures/pass_rate"], 4)

        self.run.log(dist)

        # Bar chart
        bar_data = [[cat, count] for cat, count in counts.items()]
        table = self.wandb.Table(data=bar_data, columns=["category", "count"])
        self.run.log({
            "failures/distribution": self.wandb.plot.bar(
                table, "category", "count", title="Failure Category Distribution"
            )
        })
        logger.info(f"Failure distribution: {dict(counts)}")

    # ------------------------------------------------------------------
    # Regression detection
    # ------------------------------------------------------------------

    def detect_regression(
        self,
        current_metrics: Dict,
        baseline_run_id: str,
        threshold: float = 0.05,
    ) -> Optional[Dict]:
        """
        Compare current metrics against a baseline W&B run.
        Returns dict of regressed metrics or None if no regression.
        """
        if not self.enabled:
            return None

        try:
            api = self.wandb.Api()
            baseline_run = api.run(baseline_run_id)
            baseline_summary = baseline_run.summary

            regressions = {}
            for key, current_val in current_metrics.items():
                baseline_key = f"metrics/{key}"
                if baseline_key in baseline_summary:
                    baseline_val = baseline_summary[baseline_key]
                    if baseline_val > 0:
                        drop = (baseline_val - current_val) / baseline_val
                        if drop > threshold:
                            regressions[key] = {
                                "baseline": round(baseline_val, 4),
                                "current": round(current_val, 4),
                                "drop_pct": round(drop * 100, 2),
                            }

            if regressions:
                logger.warning(f"REGRESSIONS DETECTED: {regressions}")
                self.run.log({"regression_detected": 1, "num_regressions": len(regressions)})
                self.run.summary["regressions"] = regressions
            else:
                logger.info("No regressions detected.")
                self.run.log({"regression_detected": 0})

            return regressions if regressions else None

        except Exception as e:
            logger.warning(f"Regression check failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Finish
    # ------------------------------------------------------------------

    def finish(self):
        if self.enabled and self.run:
            self.run.finish()
            logger.info("W&B run finished.")