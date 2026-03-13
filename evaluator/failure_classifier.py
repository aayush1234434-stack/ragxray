"""
evaluator/failure_classifier.py

LLM-as-judge that classifies each RAG response into one of 4 failure categories
(or marks it as a pass). Uses the Anthropic API.
"""

import json
import logging
import urllib.request
from dataclasses import dataclass
from typing import Optional
import os

from pipeline.rag_pipeline import RAGResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

FAILURE_CATEGORIES = {
    "PASS": "The answer is correct and faithfully supported by the retrieved context.",
    "RETRIEVAL_DRIFT": (
        "The retrieved chunks are topically related to the query but do not contain "
        "the information needed to answer it. The model either says it doesn't know "
        "or produces a vague non-answer."
    ),
    "CONTEXT_DROP": (
        "The correct information IS present in the retrieved context, but the model "
        "ignores it and produces an incorrect or irrelevant answer."
    ),
    "HALLUCINATION": (
        "The model generates a confident, specific answer that is NOT supported by "
        "the retrieved context and contradicts or goes beyond the provided information."
    ),
    "REASONING_FAILURE": (
        "The answer requires connecting information across multiple chunks or "
        "multi-step inference, and the model fails to reason correctly even though "
        "the relevant information is present."
    ),
}

JUDGE_PROMPT = """You are an expert RAG (Retrieval-Augmented Generation) evaluator.

Your task: classify a RAG system's answer into exactly ONE failure category.

---
FAILURE TAXONOMY:
- PASS: Answer is correct and faithfully grounded in the context.
- RETRIEVAL_DRIFT: Retrieved chunks do not contain the answer; model says "I don't know" or gives a vague non-answer.
- CONTEXT_DROP: The answer IS in the retrieved chunks, but the model ignored it and gave a wrong answer.
- HALLUCINATION: The model confidently states information NOT in the context (makes something up).
- REASONING_FAILURE: Relevant info is in the context, but the model failed to connect or reason across it.

---
QUERY: {query}

GROUND TRUTH ANSWER: {ground_truth}

RETRIEVED CONTEXT:
{context}

MODEL'S ANSWER: {answer}

---
Respond with a JSON object only (no markdown, no explanation outside the JSON):
{{
  "category": "<one of: PASS, RETRIEVAL_DRIFT, CONTEXT_DROP, HALLUCINATION, REASONING_FAILURE>",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<one sentence explaining your classification>"
}}"""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FailureLabel:
    query_id: str
    category: str
    confidence: float
    reasoning: str
    is_failure: bool


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class FailureClassifier:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model

    def _build_context_str(self, result: RAGResult) -> str:
        parts = []
        for r in result.retrieved_chunks:
            parts.append(f"[Chunk {r.rank} | score={r.score:.3f}]\n{r.chunk.text}")
        return "\n\n".join(parts)

    def _call_api(self, prompt: str) -> dict:
        payload = json.dumps({
            "model": self.model,
            "max_tokens": 512,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
        raw = data["content"][0]["text"].strip()

        # Strip markdown fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)

    def classify(self, result: RAGResult) -> FailureLabel:
        context_str = self._build_context_str(result)
        prompt = JUDGE_PROMPT.format(
            query=result.query,
            ground_truth=result.ground_truth,
            context=context_str,
            answer=result.generated_answer,
        )
        try:
            parsed = self._call_api(prompt)
            category = parsed.get("category", "HALLUCINATION")
            confidence = float(parsed.get("confidence", 0.5))
            reasoning = parsed.get("reasoning", "")
        except Exception as e:
            logger.warning(f"Judge API call failed for {result.query_id}: {e}")
            category = "HALLUCINATION"
            confidence = 0.0
            reasoning = f"Classification failed: {e}"

        return FailureLabel(
            query_id=result.query_id,
            category=category,
            confidence=confidence,
            reasoning=reasoning,
            is_failure=(category != "PASS"),
        )

    def classify_batch(self, results: list) -> list:
        labels = []
        for i, result in enumerate(results):
            logger.info(f"Classifying [{i+1}/{len(results)}] {result.query_id}...")
            label = self.classify(result)
            labels.append(label)
            logger.info(f"  → {label.category} (confidence={label.confidence:.2f})")
        return labels