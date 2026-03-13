"""
scorer/retrieval_scorer.py

Retrieval quality metrics:
  - Context Relevance Score (cosine similarity of query vs retrieved chunks)
  - Answer Faithfulness Score (cosine similarity of answer vs context)
  - Chunk Precision (fraction of retrieved chunks containing ground truth keywords)
  - BM25 Recall (keyword recall in retrieved context)
"""

import re
import logging
from typing import List, Dict
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from pipeline.rag_pipeline import RAGResult

logger = logging.getLogger(__name__)


@dataclass
class RetrievalScores:
    query_id: str
    context_relevance: float      # avg cosine(query, chunk) for top-k chunks
    answer_faithfulness: float    # cosine(answer, full_context)
    chunk_precision: float        # fraction of chunks with ground truth keywords
    keyword_recall: float         # fraction of GT keywords in full context


class RetrievalScorer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info(f"Loading scorer model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def _embed(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True)

    def _keywords(self, text: str) -> set:
        """Simple keyword extraction: lowercase alpha tokens, length > 3."""
        return set(w.lower() for w in re.findall(r"[a-zA-Z]{4,}", text))

    def score(self, result: RAGResult) -> RetrievalScores:
        # --- Build strings ---
        query = result.query
        chunks = [r.chunk.text for r in result.retrieved_chunks]
        full_context = " ".join(chunks)
        answer = result.generated_answer
        gt = result.ground_truth

        # --- Embeddings ---
        all_texts = [query, answer, full_context] + chunks
        embs = self._embed(all_texts)
        q_emb = embs[0]
        a_emb = embs[1]
        ctx_emb = embs[2]
        chunk_embs = embs[3:]

        # Context Relevance: mean cosine(query, each chunk)
        context_relevance = float(np.mean([q_emb @ c for c in chunk_embs]))

        # Answer Faithfulness: cosine(answer, full_context)
        answer_faithfulness = float(a_emb @ ctx_emb)

        # Chunk Precision: fraction of chunks containing any GT keyword
        gt_keywords = self._keywords(gt)
        if gt_keywords:
            hits = sum(
                1 for chunk_text in chunks
                if self._keywords(chunk_text) & gt_keywords
            )
            chunk_precision = hits / len(chunks) if chunks else 0.0
        else:
            chunk_precision = 1.0  # no keywords to check → assume fine

        # Keyword Recall: fraction of GT keywords present in full context
        ctx_keywords = self._keywords(full_context)
        if gt_keywords:
            keyword_recall = len(gt_keywords & ctx_keywords) / len(gt_keywords)
        else:
            keyword_recall = 1.0

        return RetrievalScores(
            query_id=result.query_id,
            context_relevance=round(context_relevance, 4),
            answer_faithfulness=round(answer_faithfulness, 4),
            chunk_precision=round(chunk_precision, 4),
            keyword_recall=round(keyword_recall, 4),
        )

    def score_batch(self, results: List[RAGResult]) -> List[RetrievalScores]:
        scores = []
        for result in results:
            s = self.score(result)
            scores.append(s)
            logger.debug(
                f"[{s.query_id}] rel={s.context_relevance:.3f} "
                f"faith={s.answer_faithfulness:.3f} "
                f"prec={s.chunk_precision:.3f} "
                f"recall={s.keyword_recall:.3f}"
            )
        return scores

    @staticmethod
    def aggregate(scores: List[RetrievalScores]) -> Dict[str, float]:
        """Compute macro-averages across all scored results."""
        if not scores:
            return {}
        return {
            "mean_context_relevance": round(np.mean([s.context_relevance for s in scores]), 4),
            "mean_answer_faithfulness": round(np.mean([s.answer_faithfulness for s in scores]), 4),
            "mean_chunk_precision": round(np.mean([s.chunk_precision for s in scores]), 4),
            "mean_keyword_recall": round(np.mean([s.keyword_recall for s in scores]), 4),
        }