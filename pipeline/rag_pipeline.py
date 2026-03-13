"""
pipeline/rag_pipeline.py
RAG pipeline: load corpus → chunk → embed → FAISS index → retrieve → generate
"""

import os
import re
import json
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    text: str
    doc_title: str
    chunk_id: str
    embedding: Optional[np.ndarray] = None


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    rank: int


@dataclass
class RAGResult:
    query_id: str
    query: str
    ground_truth: str
    retrieved_chunks: List[RetrievalResult]
    generated_answer: str
    prompt_used: str


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

def load_corpus(corpus_path: str) -> List[Dict[str, str]]:
    """Parse the custom corpus.txt format into list of {title, text} dicts."""
    with open(corpus_path, "r") as f:
        raw = f.read()

    docs = []
    blocks = re.split(r"===DOC_START===", raw)
    for block in blocks:
        block = block.strip()
        if not block or "===DOC_END===" not in block:
            continue
        content = block.replace("===DOC_END===", "").strip()
        title_match = re.match(r"TITLE:\s*(.+)\n", content)
        title = title_match.group(1).strip() if title_match else "Untitled"
        text = content[title_match.end():].strip() if title_match else content
        docs.append({"title": title, "text": text})

    logger.info(f"Loaded {len(docs)} documents from corpus.")
    return docs


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_documents(
    docs: List[Dict[str, str]],
    chunk_size: int = 200,
    chunk_overlap: int = 30,
) -> List[Chunk]:
    """Split documents into overlapping word-level chunks."""
    chunks = []
    for doc in docs:
        words = doc["text"].split()
        start = 0
        chunk_idx = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            text = " ".join(words[start:end])
            chunk_id = f"{doc['title'][:20].replace(' ', '_')}_{chunk_idx}"
            chunks.append(Chunk(text=text, doc_title=doc["title"], chunk_id=chunk_id))
            if end == len(words):
                break
            start += chunk_size - chunk_overlap
            chunk_idx += 1

    logger.info(f"Created {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap}).")
    return chunks


# ---------------------------------------------------------------------------
# Embedding + FAISS index
# ---------------------------------------------------------------------------

class VectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.chunks: List[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None

    def build(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        texts = [c.text for c in chunks]
        logger.info(f"Embedding {len(texts)} chunks...")
        embs = self.model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        self.embeddings = embs
        for i, chunk in enumerate(self.chunks):
            chunk.embedding = embs[i]
        logger.info("Vector store built.")

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievalResult]:
        """Cosine similarity retrieval (embeddings are already L2-normalised)."""
        q_emb = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ q_emb  # dot product = cosine for unit vectors
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top_indices):
            results.append(RetrievalResult(
                chunk=self.chunks[idx],
                score=float(scores[idx]),
                rank=rank + 1,
            ))
        return results


# ---------------------------------------------------------------------------
# BM25 retriever
# ---------------------------------------------------------------------------

class BM25Retriever:
    def __init__(self, chunks: List[Chunk]):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("Install rank_bm25: pip install rank-bm25 --break-system-packages")

        self.chunks = chunks
        tokenized = [c.text.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index built.")

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievalResult]:
        scores = self.bm25.get_scores(query.lower().split())
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top_indices):
            results.append(RetrievalResult(
                chunk=self.chunks[idx],
                score=float(scores[idx]),
                rank=rank + 1,
            ))
        return results


# ---------------------------------------------------------------------------
# Hybrid retriever (RRF fusion)
# ---------------------------------------------------------------------------

class HybridRetriever:
    """Reciprocal Rank Fusion of dense + sparse retrievers."""

    def __init__(self, vector_store: VectorStore, bm25: BM25Retriever, k: int = 60):
        self.vs = vector_store
        self.bm25 = bm25
        self.k = k

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievalResult]:
        dense = self.vs.retrieve(query, top_k=top_k * 2)
        sparse = self.bm25.retrieve(query, top_k=top_k * 2)

        scores: Dict[str, float] = {}
        chunk_map: Dict[str, Chunk] = {}

        for rank, r in enumerate(dense):
            scores[r.chunk.chunk_id] = scores.get(r.chunk.chunk_id, 0) + 1 / (self.k + rank + 1)
            chunk_map[r.chunk.chunk_id] = r.chunk

        for rank, r in enumerate(sparse):
            scores[r.chunk.chunk_id] = scores.get(r.chunk.chunk_id, 0) + 1 / (self.k + rank + 1)
            chunk_map[r.chunk.chunk_id] = r.chunk

        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]
        results = []
        for rank, cid in enumerate(sorted_ids):
            results.append(RetrievalResult(
                chunk=chunk_map[cid],
                score=scores[cid],
                rank=rank + 1,
            ))
        return results


# ---------------------------------------------------------------------------
# Generator (uses Anthropic API via the artifact bridge)
# ---------------------------------------------------------------------------

PROMPT_TEMPLATES = {
    "basic": (
        "Context:\n{context}\n\n"
        "Question: {query}\n\n"
        "Answer:"
    ),
    "faithful": (
        "You are a precise assistant. Answer ONLY using the context below. "
        "If the answer is not in the context, say 'I don't know'.\n\n"
        "Context:\n{context}\n\n"
        "Question: {query}\n\n"
        "Answer:"
    ),
    "cot": (
        "Context:\n{context}\n\n"
        "Question: {query}\n\n"
        "Think step by step, then provide a concise answer.\n\n"
        "Answer:"
    ),
}


class RAGPipeline:
    def __init__(
        self,
        retriever,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        top_k: int = 3,
        prompt_variant: str = "faithful",
        max_tokens: int = 256,
    ):
        self.retriever = retriever
        self.model = model
        self.top_k = top_k
        self.prompt_template = PROMPT_TEMPLATES.get(prompt_variant, PROMPT_TEMPLATES["faithful"])
        self.prompt_variant = prompt_variant
        self.max_tokens = max_tokens
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    def _build_context(self, chunks: List[RetrievalResult]) -> str:
        parts = []
        for r in chunks:
            parts.append(f"[{r.rank}] ({r.chunk.doc_title})\n{r.chunk.text}")
        return "\n\n".join(parts)

    def _generate(self, prompt: str) -> str:
        """Call Anthropic API directly."""
        import urllib.request
        import json as _json

        payload = _json.dumps({
            "model": self.model,
            "max_tokens": self.max_tokens,
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
            data = _json.loads(resp.read())
        return data["content"][0]["text"].strip()

    def run(self, query_id: str, query: str, ground_truth: str) -> RAGResult:
        retrieved = self.retriever.retrieve(query, top_k=self.top_k)
        context = self._build_context(retrieved)
        prompt = self.prompt_template.format(context=context, query=query)
        answer = self._generate(prompt)
        return RAGResult(
            query_id=query_id,
            query=query,
            ground_truth=ground_truth,
            retrieved_chunks=retrieved,
            generated_answer=answer,
            prompt_used=self.prompt_variant,
        )