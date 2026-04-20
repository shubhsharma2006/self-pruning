"""
SparsityExplorer — RAG-based analysis of pruning results.
=========================================================
Proper RAG pipeline:
  1. Embed layer stats as natural language text via OpenAI embeddings
  2. Store in FAISS index (dim=1536, text-embedding-3-small)
  3. Query by embedding the user question → cosine search → top-k context
  4. Send context + question to GPT-4o-mini for an answer

Usage:
  explorer = SparsityExplorer()
  explorer.index_results(model.compute_sparsity())
  answer = explorer.ask("Which layer pruned the most and why?")
  print(answer)

Note: Requires OPENAI_API_KEY env var.
"""

import os
from typing import Dict, List, Optional

import numpy as np
import faiss
from openai import OpenAI

from database import logger

EMBED_MODEL  = "text-embedding-3-small"
CHAT_MODEL   = "gpt-4o-mini"
EMBED_DIM    = 1536
TOP_K        = 3


class SparsityExplorer:
    """
    RAG explorer: embeds layer-stat descriptions, retrieves relevant context,
    answers questions with an LLM.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.client       = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        # Flat L2 index on real text embeddings (dim=1536)
        self.index        = faiss.IndexFlatIP(EMBED_DIM)  # Inner product = cosine after normalising
        self.knowledge_base: List[str] = []   # raw text chunks for display
        self._embeddings : List[np.ndarray] = []

    # Ingestion

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of strings and return (N, 1536) float32 array, L2-normalised."""
        resp   = self.client.embeddings.create(model=EMBED_MODEL, input=texts)
        vecs   = np.array([r.embedding for r in resp.data], dtype="float32")
        # Normalise for cosine similarity via inner product
        norms  = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / (norms + 1e-9)

    def index_results(self, layer_stats: Dict, model_id: str = "model") -> None:
        """
        Convert layer_stats dict → natural language chunks → embed → add to FAISS.
        """
        chunks = []
        for name, stats in layer_stats.items():
            if name == "overall":
                text = (
                    f"Overall network ({model_id}): "
                    f"{stats['total']:,} total gates, "
                    f"{stats['pruned']:,} pruned ({stats['sparsity_%']:.1f}% sparsity)."
                )
            else:
                text = (
                    f"Layer {name} in {model_id}: "
                    f"{stats['total']:,} total gates, "
                    f"{stats['pruned']:,} pruned. "
                    f"Sparsity: {stats['sparsity_%']:.1f}%. "
                    f"{'Highly pruned — this layer had mostly redundant weights.' if stats['sparsity_%'] > 90 else ''}"
                    f"{'Moderately pruned — some weights retained.' if 50 < stats['sparsity_%'] <= 90 else ''}"
                    f"{'Lightly pruned — most weights important.' if stats['sparsity_%'] <= 50 else ''}"
                )
            chunks.append(text)

        if not chunks:
            logger.warning("No chunks to index.")
            return

        vecs = self._embed(chunks)
        self.index.add(vecs)
        self.knowledge_base.extend(chunks)
        logger.info(f"Indexed {len(chunks)} text chunks into FAISS (dim={EMBED_DIM}).")

    # Retrieval + Generation

    def ask(self, query: str, top_k: int = TOP_K) -> str:
        """
        Embed query → FAISS search → retrieve context → GPT-4o-mini answer.
        """
        if self.index.ntotal == 0:
            return "No data indexed yet. Call index_results() first."

        # Embed query
        q_vec = self._embed([query])   # (1, 1536)

        # FAISS search
        k      = min(top_k, self.index.ntotal)
        _, ids = self.index.search(q_vec, k)
        context_chunks = [self.knowledge_base[i] for i in ids[0] if i >= 0]
        context = "\n".join(f"- {c}" for c in context_chunks)

        # LLM generation
        system = (
            "You are an AI engineering expert specializing in neural network pruning. "
            "Answer the question using ONLY the context provided. "
            "Be concise, technical, and precise."
        )
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        resp = self.client.chat.completions.create(
            model    = CHAT_MODEL,
            messages = [
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            max_tokens  = 300,
            temperature = 0.1,
        )
        return resp.choices[0].message.content.strip()

    def ask_no_llm(self, query: str, top_k: int = TOP_K) -> str:
        """
        Retrieval only — returns the most relevant context chunks without LLM.
        Useful for testing without an API key.
        """
        if self.index.ntotal == 0:
            return "No data indexed."
        q_vec  = self._embed([query])
        k      = min(top_k, self.index.ntotal)
        _, ids = self.index.search(q_vec, k)
        return "\n".join(self.knowledge_base[i] for i in ids[0] if i >= 0)


if __name__ == "__main__":
    # Demo: index dummy stats and query without LLM (no API key needed)
    explorer = SparsityExplorer()

    dummy_stats = {
        "layer_0": {"total": 32768, "pruned": 30502, "sparsity_%": 93.1},
        "layer_1": {"total": 131072, "pruned": 131037, "sparsity_%": 99.97},
        "layer_2": {"total": 32768, "pruned": 32632, "sparsity_%": 99.6},
        "layer_3": {"total": 1280, "pruned": 1111, "sparsity_%": 86.8},
        "overall": {"total": 197888, "pruned": 195282, "sparsity_%": 98.68},
    }

    api_key = os.getenv("OPENAI_API_KEY", "")
    explorer.index_results(dummy_stats)

    q = "Which layer is most pruned and what does that tell us?"
    if api_key:
        print("Answer (with LLM):", explorer.ask(q))
    else:
        print("Answer (retrieval only):", explorer.ask_no_llm(q))
