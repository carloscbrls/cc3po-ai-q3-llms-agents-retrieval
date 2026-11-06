"""A simple in-memory vector store with cosine search.

Session 15. This is the retrieval half of RAG.

Design note: **no ChromaDB, no external service.** Chroma is available as an
optional extra (``pip install -e ".[chroma]"``) but the default path is a
plain list plus numpy, because:

* it has zero install friction -- important on locked-down laptops
* it has no server to start or crash
* it is 100 lines and you can read all of it
* the *concepts* (chunk, embed, cosine search, top-k) are identical

If you want to see the same thing with a real vector DB, the optional
extra and the README show you how.

The store also implements a small **BM25-ish lexical scorer** so we can
compare dense retrieval against lexical retrieval, and a simple **reranker**
that boosts chunks which contain the query's rare terms.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "Chunk",
    "SearchResult",
    "VectorStore",
    "bm25_score",
    "tokenize",
    "load_corpus",
]

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "of", "to", "in",
    "on", "for", "and", "or", "it", "this", "that", "with", "as", "by", "at",
}


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, stopwords removed."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


@dataclass
class Chunk:
    """One document (or a slice of one) in the store."""

    id: str
    text: str
    metadata: dict = field(default_factory=dict)
    vector: np.ndarray | None = None

    @property
    def title(self) -> str:
        return self.metadata.get("title", self.id)


@dataclass
class SearchResult:
    """A retrieved chunk plus its score."""

    chunk: Chunk
    score: float
    rank: int = 0
    dense_score: float | None = None
    lexical_score: float | None = None
    rerank_score: float | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.chunk.id,
            "title": self.chunk.title,
            "score": round(self.score, 6),
            "rank": self.rank,
            "dense_score": self.dense_score,
            "lexical_score": self.lexical_score,
            "rerank_score": self.rerank_score,
            "text": self.chunk.text,
        }


def bm25_score(
    query_tokens: Sequence[str],
    doc_tokens: Sequence[str],
    doc_freqs: dict[str, int],
    avg_doc_len: float,
    n_docs: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """A compact BM25 implementation.

    BM25 rewards documents that contain the query terms, discounted when the
    term appears in most documents (IDF) and when the document is long.
    """
    if not doc_tokens or not query_tokens:
        return 0.0
    counts = Counter(doc_tokens)
    dl = len(doc_tokens)
    score = 0.0
    for term in set(query_tokens):
        tf = counts.get(term, 0)
        if tf == 0:
            continue
        n_q = doc_freqs.get(term, 0)
        idf = math.log(1 + (n_docs - n_q + 0.5) / (n_q + 0.5))
        denom = tf + k1 * (1 - b + b * dl / max(avg_doc_len, 1e-9))
        score += idf * (tf * (k1 + 1)) / denom
    return score


class VectorStore:
    """An in-memory store: a list of chunks plus their vectors.

    Args:
        embed_fn: ``str -> np.ndarray``. Required for dense search.
    """

    def __init__(self, embed_fn=None, name: str = "vector-store") -> None:
        self.embed_fn = embed_fn
        self.chunks: list[Chunk] = []
        self.name = name
        self._doc_freqs: dict[str, int] = {}
        self._avg_doc_len: float = 1.0
        self._tokenized: list[list[str]] = []

    # -- building ------------------------------------------------------
    def add(self, chunk: Chunk) -> None:
        """Add a chunk (embedding it if we have an embed function)."""
        if self.embed_fn is not None and chunk.vector is None:
            chunk.vector = np.asarray(self.embed_fn(chunk.text), dtype=np.float64)
        self.chunks.append(chunk)

    def add_many(self, chunks: Iterable[Chunk]) -> None:
        for c in chunks:
            self.add(c)

    def add_text(
        self,
        doc_id: str,
        text: str,
        metadata: dict | None = None,
        embed: bool = True,
    ) -> Chunk:
        """Add raw text. Convenience for the notebooks."""
        chunk = Chunk(id=doc_id, text=text, metadata=metadata or {})
        if embed and self.embed_fn is not None:
            chunk.vector = np.asarray(self.embed_fn(text), dtype=np.float64)
        self.add(chunk)
        return chunk

    def finalize(self) -> None:
        """Compute BM25 statistics. Call after adding all documents."""
        self._tokenized = [tokenize(c.text) for c in self.chunks]
        df: Counter = Counter()
        total_len = 0
        for toks in self._tokenized:
            total_len += len(toks)
            for term in set(toks):
                df[term] += 1
        self._doc_freqs = dict(df)
        self._avg_doc_len = (total_len / len(self._tokenized)) if self._tokenized else 1.0

    # -- searching -----------------------------------------------------
    def search(
        self,
        query: str,
        k: int = 5,
        mode: str = "dense",
        rerank: bool = False,
    ) -> list[SearchResult]:
        """Search the store.

        Args:
            query: the query string
            k: how many results to return
            mode: "dense" (vector), "lexical" (BM25), or "hybrid" (both)
            rerank: apply the lexical reranker on top of the candidate set

        Returns:
            A list of :class:`SearchResult`, best first.
        """
        if not self.chunks:
            return []

        if mode == "dense":
            if self.embed_fn is None:
                raise ValueError("dense search needs an embed_fn")
            return self._dense_search(query, k, rerank=rerank)
        if mode == "lexical":
            return self._lexical_search(query, k, rerank=rerank)
        if mode == "hybrid":
            return self._hybrid_search(query, k, rerank=rerank)
        raise ValueError(f"unknown mode: {mode!r} (use 'dense', 'lexical', 'hybrid')")

    def _dense_search(self, query: str, k: int, rerank: bool = False) -> list[SearchResult]:
        qvec = np.asarray(self.embed_fn(query), dtype=np.float64)
        scored: list[SearchResult] = []
        for chunk in self.chunks:
            if chunk.vector is None:
                continue
            score = _cosine(qvec, chunk.vector)
            scored.append(SearchResult(chunk=chunk, score=score, dense_score=score))
        scored.sort(key=lambda r: r.score, reverse=True)
        return self._apply_rerank(query, scored, k, rerank)

    def _lexical_search(self, query: str, k: int, rerank: bool = False) -> list[SearchResult]:
        q_tokens = tokenize(query)
        scored: list[SearchResult] = []
        for chunk, toks in zip(self.chunks, self._tokenized):
            score = bm25_score(
                q_tokens, toks, self._doc_freqs, self._avg_doc_len, len(self.chunks)
            )
            scored.append(SearchResult(chunk=chunk, score=score, lexical_score=score))
        scored.sort(key=lambda r: r.score, reverse=True)
        return self._apply_rerank(query, scored, k, rerank)

    def _hybrid_search(self, query: str, k: int, rerank: bool = False) -> list[SearchResult]:
        dense = self._dense_search(query, k=len(self.chunks), rerank=False)
        lexical = self._lexical_search(query, k=len(self.chunks), rerank=False)
        merged: dict[str, SearchResult] = {}
        for r in dense:
            merged[r.chunk.id] = r
        for r in lexical:
            if r.chunk.id in merged:
                merged[r.chunk.id].lexical_score = r.lexical_score
                merged[r.chunk.id].score = 0.6 * (merged[r.chunk.id].dense_score or 0.0) + 0.4 * r.score
            else:
                merged[r.chunk.id] = r
        out = sorted(merged.values(), key=lambda r: r.score, reverse=True)
        return self._apply_rerank(query, out, k, rerank)

    def _apply_rerank(
        self, query: str, scored: list[SearchResult], k: int, rerank: bool
    ) -> list[SearchResult]:
        """Take the top-N candidates and boost the ones with rare query terms.

        This is a stand-in for a cross-encoder reranker. The real thing
        (a small transformer that scores query/passage pairs) is the same
        idea with a better model.
        """
        candidates = scored[: max(k * 3, k)]
        if rerank and candidates:
            q_terms = set(tokenize(query))
            for r in candidates:
                d_terms = set(tokenize(r.chunk.text))
                overlap = len(q_terms & d_terms)
                rare_bonus = overlap / max(len(q_terms), 1)
                base = r.score if r.score is not None else 0.0
                # Scale the bonus relative to the top score so it is comparable.
                top = max((c.score or 0.0) for c in candidates) or 1.0
                r.rerank_score = base + 0.15 * top * rare_bonus
                r.score = r.rerank_score
            candidates.sort(key=lambda r: r.score, reverse=True)
        top = candidates[:k]
        for i, r in enumerate(top, start=1):
            r.rank = i
        return top

    # -- misc ----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.chunks)

    def get(self, doc_id: str) -> Chunk | None:
        for c in self.chunks:
            if c.id == doc_id:
                return c
        return None

    def ids(self) -> list[str]:
        return [c.id for c in self.chunks]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def load_corpus(corpus_dir: str | Path, embed_fn=None, store: VectorStore | None = None) -> VectorStore:
    """Load every ``*.md`` file in a directory into a store.

    Args:
        corpus_dir: directory of markdown files
        embed_fn: optional embedding function
        store: an existing store to add to (defaults to a new one)

    Returns:
        The populated :class:`VectorStore` (with BM25 stats finalized).
    """
    corpus_dir = Path(corpus_dir)
    if store is None:
        store = VectorStore(embed_fn=embed_fn)
    for path in sorted(corpus_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = path.stem
        # First markdown heading, if there is one, makes a nicer title.
        for line in text.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
        store.add_text(
            path.stem,
            text,
            metadata={"title": title, "path": str(path), "source": path.name},
            embed=embed_fn is not None,
        )
    store.finalize()
    return store


def _smoke_test() -> None:
    from embeddings_local import hash_embedding

    store = VectorStore(embed_fn=lambda t: hash_embedding(t, dim=128))
    store.add_text("a", "Transformers use attention to model sequences.")
    store.add_text("b", "Retrieval augmented generation grounds answers in documents.")
    store.add_text("c", "Convolutional networks slide filters over images.")
    store.finalize()
    print("chunks:", len(store))
    for r in store.search("attention transformers", k=2, mode="lexical"):
        print(f"  {r.chunk.id}: {r.score:.4f}  {r.chunk.text[:50]}")


if __name__ == "__main__":
    _smoke_test()
