"""Local embeddings for Q3, with a hard offline guarantee.

Same design rule as Q2: **a lab must never hard-depend on a network call.**

Tries a local Ollama endpoint first, falls back to a deterministic
hash-based embedding. No API key, ever. No torch, ever.
"""

from __future__ import annotations

import hashlib
import os
import struct
from typing import Sequence

import numpy as np

__all__ = [
    "DEFAULT_OLLAMA_URL",
    "DEFAULT_MODEL",
    "hash_embedding",
    "ollama_embed",
    "EmbeddingBackend",
    "embed",
    "embed_many",
    "cosine_similarity",
]

DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "bge-m3")
DEFAULT_DIM = 256

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None  # type: ignore[assignment]

try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore[assignment]


def hash_embedding(text: str, dim: int = DEFAULT_DIM) -> np.ndarray:
    """A deterministic, offline, unit-length embedding built from token hashes.

    Not semantically meaningful. Perfectly fine for exercising plumbing,
    and identical inputs always give identical vectors.
    """
    vec = np.zeros(dim, dtype=np.float64)
    tokens = [t for t in text.lower().split() if t]
    if not tokens:
        tokens = [text or "empty"]
    for tok in tokens:
        digest = hashlib.sha256(tok.encode("utf-8")).digest()
        idx = struct.unpack(">Q", digest[:8])[0] % dim
        sign = 1.0 if digest[8] % 2 == 0 else -1.0
        vec[idx] += sign
        idx2 = struct.unpack(">Q", digest[8:16])[0] % dim
        vec[idx2] += sign * 0.5
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        vec[0] = 1.0
        norm = 1.0
    return vec / norm


def ollama_embed(
    texts: Sequence[str],
    model: str = DEFAULT_MODEL,
    url: str = DEFAULT_OLLAMA_URL,
    timeout: float = 5.0,
) -> list[np.ndarray] | None:
    """Call a local Ollama embeddings endpoint. Returns None if unavailable."""
    if httpx is not None:
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(
                    f"{url.rstrip('/')}/api/embed",
                    json={"model": model, "input": list(texts)},
                )
                resp.raise_for_status()
                payload = resp.json()
            vectors = payload.get("embeddings")
            if vectors:
                return [np.asarray(v, dtype=np.float64) for v in vectors]
        except Exception:
            return None
        return None

    if requests is not None:
        try:
            resp = requests.post(
                f"{url.rstrip('/')}/api/embed",
                json={"model": model, "input": list(texts)},
                timeout=timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
            vectors = payload.get("embeddings")
            if vectors:
                return [np.asarray(v, dtype=np.float64) for v in vectors]
        except Exception:
            return None
    return None


class EmbeddingBackend:
    """Picks a backend once and remembers it."""

    def __init__(
        self,
        mode: str = "auto",
        model: str = DEFAULT_MODEL,
        url: str = DEFAULT_OLLAMA_URL,
        dim: int = DEFAULT_DIM,
    ) -> None:
        if mode not in {"auto", "ollama", "hash"}:
            raise ValueError(f"unknown mode: {mode!r}")
        self.mode = mode
        self.model = model
        self.url = url
        self.dim = dim
        self._resolved: str | None = None

    @property
    def resolved(self) -> str:
        if self._resolved is None:
            self._resolve()
        assert self._resolved is not None
        return self._resolved

    def _resolve(self) -> None:
        if self.mode == "hash":
            self._resolved = "hash"
            return
        probe = ollama_embed(["ping"], model=self.model, url=self.url, timeout=2.0)
        if probe:
            self._resolved = "ollama"
            return
        if self.mode == "ollama":
            raise RuntimeError(
                f"Could not reach Ollama at {self.url}. Start it with `ollama serve` "
                "or use mode='hash'."
            )
        self._resolved = "hash"

    def embed_many(self, texts: Sequence[str]) -> list[np.ndarray]:
        if not texts:
            return []
        if self.resolved == "ollama":
            vectors = ollama_embed(texts, model=self.model, url=self.url)
            if vectors is None:
                self._resolved = "hash"
            else:
                return vectors
        return [hash_embedding(t, dim=self.dim) for t in texts]

    def embed(self, text: str) -> np.ndarray:
        return self.embed_many([text])[0]


def embed(text: str, **kwargs) -> np.ndarray:
    backend = kwargs.pop("backend", None) or EmbeddingBackend(**kwargs)
    return backend.embed(text)


def embed_many(texts: Sequence[str], **kwargs) -> list[np.ndarray]:
    backend = kwargs.pop("backend", None) or EmbeddingBackend(**kwargs)
    return backend.embed_many(texts)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)
