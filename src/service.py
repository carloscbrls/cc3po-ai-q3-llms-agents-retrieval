"""A FastAPI service with a semantic cache, plus latency measurement.

Session 18. What "deploying" actually means for an AI feature:

- a **health endpoint** so your orchestrator knows you are alive
- a **prediction endpoint** with a validated request body
- a **semantic cache** -- because the same question asked in different
  words should not cost money twice
- **latency measurement** -- p50/p95, because the average hides your tail
- a **cost model** -- "per 1000 requests" is the number people budget with

The service runs entirely locally against the mock LLM. No API key.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent import Agent, AgentConfig
from embeddings_local import EmbeddingBackend, cosine_similarity
from mock_llm import MockLLM
from tools import registry as default_registry

__all__ = [
    "PredictRequest",
    "PredictResponse",
    "SemanticCache",
    "LatencyTracker",
    "create_app",
    "app",
]


# ----------------------------------------------------------------------
# Request / response models
# ----------------------------------------------------------------------
class PredictRequest(BaseModel):
    """The request body for POST /predict."""

    prompt: str = Field(..., min_length=1, description="The user's question or task")
    use_cache: bool = Field(True, description="Whether to consult the semantic cache")
    similarity_threshold: float = Field(
        0.92, ge=0.0, le=1.0, description="Cache hit threshold (cosine similarity)"
    )


class PredictResponse(BaseModel):
    """The response body for POST /predict."""

    answer: str
    cached: bool = False
    cache_key: str | None = None
    cache_similarity: float | None = None
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    tokens: int = 0
    stop_reason: str = "completed"
    iterations: int = 0


# ----------------------------------------------------------------------
# Semantic cache
# ----------------------------------------------------------------------
@dataclass
class SemanticCache:
    """Caches answers by *meaning*, not by exact string.

    Two prompts that embed to nearly the same vector share a cache entry.
    That is the whole idea: users phrase things differently, and you should
    not pay model prices for the same answer twice.
    """

    threshold: float = 0.92
    embed_fn: Any = None
    max_entries: int = 512
    _entries: list[dict] = field(default_factory=list)
    hits: int = 0
    misses: int = 0

    def __post_init__(self) -> None:
        if self.embed_fn is None:
            backend = EmbeddingBackend(mode="auto", dim=256)
            self.embed_fn = backend.embed

    def _vector(self, text: str) -> np.ndarray:
        return np.asarray(self.embed_fn(text), dtype=np.float64)

    def get(self, text: str, threshold: float | None = None) -> tuple[Any, float] | None:
        """Look up ``text``. Returns (entry, similarity) or None."""
        thr = self.threshold if threshold is None else threshold
        if not self._entries:
            self.misses += 1
            return None
        qvec = self._vector(text)
        best = None
        best_sim = -1.0
        for entry in self._entries:
            sim = cosine_similarity(qvec, entry["vector"])
            if sim > best_sim:
                best_sim = sim
                best = entry
        if best is not None and best_sim >= thr:
            self.hits += 1
            return best, best_sim
        self.misses += 1
        return None

    def put(self, text: str, value: Any) -> None:
        """Store an answer."""
        if len(self._entries) >= self.max_entries:
            # Simple FIFO eviction.
            self._entries.pop(0)
        self._entries.append({"text": text, "vector": self._vector(text), "value": value})

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "entries": len(self._entries),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (self.hits / total) if total else 0.0,
        }

    def clear(self) -> None:
        self._entries.clear()
        self.hits = 0
        self.misses = 0


# ----------------------------------------------------------------------
# Latency tracking
# ----------------------------------------------------------------------
class LatencyTracker:
    """Records request durations so you can report p50/p95."""

    def __init__(self) -> None:
        self.durations: list[float] = []

    def record(self, ms: float) -> None:
        self.durations.append(ms)

    def percentile(self, p: float) -> float:
        """The p-th percentile (0-100) of recorded latencies."""
        if not self.durations:
            return 0.0
        ordered = sorted(self.durations)
        if len(ordered) == 1:
            return ordered[0]
        # Linear interpolation between closest ranks.
        rank = (p / 100.0) * (len(ordered) - 1)
        low = math.floor(rank)
        high = math.ceil(rank)
        if low == high:
            return ordered[int(rank)]
        frac = rank - low
        return ordered[low] * (1 - frac) + ordered[high] * frac

    def summary(self) -> dict:
        if not self.durations:
            return {"n": 0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0, "max_ms": 0.0}
        return {
            "n": len(self.durations),
            "p50_ms": round(self.percentile(50), 3),
            "p95_ms": round(self.percentile(95), 3),
            "p99_ms": round(self.percentile(99), 3),
            "mean_ms": round(sum(self.durations) / len(self.durations), 3),
            "max_ms": round(max(self.durations), 3),
        }

    def clear(self) -> None:
        self.durations.clear()


# ----------------------------------------------------------------------
# The app
# ----------------------------------------------------------------------
def create_app(
    llm: Any | None = None,
    agent: Agent | None = None,
    cache: SemanticCache | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    Everything is injectable so the tests can pass in their own objects.
    """
    llm = llm if llm is not None else MockLLM()
    agent = agent if agent is not None else Agent(llm, tools=default_registry, config=AgentConfig())
    cache = cache if cache is not None else SemanticCache()

    application = FastAPI(
        title="CC3PO AI Builders -- Q3 inference service",
        description="A small, local, agent-backed API with a semantic cache.",
        version="0.3.0",
    )

    application.state.llm = llm
    application.state.agent = agent
    application.state.cache = cache
    application.state.latency = LatencyTracker()
    application.state.cost_per_request = 0.002

    @application.get("/health")
    def health() -> dict:
        """Liveness + readiness. Orchestrators hit this constantly."""
        return {
            "status": "ok",
            "service": "cc3po-q3-inference",
            "version": "0.3.0",
            "llm": getattr(llm, "name", type(llm).__name__),
            "cache": cache.stats(),
        }

    @application.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        """Run the agent on a prompt, consulting the semantic cache first."""
        start = time.perf_counter()

        if req.use_cache:
            hit = cache.get(req.prompt, threshold=req.similarity_threshold)
            if hit is not None:
                entry, sim = hit
                elapsed = (time.perf_counter() - start) * 1000
                application.state.latency.record(elapsed)
                return PredictResponse(
                    answer=entry["value"]["answer"],
                    cached=True,
                    cache_key=entry["text"],
                    cache_similarity=round(sim, 6),
                    latency_ms=round(elapsed, 3),
                    cost_usd=0.0,  # a cache hit costs nothing
                    tokens=0,
                    stop_reason=entry["value"].get("stop_reason", "completed"),
                    iterations=entry["value"].get("iterations", 0),
                )

        run = agent.run(req.prompt)
        elapsed = (time.perf_counter() - start) * 1000
        application.state.latency.record(elapsed)

        cost = run.total_cost or application.state.cost_per_request * 0.0

        if req.use_cache:
            cache.put(
                req.prompt,
                {
                    "answer": run.answer,
                    "stop_reason": run.stop_reason,
                    "iterations": run.iterations,
                },
            )

        return PredictResponse(
            answer=run.answer,
            cached=False,
            latency_ms=round(elapsed, 3),
            cost_usd=round(cost, 8),
            tokens=run.total_tokens,
            stop_reason=run.stop_reason,
            iterations=run.iterations,
        )

    @application.get("/metrics")
    def metrics() -> dict:
        """Latency percentiles and cache stats."""
        stats = application.state.latency.summary()
        cache_stats = cache.stats()
        # Cost per 1000 requests, using the observed average.
        avg_cost = application.state.cost_per_request
        return {
            "latency": stats,
            "cache": cache_stats,
            "cost_per_1000_requests_usd": round(avg_cost * 1000, 4),
            "note": "Cost model is simulated locally -- see README for the real numbers.",
        }

    @application.post("/cache/clear")
    def clear_cache() -> dict:
        cache.clear()
        application.state.latency.clear()
        return {"status": "cleared"}

    return application


# A module-level app so `uvicorn src.service:app` works.
app = create_app()


def _smoke_test() -> None:
    from fastapi.testclient import TestClient

    client = TestClient(app)
    print("health:", client.get("/health").json())

    r1 = client.post("/predict", json={"prompt": "What is the weather in Stockton?"})
    print("predict 1:", r1.json())

    r2 = client.post("/predict", json={"prompt": "What is the weather in Stockton?"})
    print("predict 2 (cache):", r2.json())

    print("metrics:", json.dumps(client.get("/metrics").json(), indent=2))


if __name__ == "__main__":
    _smoke_test()
