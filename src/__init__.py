"""CC3PO AI Builders -- Q3 scaffold package.

    tools            JSON-schema tools, registry, dispatcher
    mock_llm         a deterministic scripted LLM (no API key)
    agent            the agent loop with tripwires
    vectorstore      in-memory vector store + BM25 + reranking
    embeddings_local local Ollama embeddings with an offline fallback
    memory           4 memory types, write policy, TTL eviction
    evals            exact match, fuzzy match, LLM-as-judge
    service          a FastAPI app with a semantic cache
"""

__all__ = [
    "tools",
    "mock_llm",
    "agent",
    "vectorstore",
    "embeddings_local",
    "memory",
    "evals",
    "service",
]

__version__ = "0.3.0"
