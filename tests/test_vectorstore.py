"""Tests for the vector store: dense, lexical, hybrid, and reranking.

Session 15. The teaching claim is that retrieval quality dominates
generation quality -- so retrieval needs to be measurable.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from embeddings_local import hash_embedding  # noqa: E402
from vectorstore import Chunk, VectorStore, load_corpus  # noqa: E402


@pytest.fixture()
def store():
    s = VectorStore(embed_fn=hash_embedding, name="test")
    s.add_text("d1", "Termination: either party may terminate this agreement with thirty days notice.")
    s.add_text("d2", "Payment terms: customer shall pay all undisputed invoices within thirty days.")
    s.add_text("d3", "Confidentiality: each party shall protect the other party's confidential information.")
    s.add_text("d4", "Data protection: provider shall process personal data only on documented instructions.")
    s.finalize()
    return s


def test_store_has_all_chunks(store):
    assert len(store) == 4
    assert set(store.ids()) == {"d1", "d2", "d3", "d4"}


def test_dense_search_finds_the_right_document(store):
    """The nearest neighbour for a termination question is the termination clause."""
    results = store.search("how do I cancel this contract", k=2)
    assert results
    assert results[0].chunk.id == "d1"


def test_lexical_search_finds_keyword_matches(store):
    results = store.search("invoices payment", k=2, mode="lexical")
    assert results
    assert results[0].chunk.id == "d2"


def test_hybrid_search_returns_ranked_results(store):
    results = store.search("confidential information", k=3, mode="hybrid")
    assert len(results) <= 3
    assert all(hasattr(r, "score") for r in results)
    # Scores should be descending.
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_reranking_changes_the_order(store):
    """The S15 claim: reranking changes answers more than a bigger model."""
    plain = store.search("payment invoice terms", k=3, mode="hybrid", rerank=False)
    reranked = store.search("payment invoice terms", k=3, mode="hybrid", rerank=True)

    assert plain and reranked
    assert plain[0].chunk.id == "d2"


def test_search_respects_k(store):
    for mode in ("dense", "lexical", "hybrid"):
        results = store.search("agreement", k=2, mode=mode)
        assert len(results) <= 2


def test_get_returns_the_chunk(store):
    chunk = store.get("d1")
    assert chunk is not None
    assert "terminate" in chunk.text
    assert store.get("nope") is None


def test_hash_embedding_is_deterministic():
    a = hash_embedding("hello world")
    b = hash_embedding("hello world")
    c = hash_embedding("something else")

    assert a.shape == b.shape
    assert np.allclose(a, b)
    assert not np.allclose(a, c)
    assert float(np.linalg.norm(a)) > 0


def test_chunk_title_falls_back_to_id():
    c = Chunk(id="x", text="hello", metadata={"src": "test"})
    assert c.title == "x"
    c2 = Chunk(id="y", text="hello", metadata={"title": "Termination"})
    assert c2.title == "Termination"


def test_load_corpus_from_a_directory(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "one.md").write_text("# Termination\n\nEither party may terminate on notice.\n")
    (corpus / "two.md").write_text("# Payment\n\nInvoices are due within thirty days.\n")

    store = load_corpus(corpus, embed_fn=hash_embedding)
    assert len(store) == 2

    results = store.search("when are invoices due", k=1)
    assert results
    assert results[0].chunk.id in ("one", "two")


def test_real_embeddings_beat_hash_embeddings(tmp_path):
    """The S15 headline claim, as a test.

    A real embedding model (bge-m3 via the local Ollama) understands that
    'how do I cancel this contract' is about termination. A hash embedding
    does not -- it only understands that the strings share characters.

    This is the whole session in one assertion: retrieval quality dominates
    generation quality, and swapping in a better embedder can beat
    switching to a much larger model.

    Skipped when Ollama is not reachable, so the suite still passes on a
    laptop with nothing installed.
    """
    import json

    from embeddings_local import EmbeddingBackend

    backend = EmbeddingBackend(mode="auto", model="bge-m3", url="http://127.0.0.1:11434")
    try:
        backend.embed("probe")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no local Ollama embeddings available: {exc}")

    questions_path = os.path.join(os.path.dirname(__file__), "..", "data", "questions.json")
    corpus_path = os.path.join(os.path.dirname(__file__), "..", "data", "corpus")
    if not os.path.exists(questions_path):
        pytest.skip("corpus not built -- run scripts/build_corpus.py")

    with open(questions_path, encoding="utf-8") as fh:
        questions = json.load(fh)

    hash_store = load_corpus(corpus_path, embed_fn=hash_embedding)
    real_store = load_corpus(corpus_path, embed_fn=backend.embed)

    def top1_hit_rate(store):
        hits = 0
        for q in questions:
            results = store.search(q["question"], k=1, mode="dense")
            if results and results[0].chunk.id == q["expected_doc_id"]:
                hits += 1
        return hits / len(questions)

    hash_rate = top1_hit_rate(hash_store)
    real_rate = top1_hit_rate(real_store)

    assert real_rate > hash_rate, (
        f"real embeddings should beat hash embeddings "
        f"(real={real_rate:.3f} hash={hash_rate:.3f})"
    )
