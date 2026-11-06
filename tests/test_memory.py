"""Tests for the memory system.

Session 16. Four kinds of memory, a write policy, TTL eviction, and a
memory-poisoning test that proves a system can believe something false.
"""

from __future__ import annotations

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memory import (  # noqa: E402
    InMemoryBackend,
    MemoryKind,
    MemoryRecord,
    MemoryStore,
    SQLiteBackend,
)


def test_write_and_recall():
    store = MemoryStore(backend=InMemoryBackend())
    rec = store.write(MemoryKind.SEMANTIC, "Carlos runs CC3PO from Lathrop, California.")
    assert rec is not None
    assert rec.kind == MemoryKind.SEMANTIC

    found = store.recall_semantic("where is the company based")
    assert found, "recall should return the record"
    assert "Lathrop" in found[0].content


def test_low_confidence_write_is_rejected():
    """The write policy is not optional -- it is what stops garbage."""
    store = MemoryStore(backend=InMemoryBackend(), min_confidence_to_write=0.5)
    rec = store.write(MemoryKind.SEMANTIC, "a fact I am not sure about", confidence=0.1)

    assert rec is None
    assert store.rejected_writes
    assert store.rejected_writes[0]["reason"] == "low confidence"
    assert store.backend.all() == []


def test_unknown_memory_kind_raises():
    store = MemoryStore(backend=InMemoryBackend())
    with pytest.raises(ValueError):
        store.write("not-a-kind", "something")


def test_ttl_eviction_removes_expired_records():
    """Forgetting is a policy, not an accident."""
    backend = InMemoryBackend()
    store = MemoryStore(backend=backend)

    store.write(MemoryKind.WORKING, "temporary scratch note", ttl_seconds=0.01)
    assert len(backend.all()) == 1

    time.sleep(0.05)
    evicted = store.evict_expired()

    assert len(evicted) == 1
    assert backend.all() == []


def test_record_expiry_is_computed_from_ttl():
    rec = MemoryRecord(id="x", kind=MemoryKind.SEMANTIC, content="c", ttl_seconds=60)
    assert rec.expires_at > rec.created_at
    assert not rec.expired

    old = MemoryRecord(id="y", kind=MemoryKind.SEMANTIC, content="c",
                       created_at=time.time() - 1000, ttl_seconds=60)
    assert old.expired


def test_memory_poisoning_a_fresh_store():
    """The S16 teaching moment, on an empty store.

    With nothing to contradict, a confident false fact is accepted. This is
    why 'the model remembers it' is not the same as 'it is true'.
    """
    store = MemoryStore(backend=InMemoryBackend())
    rec = store.write(MemoryKind.SEMANTIC, "The CC3PO office is located in Sacramento.")
    assert rec is not None

    recalled = store.recall_semantic("where is the CC3PO office")
    assert recalled
    assert "Sacramento" in recalled[0].content
    # It is stored with full confidence. Nothing flagged it as false.
    assert recalled[0].confidence == 1.0


def test_conflicting_high_salience_fact_is_rejected():
    """The policy can protect an existing fact from being overwritten."""
    store = MemoryStore(backend=InMemoryBackend(), conflict_threshold=0.15)

    # A long, specific, high-salience fact.
    store.write(
        MemoryKind.SEMANTIC,
        "The CC3PO LLC headquarters is at 1234 Morada Lane, Lathrop, California 95241.",
        salience_score=0.95,
    )
    # A contradiction on the SAME key phrase with a different number.
    rec = store.write(
        MemoryKind.SEMANTIC,
        "The CC3PO LLC headquarters is at 9876 Market Street, Sacramento, California 95814.",
        salience_score=0.10,
    )

    assert rec is None
    assert store.rejected_writes
    recalled = store.recall_semantic("where is CC3PO")
    assert any("Morada" in r.content for r in recalled)


def test_sqlite_backend_persists_records(tmp_path):
    """Session 16's 'walk away and come back' needs real persistence."""
    db = tmp_path / "mem.db"
    store = MemoryStore(backend=SQLiteBackend(str(db)))
    store.write(MemoryKind.SEMANTIC, "Remembered across sessions.")

    reopened = MemoryStore(backend=SQLiteBackend(str(db)))
    found = reopened.recall_semantic("what is remembered")
    assert found
    assert "across sessions" in found[0].content


def test_stats_and_clear():
    store = MemoryStore(backend=InMemoryBackend())
    store.write(MemoryKind.SEMANTIC, "one")
    store.write(MemoryKind.EPISODIC, "two")
    store.write(MemoryKind.PROCEDURAL, "three")

    stats = store.stats()
    assert stats.get("total", 0) >= 3

    store.clear()
    assert store.backend.all() == []


def test_clear_working_only():
    store = MemoryStore(backend=InMemoryBackend())
    store.write(MemoryKind.WORKING, "scratch")
    store.write(MemoryKind.SEMANTIC, "keep me")

    store.clear_working()
    remaining = store.backend.all()
    assert all(r.kind != MemoryKind.WORKING for r in remaining)
    assert any(r.kind == MemoryKind.SEMANTIC for r in remaining)


def test_four_memory_kinds_exist():
    assert MemoryKind.WORKING == "working"
    assert MemoryKind.EPISODIC == "episodic"
    assert MemoryKind.SEMANTIC == "semantic"
    assert MemoryKind.PROCEDURAL == "procedural"
    assert len(MemoryKind.ALL) == 4
