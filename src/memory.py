"""Four kinds of memory, with a write policy and TTL eviction.

Session 16. This is where the class meets reality: an agent that remembers
things is an agent that can be *poisoned*.

The four kinds:

- **Working memory** -- the scratchpad for the current task. Cleared when
  the task ends. Like a whiteboard.
- **Episodic memory** -- "what happened". Specific events with a timestamp.
  Like a journal.
- **Semantic memory** -- "what is true". General facts about the world.
  Like a wiki.
- **Procedural memory** -- "how to do it". Rules, steps, playbooks.
  Like a runbook.

The write policy is the interesting part. Not everything gets written:

- **Salience**: how important is this? A high-salience fact survives longer.
- **TTL (time to live)**: how long does it stay valid?
- **Confidence**: how sure are we? A low-confidence write should not
  overwrite a high-confidence fact.
- **Contradiction**: a new fact that contradicts an old one does not simply
  replace it. The old one gets demoted and the conflict is recorded.

That last rule is what makes the poisoning test in the notebook survivable:
inject a false fact, watch the system believe it, then watch the TTL and
salience policy evict it.

Storage is plain Python by default. SQLite is available via
``MemoryStore(path=...)`` and behaves identically.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Iterable, Literal

__all__ = [
    "MemoryKind",
    "MemoryRecord",
    "MemoryStore",
    "InMemoryBackend",
    "SQLiteBackend",
    "salience",
]


class MemoryKind:
    """The four memory types."""

    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"

    ALL = (WORKING, EPISODIC, SEMANTIC, PROCEDURAL)


MemoryType = Literal["working", "episodic", "semantic", "procedural"]


def salience(text: str, kind: str = MemoryKind.SEMANTIC) -> float:
    """A crude but honest salience score in [0, 1].

    Heuristics: length (longer claims carry more), the presence of
    numbers or proper nouns (more specific), and a boost for procedural
    memories (rules should stick).
    """
    if not text:
        return 0.0
    words = text.split()
    length_score = min(len(words) / 25.0, 1.0)
    specificity = 0.0
    if any(ch.isdigit() for ch in text):
        specificity += 0.2
    if any(ch.isupper() for ch in text):
        specificity += 0.2
    if any(w in text.lower() for w in ("always", "never", "must", "shall", "policy")):
        specificity += 0.2
    base = 0.4 * length_score + min(specificity, 0.6)
    if kind == MemoryKind.PROCEDURAL:
        base = min(1.0, base + 0.2)
    return round(min(1.0, max(0.0, base)), 4)


@dataclass
class MemoryRecord:
    """One memory item."""

    id: str
    kind: str
    content: str
    salience: float = 0.0
    confidence: float = 1.0
    created_at: float = field(default_factory=time.time)
    ttl_seconds: float = 3600.0
    metadata: dict = field(default_factory=dict)
    # Set when a newer record contradicts this one.
    superseded_by: str | None = None

    @property
    def expires_at(self) -> float:
        return self.created_at + self.ttl_seconds

    @property
    def expired(self) -> bool:
        return time.time() >= self.expires_at

    def age_seconds(self) -> float:
        return time.time() - self.created_at

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "content": self.content,
            "salience": self.salience,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "ttl_seconds": self.ttl_seconds,
            "expires_at": self.expires_at,
            "metadata": self.metadata,
            "superseded_by": self.superseded_by,
        }


# ----------------------------------------------------------------------
# Backends
# ----------------------------------------------------------------------
class InMemoryBackend:
    """Keeps records in a Python list. The default."""

    def __init__(self) -> None:
        self._records: list[MemoryRecord] = []

    def add(self, record: MemoryRecord) -> None:
        self._records.append(record)

    def all(self) -> list[MemoryRecord]:
        return list(self._records)

    def remove(self, record_id: str) -> None:
        self._records = [r for r in self._records if r.id != record_id]

    def update(self, record: MemoryRecord) -> None:
        for i, r in enumerate(self._records):
            if r.id == record.id:
                self._records[i] = record
                return

    def clear(self) -> None:
        self._records.clear()


class SQLiteBackend:
    """Persists records to a SQLite file. Same behaviour, survives a restart.

    Use it when you want to show students that memory is just data.
    """

    def __init__(self, path: str = ":memory:") -> None:
        self.path = path
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                content TEXT NOT NULL,
                salience REAL NOT NULL,
                confidence REAL NOT NULL,
                created_at REAL NOT NULL,
                ttl_seconds REAL NOT NULL,
                metadata TEXT NOT NULL,
                superseded_by TEXT
            )
            """
        )
        self._conn.commit()

    def add(self, record: MemoryRecord) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO memories VALUES (?,?,?,?,?,?,?,?,?)",
            (
                record.id,
                record.kind,
                record.content,
                record.salience,
                record.confidence,
                record.created_at,
                record.ttl_seconds,
                json.dumps(record.metadata),
                record.superseded_by,
            ),
        )
        self._conn.commit()

    def all(self) -> list[MemoryRecord]:
        rows = self._conn.execute("SELECT * FROM memories").fetchall()
        return [
            MemoryRecord(
                id=r[0],
                kind=r[1],
                content=r[2],
                salience=r[3],
                confidence=r[4],
                created_at=r[5],
                ttl_seconds=r[6],
                metadata=json.loads(r[7]),
                superseded_by=r[8],
            )
            for r in rows
        ]

    def remove(self, record_id: str) -> None:
        self._conn.execute("DELETE FROM memories WHERE id = ?", (record_id,))
        self._conn.commit()

    def update(self, record: MemoryRecord) -> None:
        self.add(record)

    def clear(self) -> None:
        self._conn.execute("DELETE FROM memories")
        self._conn.commit()


# ----------------------------------------------------------------------
# The store
# ----------------------------------------------------------------------
DEFAULT_TTL = {
    MemoryKind.WORKING: 600.0,       # 10 minutes
    MemoryKind.EPISODIC: 86400.0,     # 1 day
    MemoryKind.SEMANTIC: 604800.0,    # 1 week
    MemoryKind.PROCEDURAL: 2592000.0,  # 30 days
}


class MemoryStore:
    """The memory system: write policy, retrieval, TTL eviction.

    Args:
        backend: an ``InMemoryBackend`` or ``SQLiteBackend``.
        min_confidence_to_write: writes below this confidence are rejected.
        conflict_threshold: salience difference needed for a new fact to
            supersede an existing one.
    """

    def __init__(
        self,
        backend=None,
        min_confidence_to_write: float = 0.2,
        conflict_threshold: float = 0.15,
    ) -> None:
        self.backend = backend if backend is not None else InMemoryBackend()
        self.min_confidence_to_write = min_confidence_to_write
        self.conflict_threshold = conflict_threshold
        self.rejected_writes: list[dict] = []

    # -- writing -------------------------------------------------------
    def write(
        self,
        kind: str,
        content: str,
        confidence: float = 1.0,
        ttl_seconds: float | None = None,
        metadata: dict | None = None,
        salience_score: float | None = None,
    ) -> MemoryRecord | None:
        """Write a memory. Returns the record, or None if the write was rejected.

        Rejected when confidence is too low, or when the content contradicts
        a high-salience existing fact of the same kind.
        """
        if kind not in MemoryKind.ALL:
            raise ValueError(f"unknown memory kind: {kind!r}")

        if confidence < self.min_confidence_to_write:
            self.rejected_writes.append(
                {"kind": kind, "content": content, "reason": "low confidence", "confidence": confidence}
            )
            return None

        ttl = ttl_seconds if ttl_seconds is not None else DEFAULT_TTL.get(kind, 3600.0)
        score = salience(content, kind) if salience_score is None else salience_score

        # Conflict check against existing semantic/procedural facts.
        if kind in (MemoryKind.SEMANTIC, MemoryKind.PROCEDURAL):
            for existing in self.backend.all():
                if existing.kind != kind or existing.superseded_by is not None:
                    continue
                if not self._contradicts(content, existing.content):
                    continue
                if existing.salience - score > self.conflict_threshold:
                    # The existing fact is meaningfully more salient. Refuse.
                    self.rejected_writes.append(
                        {
                            "kind": kind,
                            "content": content,
                            "reason": f"contradicts more salient memory {existing.id}",
                            "conflicting_with": existing.id,
                        }
                    )
                    return None
                # The new fact wins: demote the old one.
                existing.superseded_by = "pending"
                self.backend.update(existing)

        record = MemoryRecord(
            id=uuid.uuid4().hex[:12],
            kind=kind,
            content=content,
            salience=score,
            confidence=confidence,
            ttl_seconds=ttl,
            metadata=metadata or {},
        )
        self.backend.add(record)

        # Tie up the superseded record's dangling pointer.
        for existing in self.backend.all():
            if existing.superseded_by == "pending" and existing.id != record.id:
                if self._contradicts(content, existing.content):
                    existing.superseded_by = record.id
                    self.backend.update(existing)
        return record

    @staticmethod
    def _contradicts(new: str, old: str) -> bool:
        """Very simple contradiction detection: shared subject, different claim.

        Good enough for the lab: it catches "X is 5" vs "X is 7" and
        "always do Y" vs "never do Y".
        """
        negators = ("not ", "never ", "no ", "isn't ", "doesn't ", "don't ")
        a = new.lower()
        b = old.lower()
        a_neg = any(n in a for n in negators)
        b_neg = any(n in b for n in negators)
        if a_neg != b_neg:
            # Same topic, opposite polarity.
            a_words = {w for w in a.split() if len(w) > 3}
            b_words = {w for w in b.split() if len(w) > 3}
            overlap = len(a_words & b_words)
            return overlap >= 2
        # Numeric contradiction on the same key phrase.
        import re

        a_nums = re.findall(r"\d+", a)
        b_nums = re.findall(r"\d+", b)
        if a_nums and b_nums and a_nums != b_nums:
            a_words = {w for w in a.split() if len(w) > 3}
            b_words = {w for w in b.split() if len(w) > 3}
            if len(a_words & b_words) >= 2:
                return True
        return False

    # -- reading -------------------------------------------------------
    def recall(
        self,
        kind: str | None = None,
        query: str | None = None,
        limit: int = 10,
        include_expired: bool = False,
        include_superseded: bool = False,
    ) -> list[MemoryRecord]:
        """Retrieve memories, best first.

        Ordering is by a simple relevance score: salience, boosted by
        confidence and by query-term overlap, penalised by age.
        """
        now = time.time()
        records = self.backend.all()

        if kind is not None:
            records = [r for r in records if r.kind == kind]
        if not include_expired:
            records = [r for r in records if not r.expired]
        if not include_superseded:
            records = [r for r in records if r.superseded_by is None]

        q_terms = set(query.lower().split()) if query else set()

        def score(r: MemoryRecord) -> float:
            s = r.salience * r.confidence
            if q_terms:
                content_terms = set(r.content.lower().split())
                overlap = len(q_terms & content_terms) / max(len(q_terms), 1)
                s += 0.5 * overlap
            # Gentle age decay so old, low-salience items sink.
            s *= math.exp(-r.age_seconds() / max(r.ttl_seconds, 1.0))
            return s

        records.sort(key=score, reverse=True)
        return records[:limit]

    def recall_semantic(self, query: str | None = None, limit: int = 10) -> list[MemoryRecord]:
        """Only semantic memory."""
        return self.recall(kind=MemoryKind.SEMANTIC, query=query, limit=limit)

    def get(self, record_id: str) -> MemoryRecord | None:
        for r in self.backend.all():
            if r.id == record_id:
                return r
        return None

    # -- maintenance ---------------------------------------------------
    def evict_expired(self) -> list[str]:
        """Remove expired records. Returns the ids that were evicted."""
        expired = [r.id for r in self.backend.all() if r.expired]
        for rid in expired:
            self.backend.remove(rid)
        return expired

    def clear_working(self) -> int:
        """Working memory is a scratchpad -- wipe it when the task ends."""
        working = [r.id for r in self.backend.all() if r.kind == MemoryKind.WORKING]
        for rid in working:
            self.backend.remove(rid)
        return len(working)

    def clear(self) -> None:
        self.backend.clear()

    def stats(self) -> dict:
        records = self.backend.all()
        out = {kind: 0 for kind in MemoryKind.ALL}
        for r in records:
            out[r.kind] = out.get(r.kind, 0) + 1
        out["total"] = len(records)
        out["expired"] = sum(1 for r in records if r.expired)
        out["rejected_writes"] = len(self.rejected_writes)
        return out

    def to_dict(self) -> dict:
        return {
            "stats": self.stats(),
            "rejected_writes": list(self.rejected_writes),
            "records": [r.to_dict() for r in self.backend.all()],
        }


def _smoke_test() -> None:
    store = MemoryStore()
    store.write(MemoryKind.SEMANTIC, "The support team works in Pacific time.", confidence=0.9)
    store.write(MemoryKind.PROCEDURAL, "Always run the smoke test before deploying.", confidence=1.0)
    store.write(MemoryKind.EPISODIC, "User asked about invoice 1001 on Monday.", confidence=0.8)
    store.write(MemoryKind.WORKING, "Draft reply in progress", confidence=1.0, ttl_seconds=60)

    print("stats:", store.stats())
    print()
    for r in store.recall(limit=5):
        print(f"  [{r.kind:10s}] sal={r.salience:.2f} {r.content}")

    # A low-confidence write is rejected.
    rejected = store.write(MemoryKind.SEMANTIC, "Maybe the moon is cheese?", confidence=0.05)
    print()
    print("low-confidence write rejected:", rejected is None)
    print("rejections:", store.rejected_writes)


if __name__ == "__main__":
    _smoke_test()
