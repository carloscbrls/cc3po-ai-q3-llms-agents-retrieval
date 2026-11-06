"""Evaluation helpers: exact match, fuzzy match, and LLM-as-judge.

Session 17. The lesson of the session is not "write evals." It is:

    **An eval suite is a product feature with a regression alarm attached.**

Once the suite exists, injecting a known failure should make the score
drop by a known amount. If it does not, your suite is not measuring what
you think it is.

Everything here is offline and deterministic. The "LLM judge" has two
implementations:

- ``MockJudge``  -- deterministic, rule-based, no network. The default.
- ``LLMJudge``   -- a stub that shows you exactly where the real API call
  goes, with a clear "not implemented offline" message. We do not ship a
  version that silently spends your money.
"""

from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

__all__ = [
    "EvalQuestion",
    "EvalResult",
    "exact_match",
    "fuzzy_match",
    "normalize",
    "MockJudge",
    "LLMJudge",
    "EvalSuite",
    "score_suite",
]


def normalize(text: str) -> str:
    """Lowercase, strip punctuation/articles, collapse whitespace."""
    text = text.lower()
    text = "".join(ch if ch not in string.punctuation else " " for ch in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> float:
    """1.0 if the normalized strings match exactly, else 0.0."""
    return 1.0 if normalize(prediction) == normalize(reference) else 0.0


def fuzzy_match(prediction: str, reference: str, threshold: float = 0.8) -> float:
    """Token-level F1 between prediction and reference.

    This is the standard SQuAD-style metric. It gives partial credit for
    getting most of the answer right, which matters a lot when you are
    comparing against a judge.
    """
    pred_tokens = normalize(prediction).split()
    ref_tokens = normalize(reference).split()
    if not pred_tokens or not ref_tokens:
        return 1.0 if pred_tokens == ref_tokens else 0.0

    common = 0
    remaining = list(ref_tokens)
    for tok in pred_tokens:
        if tok in remaining:
            common += 1
            remaining.remove(tok)
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    # Convert F1 into a 0..1 "did it basically get it" score.
    return 1.0 if f1 >= threshold else f1


@dataclass
class EvalQuestion:
    """One question in an eval set."""

    id: str
    question: str
    reference: str
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalResult:
    """One scored question."""

    question_id: str
    score: float
    passed: bool
    prediction: str = ""
    reference: str = ""
    judge: str = "exact_match"
    reason: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "score": round(self.score, 4),
            "passed": self.passed,
            "prediction": self.prediction,
            "reference": self.reference,
            "judge": self.judge,
            "reason": self.reason,
            "tags": self.tags,
        }


class MockJudge:
    """A deterministic, offline stand-in for an LLM judge.

    It does not understand language. It checks whether the prediction
    contains the reference's key content -- which is enough to detect the
    kind of regression we inject in the lab, and it never flakes.
    """

    name = "mock"

    def judge(self, question: str, prediction: str, reference: str) -> tuple[float, str]:
        """Return (score in [0,1], reason)."""
        if not prediction.strip():
            return 0.0, "empty prediction"

        ref_norm = normalize(reference)
        pred_norm = normalize(prediction)

        if ref_norm and ref_norm in pred_norm:
            return 1.0, "reference contained verbatim"

        ref_tokens = set(ref_norm.split())
        pred_tokens = set(pred_norm.split())
        if not ref_tokens:
            return 1.0, "empty reference -- nothing to check"
        overlap = len(ref_tokens & pred_tokens) / len(ref_tokens)
        if overlap >= 0.8:
            return 1.0, f"covered {overlap:.0%} of the reference"
        if overlap >= 0.4:
            return 0.5, f"partial match ({overlap:.0%} of reference covered)"
        return 0.0, f"missing key content ({overlap:.0%} covered)"


class LLMJudge:
    """A stub showing exactly where a real LLM judge goes.

    We deliberately do **not** ship a version that calls out to a paid API
    by default. If you want to wire one up, subclass this and implement
    ``judge`` -- or use ``from_openai`` below.

    Args:
        model: the model you would use
        api_key: an optional key; when absent the judge runs in offline
            "echo mode" and delegates to a MockJudge.
        strict: when True, an offline judge raises instead of falling back.
    """

    name = "llm"

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None, strict: bool = False):
        self.model = model
        self.api_key = api_key
        self.strict = strict
        self._fallback = MockJudge()

    def judge(self, question: str, prediction: str, reference: str) -> tuple[float, str]:
        if not self.api_key:
            if self.strict:
                raise RuntimeError(
                    "LLMJudge has no API key. Set one, or use MockJudge for offline work."
                )
            score, reason = self._fallback.judge(question, prediction, reference)
            return score, f"[offline echo] {reason}"
        raise NotImplementedError(
            "Wire up your provider here. Use MockJudge for offline labs."
        )

    @classmethod
    def from_openai(cls, model: str = "gpt-4o-mini", api_key: str | None = None):
        """Return an LLMJudge configured for OpenAI-style chat completions.

        The HTTP call is intentionally not included -- students should write
        it themselves in the lab. This class gives them the shape to write
        into.
        """
        return cls(model=model, api_key=api_key)


class EvalSuite:
    """A set of questions plus a scorer."""

    def __init__(
        self,
        questions: Sequence[EvalQuestion] | None = None,
        judge: Any | None = None,
        pass_threshold: float = 0.5,
    ) -> None:
        self.questions: list[EvalQuestion] = list(questions or [])
        self.judge = judge if judge is not None else MockJudge()
        self.pass_threshold = pass_threshold

    def add(self, question: EvalQuestion) -> EvalQuestion:
        self.questions.append(question)
        return question

    def extend(self, questions: Iterable[EvalQuestion]) -> None:
        for q in questions:
            self.add(q)

    def score(
        self,
        predictions: dict[str, str] | Callable[[EvalQuestion], str],
    ) -> list[EvalResult]:
        """Score every question.

        Args:
            predictions: a mapping of question id -> prediction, or a
                callable that produces a prediction for a question.

        Returns:
            A list of :class:`EvalResult`, one per question.
        """
        results: list[EvalResult] = []
        for q in self.questions:
            if callable(predictions):
                pred = predictions(q)
            else:
                pred = predictions.get(q.id, "")
            try:
                score, reason = self.judge.judge(q.question, pred, q.reference)
            except Exception as exc:  # noqa: BLE001
                score, reason = 0.0, f"judge error: {type(exc).__name__}: {exc}"
            results.append(
                EvalResult(
                    question_id=q.id,
                    score=float(score),
                    passed=float(score) >= self.pass_threshold,
                    prediction=pred,
                    reference=q.reference,
                    judge=getattr(self.judge, "name", type(self.judge).__name__),
                    reason=reason,
                    tags=list(q.tags),
                )
            )
        return results

    def report(self, results: Sequence[EvalResult]) -> dict:
        """Aggregate results into a summary dict."""
        n = len(results)
        if n == 0:
            return {"n": 0, "pass_rate": 0.0, "mean_score": 0.0, "by_tag": {}, "failures": []}
        passed = sum(1 for r in results if r.passed)
        mean = sum(r.score for r in results) / n
        by_tag: dict[str, dict] = {}
        for r in results:
            for tag in r.tags or ["untagged"]:
                bucket = by_tag.setdefault(tag, {"n": 0, "passed": 0, "score_sum": 0.0})
                bucket["n"] += 1
                bucket["passed"] += 1 if r.passed else 0
                bucket["score_sum"] += r.score
        for tag, b in by_tag.items():
            b["pass_rate"] = b["passed"] / b["n"]
            b["mean_score"] = b["score_sum"] / b["n"]
            del b["score_sum"]
        return {
            "n": n,
            "passed": passed,
            "failed": n - passed,
            "pass_rate": passed / n,
            "mean_score": mean,
            "by_tag": by_tag,
            "failures": [r.to_dict() for r in results if not r.passed],
        }

    def to_dict(self) -> dict:
        return {
            "questions": [
                {
                    "id": q.id,
                    "question": q.question,
                    "reference": q.reference,
                    "tags": q.tags,
                }
                for q in self.questions
            ]
        }


def score_suite(
    questions: Sequence[EvalQuestion],
    predictions: dict[str, str] | Callable[[EvalQuestion], str],
    judge: Any | None = None,
) -> dict:
    """One-shot helper: build a suite, score it, return the report."""
    suite = EvalSuite(questions, judge=judge)
    return suite.report(suite.score(predictions))


def _smoke_test() -> None:
    qs = [
        EvalQuestion(id="q1", question="What is the capital of France?", reference="Paris"),
        EvalQuestion(id="q2", question="What is 2 + 2?", reference="4"),
    ]
    preds = {"q1": "The capital of France is Paris.", "q2": "five"}
    report = score_suite(qs, preds)
    print(json.dumps(report, indent=2)[:900])


if __name__ == "__main__":
    _smoke_test()
