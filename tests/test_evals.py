"""Tests for the evaluation harness.

Session 17. Evals are how you know if it is actually working. The teaching
point: a system that went from 90% to 55% because someone reworded one
line of a prompt is why evals live in CI.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evals import (  # noqa: E402
    EvalQuestion,
    EvalResult,
    EvalSuite,
    exact_match,
    fuzzy_match,
    normalize,
    score_suite,
)


def test_exact_match():
    assert exact_match("The answer is 42", "The answer is 42") == 1.0
    assert exact_match("42", "43") == 0.0
    assert 0.0 <= exact_match("a", "b") <= 1.0


def test_fuzzy_match_is_forgiving_about_case_and_punctuation():
    assert fuzzy_match("The Answer is 42.", "the answer is 42") == 1.0
    assert fuzzy_match("completely different", "the answer is 42") < 0.5


def test_normalize_strips_noise():
    assert normalize("  Hello,  World!  ") == normalize("hello world")


def test_suite_scores_a_prediction():
    suite = EvalSuite()
    suite.add(EvalQuestion(id="q1", question="What is the capital?",
                            reference="Sacramento", tags=["geo"]))
    suite.add(EvalQuestion(id="q2", question="What is 2+2?", reference="4", tags=["math"]))

    results = suite.score(
        lambda q: "Sacramento" if q.id == "q1" else "five"
    )
    assert len(results) == 2
    assert all(isinstance(r, EvalResult) for r in results)

    by_id = {r.question_id: r for r in results}
    assert by_id["q1"].score == 1.0
    assert by_id["q2"].score == 0.0


def test_report_summarises_scores():
    suite = EvalSuite()
    for i in range(10):
        suite.add(EvalQuestion(id=f"q{i}", question=f"Q{i}", reference="yes"))

    results = suite.score(lambda q: "yes" if int(q.id[1:]) < 7 else "no")
    report = suite.report(results)

    assert report["n"] == 10
    assert 0.6 <= report["mean_score"] <= 0.8
    assert 0.6 <= report["pass_rate"] <= 0.8
    assert "by_tag" in report and "failures" in report


def test_score_suite_helper_runs_the_whole_thing():
    suite = EvalSuite()
    suite.add(EvalQuestion(id="q1", question="Q1", reference="a"))
    report = score_suite(suite.questions, lambda q: "a")
    assert report["n"] == 1
    assert report["mean_score"] == 1.0
    assert report["pass_rate"] == 1.0


def test_eval_question_fields():
    q = EvalQuestion(id="q1", question="Q?", reference="R", tags=["a"])
    assert q.id == "q1"
    assert q.reference == "R"
    assert q.tags == ["a"]


def test_suite_extend():
    suite = EvalSuite()
    suite.extend([EvalQuestion(id="a", question="Q", reference="R"),
                  EvalQuestion(id="b", question="Q", reference="R")])
    assert len(suite.questions) == 2
