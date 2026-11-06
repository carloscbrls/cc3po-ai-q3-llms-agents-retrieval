"""Tests for the agent loop and its tripwires.

Session 14. The tripwires are not decoration -- they are the lesson. Each
test here asserts that a specific way an agent goes wrong is actually
prevented.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent import Agent, AgentConfig, StopReason  # noqa: E402
from mock_llm import MockLLM, ScriptedRule, build_default_llm  # noqa: E402
import tools as T  # noqa: E402


def test_agent_completes_a_normal_task():
    """A two-turn conversation: tool call, then final answer."""
    llm = build_default_llm()
    agent = Agent(llm=llm, tools=T.registry, config=AgentConfig(max_iterations=8))
    run = agent.run("What is the weather in Stockton?")

    assert run.stop_reason == StopReason.COMPLETED
    assert run.succeeded
    assert run.iterations == 2
    assert "72" in run.answer
    assert len(run.steps) == 2
    assert run.total_cost > 0


def test_agent_respects_the_iteration_cap():
    """The most important test in the file.

    An agent with a script that NEVER stops must still terminate. Without
    this cap the loop runs forever and burns real money in production.
    """
    looping_rule = ScriptedRule(
        pattern=r"loop",
        content="Calling a tool again.",
        tool_calls=[{"id": "c1", "type": "function",
                     "function": {"name": "get_weather",
                                  "arguments": '{"city": "Stockton, CA", "units": "f"}'}}],
    )
    llm = MockLLM(script=[looping_rule])
    agent = Agent(llm=llm, tools=T.registry, config=AgentConfig(max_iterations=4))
    run = agent.run("please loop forever")

    assert run.stop_reason == StopReason.MAX_ITERATIONS
    assert run.iterations == 4
    assert len(run.steps) == 4
    assert not run.succeeded


def test_agent_respects_the_cost_ceiling():
    """A cheap-per-token model with a tight ceiling must still trip."""
    llm = build_default_llm(cost_per_1k_prompt_tokens=0.05, price_per_1k_completion_tokens=0.05)
    agent = Agent(llm=llm, tools=T.registry, config=AgentConfig(max_iterations=50, max_cost_usd=0.0002))
    run = agent.run("What is the weather in Stockton?")

    assert run.stop_reason == StopReason.COST_CEILING
    assert run.total_cost > 0.0002
    assert "cost ceiling" in run.answer.lower()


def test_tool_failure_is_handled_not_crashed():
    """A failing tool returns an error result; the agent keeps going."""
    llm = build_default_llm()
    agent = Agent(llm=llm, tools=T.registry, config=AgentConfig(max_iterations=6))
    run = agent.run("Look up the records in the database")

    assert run.tool_errors >= 1
    assert run.stop_reason in (StopReason.COMPLETED, StopReason.TOOL_ERROR)
    # The agent must not have crashed -- it produced a human-readable answer.
    assert isinstance(run.answer, str) and run.answer


def test_agent_can_confidently_claim_false_success():
    """The S14 teaching moment, preserved as a regression test.

    The scripted "impossible" task returns 'Done!' WITHOUT calling any
    tool. The loop has no idea. This is exactly why production agents need
    evals and verification -- and why 'the agent said it was done' is not
    evidence.
    """
    llm = build_default_llm()
    agent = Agent(llm=llm, tools=T.registry, config=AgentConfig(max_iterations=5))
    run = agent.run("Find the meaning of life using only the database tool")

    called = [c["function"]["name"] for s in run.steps for c in s.tool_calls]
    assert called == [], "the scripted impossible task should call NO tools"
    assert "successfully completed" in run.answer
    # The run "succeeded" as far as the loop is concerned -- and it is wrong.
    assert run.stop_reason == StopReason.COMPLETED


def test_every_step_is_logged():
    """If you cannot see the loop, you cannot debug it."""
    llm = build_default_llm()
    agent = Agent(llm=llm, tools=T.registry, config=AgentConfig(max_iterations=8))
    run = agent.run("What is the weather in Stockton?")

    assert len(run.steps) == run.iterations
    step = run.steps[0]
    assert step.iteration == 1
    assert step.tool_calls, "step 1 should have called a tool"
    assert step.tool_results, "step 1 should have a tool result"
    assert step.tool_results[0].ok
    assert step.prompt_tokens > 0

    log = run.render_log()
    assert "get_weather" in log
    assert "stop_reason" in log


def test_on_step_callback_fires_for_every_step():
    llm = build_default_llm()
    agent = Agent(llm=llm, tools=T.registry, config=AgentConfig(max_iterations=8))
    seen = []
    agent.run("What is the weather in Stockton?", on_step=seen.append)
    assert len(seen) == 2


def test_run_serialises_to_dict():
    llm = build_default_llm()
    agent = Agent(llm=llm, tools=T.registry, config=AgentConfig(max_iterations=8))
    d = agent.run("What is the weather in Stockton?").to_dict()

    assert d["stop_reason"] == StopReason.COMPLETED
    assert isinstance(d["steps"], list) and len(d["steps"]) == 2
    assert "total_cost" in d and "total_tokens" in d
