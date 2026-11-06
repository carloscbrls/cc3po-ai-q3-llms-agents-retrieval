"""A deterministic, scripted LLM for labs.

Session 13-14. **No API key. No network. No model download.**

The mock LLM is a lookup table: you give it a prompt, it returns a
pre-scripted response. Some responses are text, some are *tool calls*. It
also tracks how many tokens it "consumed" and what it "cost", so the agent
tripwires have something real to measure.

Why this exists:

* Every student gets identical behaviour. Nobody is left waiting on a
  rate-limited API.
* The teaching moments are *scriptable*. We can make the agent call a tool
  that fails on the third turn, every single time.
* It costs nothing and runs offline.

If you want to point the agent at a real model, pass any object with the
same ``chat(messages, tools=None)`` interface. The agent does not care.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

__all__ = [
    "LLMResponse",
    "MockLLM",
    "ScriptedRule",
    "DEFAULT_SCRIPT",
    "build_default_llm",
]


@dataclass
class LLMResponse:
    """One assistant turn.

    Either ``content`` (plain text) or ``tool_calls`` (a list of calls).
    Both can be present: some models narrate *and* call a tool.
    """

    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "tool_calls": self.tool_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "finish_reason": self.finish_reason,
        }


@dataclass
class ScriptedRule:
    """Match a prompt by regex (or substring) and return a fixed response."""

    pattern: str
    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    regex: bool = True
    # Optional: only fire on the Nth matching turn (1-indexed).
    nth: int | None = None

    def matches(self, prompt: str, turn: int) -> bool:
        if self.nth is not None and turn != self.nth:
            return False
        if self.regex:
            return re.search(self.pattern, prompt, re.IGNORECASE) is not None
        return self.pattern.lower() in prompt.lower()

    def matches_topic(self, prompt: str) -> bool:
        """True if the TOPIC matches, ignoring the turn number."""
        if self.regex:
            return re.search(self.pattern, prompt, re.IGNORECASE) is not None
        return self.pattern.lower() in prompt.lower()


def _tool_call(name: str, **args) -> dict:
    """Build an OpenAI-style tool call dict."""
    return {
        "id": f"call_{name}_{abs(hash(json.dumps(args, sort_keys=True))) % 100000}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


# The default script. Everything the labs need, in order.
#
# Each conversation is a PAIR of rules: turn 1 calls a tool, turn 2 reports
# the result. The `nth` field makes that explicit -- a rule with nth=1 only
# fires on the first matching turn, nth=2 only on the second.
#
# Why this matters: without `nth`, the first matching rule wins EVERY time,
# so the agent calls get_weather forever and never completes. That is a
# realistic bug shape, but it is not the one we want to teach in S14.
# We want: tool call -> tool result -> final answer. Three turns, done.
DEFAULT_SCRIPT: list[ScriptedRule] = [
    # The impossible task comes FIRST so it wins over the generic database
    # rule. This is the S14 teaching moment: the agent claims success
    # without ever having called a tool, and nothing in the loop notices.
    ScriptedRule(
        pattern=r"impossible|unicorn|self[- ]driving|do my taxes|find the meaning of life",
        content="Done! I successfully completed the task. Everything is in place now.",
    ),
    ScriptedRule(
        pattern=r"weather",
        content="Let me check the weather for you.",
        tool_calls=[_tool_call("get_weather", city="Stockton, CA", units="f")],
        nth=1,
    ),
    ScriptedRule(
        pattern=r"weather",
        content="It is currently 72 F and partly cloudy in Stockton.",
        nth=2,
    ),
    ScriptedRule(
        pattern=r"orders?",
        content="Looking up the orders.",
        tool_calls=[_tool_call("search_orders", customer_id="C-1042", status="any")],
        nth=1,
    ),
    ScriptedRule(
        pattern=r"orders?",
        content="I found 2 orders for customer C-1042.",
        nth=2,
    ),
    ScriptedRule(
        pattern=r"calculate|compute|\d+\s*[\+\-\*/]",
        content="",
        tool_calls=[_tool_call("calculate", expression="2 * (3 + 4)")],
        nth=1,
    ),
    ScriptedRule(
        pattern=r"calculate|compute|\d+\s*[\+\-\*/]",
        content="The result is 14.",
        nth=2,
    ),
    # The unreliable database -- used to demonstrate graceful failure.
    ScriptedRule(
        pattern=r"database|records",
        content="Querying the database.",
        tool_calls=[_tool_call("flaky_database", mode="crash")],
        nth=1,
    ),
    ScriptedRule(
        pattern=r"database|records",
        content="The database is unavailable, so I could not retrieve the records.",
        nth=2,
    ),
    ScriptedRule(
        pattern=r"hello|hi\b|hey",
        content="Hello! What can I help you with?",
    ),
]


class MockLLM:
    """A deterministic LLM stand-in.

    Args:
        script: list of :class:`ScriptedRule`. The first match wins.
        default_content: what to return when nothing matches.
        cost_per_1k_prompt_tokens: simulated price, used by the agent's
            cost-ceiling tripwire.
        price_per_1k_completion_tokens: simulated completion price.
    """

    def __init__(
        self,
        script: Sequence[ScriptedRule] | None = None,
        default_content: str = "I am not sure how to help with that.",
        cost_per_1k_prompt_tokens: float = 0.0005,
        price_per_1k_completion_tokens: float = 0.0015,
        name: str = "MockLLM",
    ) -> None:
        self.script = list(script) if script is not None else list(DEFAULT_SCRIPT)
        self.default_content = default_content
        self.cost_per_1k_prompt_tokens = cost_per_1k_prompt_tokens
        self.price_per_1k_completion_tokens = price_per_1k_completion_tokens
        self.name = name
        self.calls: list[dict] = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        # Turn counter for the CURRENT conversation. Reset whenever a fresh
        # conversation starts (see chat()).
        self._turn = 0

    # -- helpers ------------------------------------------------------
    @staticmethod
    def _estimate_tokens(messages: Sequence[dict]) -> int:
        """Rough token count: ~4 characters per token."""
        chars = 0
        for m in messages:
            chars += len(str(m.get("content", "")))
            for call in m.get("tool_calls", []) or []:
                chars += len(json.dumps(call))
        return max(1, chars // 4)

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def total_cost(self) -> float:
        return (
            self.total_prompt_tokens / 1000.0 * self.cost_per_1k_prompt_tokens
            + self.total_completion_tokens / 1000.0 * self.price_per_1k_completion_tokens
        )

    def reset(self) -> None:
        self.calls.clear()
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    # -- the main entry point -----------------------------------------
    def chat(self, messages: Sequence[dict], tools: Sequence[dict] | None = None) -> LLMResponse:
        """Return the next assistant turn.

        The signature matches OpenAI's ``client.chat.completions.create``
        closely enough that swapping in a real model is a one-line change.
        """
        prompt = "\n".join(str(m.get("content", "")) for m in messages)
        prompt_tokens = self._estimate_tokens(messages)

        # A conversation is the message list. If the caller passes a fresh
        # single user message (no assistant/tool turns yet), we are starting a
        # new conversation, so reset the turn counter.
        if len(messages) == 1 and messages[0].get("role") == "user":
            self._turn = 0
        self._turn += 1

        rule = None
        for candidate in self.script:
            if candidate.matches(prompt, turn=self._turn):
                rule = candidate
                break

        if rule is None:
            # Fallback: the turn number didn't line up (e.g. turn 3 of a
            # two-rule script). Use the LAST rule for this topic so the
            # conversation terminates with the "final answer" content
            # instead of looping forever.
            topic_matches = [c for c in self.script if c.matches_topic(prompt)]
            if topic_matches:
                rule = topic_matches[-1]

        if rule is None:
            content = self.default_content
            tool_calls: list[dict] = []
        else:
            content = rule.content
            tool_calls = list(rule.tool_calls)

        completion_tokens = self._estimate_tokens(
            [{"content": content}] + [{"tool_calls": tool_calls}]
        )

        self.calls.append({"messages": list(messages), "response": content, "tool_calls": tool_calls})
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason="tool_calls" if tool_calls else "stop",
        )

    # Convenience alias so this looks like a chat client.
    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        return self.chat([{"role": "user", "content": prompt}], **kwargs)


def build_default_llm(**kwargs) -> MockLLM:
    """A MockLLM with the standard lab script."""
    return MockLLM(**kwargs)


def _smoke_test() -> None:
    llm = MockLLM()
    r1 = llm.chat([{"role": "user", "content": "What is the weather in Stockton?"}])
    print("turn 1:", r1.content, r1.tool_calls)
    r2 = llm.chat(
        [
            {"role": "user", "content": "What is the weather in Stockton?"},
            {"role": "assistant", "content": r1.content, "tool_calls": r1.tool_calls},
            {"role": "tool", "content": json.dumps({"temp": 72}), "name": "get_weather"},
        ]
    )
    print("turn 2:", r2.content)
    print("cost so far: $", round(llm.total_cost, 8))
    print("tokens:", llm.total_tokens)


if __name__ == "__main__":
    _smoke_test()
