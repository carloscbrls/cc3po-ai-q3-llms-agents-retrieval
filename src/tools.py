"""JSON-schema tools, a registry, and a dispatcher that never crashes.

Session 13. The whole point of this module is the last function in it:
``dispatch``. A tool call can fail. The network can be down. The arguments
can be garbage. **A failing tool must return an error, not take down the
agent.** That is the entire lesson.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = [
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "ToolError",
    "register",
    "dispatch",
    "call_tool",
    "list_tools",
    "TOOL_SCHEMAS",
]


class ToolError(Exception):
    """Raised inside a tool when something goes wrong.

    The dispatcher catches this and turns it into a structured error result
    the agent can reason about (and retry around).
    """


@dataclass
class ToolResult:
    """The outcome of one tool call. Never an exception at the call site."""

    tool: str
    ok: bool
    content: Any = None
    error: str | None = None
    elapsed_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "tool": self.tool,
            "ok": self.ok,
            "content": self.content,
            "error": self.error,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "metadata": self.metadata,
        }

    def __str__(self) -> str:
        if self.ok:
            return f"ToolResult({self.tool}, ok=True, content={self.content!r})"
        return f"ToolResult({self.tool}, ok=False, error={self.error!r})"


@dataclass
class Tool:
    """A callable plus the JSON schema that describes it to a model."""

    name: str
    description: str
    parameters: dict
    fn: Callable[..., Any]

    def schema(self) -> dict:
        """The OpenAI-style function schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """A name -> Tool map with JSON-schema validation on the way in."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict,
        fn: Callable[..., Any],
    ) -> Tool:
        tool = Tool(name=name, description=description, parameters=parameters, fn=fn)
        self._tools[name] = tool
        return tool

    def tool(self, name: str, description: str, parameters: dict):
        """Decorator form: ``@registry.tool(...)``."""

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.register(name, description, parameters, fn)
            return fn

        return decorator

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict]:
        return [self._tools[n].schema() for n in self.names()]

    def validate_args(self, name: str, args: dict) -> None:
        """Check required args and primitive types. Raises ToolError."""
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError(f"unknown tool: {name!r}")
        props = tool.parameters.get("properties", {})
        required = tool.parameters.get("required", [])

        for key in required:
            if key not in args:
                raise ToolError(f"missing required argument {key!r} for tool {name!r}")

        for key, value in args.items():
            spec = props.get(key)
            if not spec:
                continue
            expected = spec.get("type")
            if expected == "string" and not isinstance(value, str):
                raise ToolError(f"argument {key!r} must be a string, got {type(value).__name__}")
            if expected == "integer" and not isinstance(value, int):
                raise ToolError(f"argument {key!r} must be an integer, got {type(value).__name__}")
            if expected == "number" and not isinstance(value, (int, float)):
                raise ToolError(f"argument {key!r} must be a number, got {type(value).__name__}")
            if expected == "boolean" and not isinstance(value, bool):
                raise ToolError(f"argument {key!r} must be a boolean, got {type(value).__name__}")
            # Enum check: if the schema declares an allowed set, enforce it.
            # This matters in production -- a model that sends units="kelvin"
            # when the tool only understands "c"/"f" should get a clean
            # error, not a silently wrong temperature.
            allowed = spec.get("enum")
            if allowed is not None and value not in allowed:
                raise ToolError(
                    f"argument {key!r} must be one of {list(allowed)}, got {value!r}"
                )

    def dispatch(self, name: str, args: dict | None = None) -> ToolResult:
        """Run a tool. **Never raises** -- always returns a ToolResult."""
        args = args or {}
        start = time.perf_counter()
        try:
            tool = self._tools.get(name)
            if tool is None:
                raise ToolError(f"unknown tool: {name!r} (have: {self.names()})")
            self.validate_args(name, args)
            content = tool.fn(**args)
            elapsed = (time.perf_counter() - start) * 1000
            return ToolResult(tool=name, ok=True, content=content, elapsed_ms=elapsed)
        except ToolError as exc:
            elapsed = (time.perf_counter() - start) * 1000
            return ToolResult(tool=name, ok=False, error=str(exc), elapsed_ms=elapsed)
        except TypeError as exc:
            elapsed = (time.perf_counter() - start) * 1000
            return ToolResult(tool=name, ok=False, error=f"bad arguments: {exc}", elapsed_ms=elapsed)
        except Exception as exc:  # noqa: BLE001 -- the whole point
            elapsed = (time.perf_counter() - start) * 1000
            return ToolResult(
                tool=name,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                elapsed_ms=elapsed,
            )


# ----------------------------------------------------------------------
# The default registry
# ----------------------------------------------------------------------
registry = ToolRegistry()


@registry.tool(
    "get_weather",
    "Get the current weather for a city.",
    {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name, e.g. 'Stockton, CA'"},
            "units": {"type": "string", "enum": ["c", "f"], "description": "Temperature units"},
        },
        "required": ["city"],
    },
)
def get_weather(city: str, units: str = "f") -> dict:
    """Deterministic fake weather so the lab never needs the network."""
    seed = sum(ord(c) for c in city.lower())
    temps_c = 8 + (seed % 25)
    if units.lower() == "c":
        temp = temps_c
    else:
        temp = round(temps_c * 9 / 5 + 32, 1)
    conditions = ["clear", "partly cloudy", "overcast", "light rain", "windy"][seed % 5]
    return {"city": city, "temp": temp, "units": units, "conditions": conditions}


@registry.tool(
    "search_orders",
    "Search customer orders by customer id and optional status.",
    {
        "type": "object",
        "properties": {
            "customer_id": {"type": "string", "description": "Customer identifier"},
            "status": {"type": "string", "description": "Order status filter"},
        },
        "required": ["customer_id"],
    },
)
def search_orders(customer_id: str, status: str = "any") -> list[dict]:
    """Deterministic fake orders."""
    seed = sum(ord(c) for c in customer_id)
    n = 1 + (seed % 3)
    statuses = ["shipped", "processing", "delivered", "cancelled"]
    out = []
    for i in range(n):
        st = statuses[(seed + i) % len(statuses)] if status == "any" else status
        out.append(
            {
                "order_id": f"ORD-{seed % 10000:04d}-{i}",
                "customer_id": customer_id,
                "status": st,
                "total_usd": round(10 + (seed * (i + 3)) % 400, 2),
            }
        )
    return out


@registry.tool(
    "calculate",
    "Evaluate a simple arithmetic expression such as '2 * (3 + 4)'.",
    {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Arithmetic expression"}
        },
        "required": ["expression"],
    },
)
def calculate(expression: str) -> dict:
    """Evaluate arithmetic. Uses ast with a strict whitelist -- never eval."""
    import ast
    import operator

    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ToolError(f"unsupported constant: {node.value!r}")
        if isinstance(node, ast.BinOp):
            op = ops.get(type(node.op))
            if op is None:
                raise ToolError(f"unsupported operator: {type(node.op).__name__}")
            return op(_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            op = ops.get(type(node.op))
            if op is None:
                raise ToolError(f"unsupported unary operator: {type(node.op).__name__}")
            return op(_eval(node.operand))
        raise ToolError(f"unsupported expression node: {type(node).__name__}")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"could not parse expression: {exc}") from exc

    try:
        value = _eval(tree)
    except ToolError:
        raise
    except ZeroDivisionError as exc:
        raise ToolError("division by zero") from exc
    except Exception as exc:  # noqa: BLE001
        raise ToolError(f"could not evaluate: {exc}") from exc

    return {"expression": expression, "result": value}


@registry.tool(
    "flaky_database",
    "A deliberately unreliable tool used to demonstrate graceful error handling.",
    {
        "type": "object",
        "properties": {
            "mode": {"type": "string", "description": "One of: ok, timeout, crash, badargs"},
        },
        "required": ["mode"],
    },
)
def flaky_database(mode: str = "ok") -> dict:
    """Fails in a controlled way so students can see the agent recover."""
    if mode == "timeout":
        time.sleep(0.05)
        raise TimeoutError("database did not respond within 5s")
    if mode == "crash":
        raise RuntimeError("connection reset by peer")
    if mode == "badargs":
        raise ValueError("malformed row: expected 12 columns, got 3")
    return {"rows": 12, "mode": mode}


def register(name: str, description: str, parameters: dict, fn: Callable[..., Any]) -> Tool:
    """Register a tool on the default registry."""
    return registry.register(name, description, parameters, fn)


def dispatch(name: str, args: dict | None = None) -> ToolResult:
    """Call a tool on the default registry. Never raises."""
    return registry.dispatch(name, args)


def call_tool(name: str, **kwargs) -> ToolResult:
    """Convenience: ``call_tool("get_weather", city="Stockton")``."""
    return registry.dispatch(name, kwargs)


def list_tools() -> list[dict]:
    """JSON schemas for every registered tool."""
    return registry.schemas()


# Schemas as a plain list -- handy for pasting into a chat completion request.
TOOL_SCHEMAS = list_tools()


def _smoke_test() -> None:
    print("registered tools:", registry.names())
    print()
    print(json.dumps(list_tools()[0], indent=2)[:400])
    print()
    print("ok  :", dispatch("get_weather", {"city": "Stockton, CA"}))
    print("calc:", dispatch("calculate", {"expression": "2 * (3 + 4)"}))
    print("bad :", dispatch("get_weather", {"city": 123}))
    print("none:", dispatch("does_not_exist", {}))
    print("fail:", dispatch("flaky_database", {"mode": "crash"}))


if __name__ == "__main__":
    _smoke_test()
