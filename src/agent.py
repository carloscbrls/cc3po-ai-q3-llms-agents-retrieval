"""The agent loop, with tripwires.

Session 14. This is the most important file in the quarter.

An agent is a loop:

    while not done:
        response = llm.chat(messages, tools=schemas)
        if response has no tool calls: done
        for each tool call: result = dispatch(...)
        append results to messages

That is the whole thing. It is also the whole thing that goes wrong in
production. So the loop here has **three mandatory tripwires**:

1. **Max iterations.** A hard cap on the loop. No infinite loops, ever.
2. **Cost ceiling.** A hard cap on simulated spend. The agent stops and
   says so rather than quietly burning money.
3. **Step logging.** Every step is recorded: what was asked, what tool ran,
   what came back, what it cost. If you cannot see the loop, you cannot
   debug it.

The other thing this module teaches: **an agent can confidently claim
success at a task it did not do.** Nothing in the loop verifies the claim.
The tripwires are what keep that survivable.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from mock_llm import LLMResponse
from tools import ToolRegistry, ToolResult, registry as default_registry

__all__ = [
    "AgentConfig",
    "AgentStep",
    "AgentRun",
    "Agent",
    "StopReason",
]


class StopReason:
    """Why the agent loop ended. Always recorded."""

    COMPLETED = "completed"
    MAX_ITERATIONS = "max_iterations"
    COST_CEILING = "cost_ceiling"
    TOOL_ERROR = "tool_error"
    LLM_ERROR = "llm_error"


@dataclass
class AgentConfig:
    """Tripwire settings. The defaults are deliberately conservative."""

    max_iterations: int = 8
    max_cost_usd: float = 0.05
    max_tool_errors: int = 3
    system_prompt: str = (
        "You are a helpful assistant. Use the available tools when they help. "
        "Be concise."
    )
    # Stop after this many consecutive tool failures.
    stop_on_tool_error: bool = False


@dataclass
class AgentStep:
    """One iteration of the loop, fully logged."""

    iteration: int
    llm_content: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "llm_content": self.llm_content,
            "tool_calls": self.tool_calls,
            "tool_results": [r.to_dict() for r in self.tool_results],
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost": round(self.cost, 8),
            "elapsed_ms": round(self.elapsed_ms, 3),
        }


@dataclass
class AgentRun:
    """The result of one agent run."""

    answer: str = ""
    steps: list[AgentStep] = field(default_factory=list)
    stop_reason: str = StopReason.COMPLETED
    total_cost: float = 0.0
    total_tokens: int = 0
    iterations: int = 0
    tool_errors: int = 0

    @property
    def succeeded(self) -> bool:
        return self.stop_reason == StopReason.COMPLETED

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "steps": [s.to_dict() for s in self.steps],
            "stop_reason": self.stop_reason,
            "total_cost": round(self.total_cost, 8),
            "total_tokens": self.total_tokens,
            "iterations": self.iterations,
            "tool_errors": self.tool_errors,
        }

    def render_log(self) -> str:
        """A human-readable trace. This is what you show the class."""
        lines = []
        for step in self.steps:
            lines.append(f"--- step {step.iteration} ---")
            if step.llm_content:
                lines.append(f"  llm: {step.llm_content}")
            for call in step.tool_calls:
                fn = call.get("function", {})
                lines.append(f"  call: {fn.get('name')}({fn.get('arguments')})")
            for res in step.tool_results:
                status = "ok" if res.ok else f"ERROR: {res.error}"
                lines.append(f"  result[{res.tool}]: {status}")
            lines.append(
                f"  tokens: {step.prompt_tokens}+{step.completion_tokens}  "
                f"cost: ${step.cost:.8f}"
            )
        lines.append(f"=== stop_reason: {self.stop_reason} ===")
        lines.append(f"=== total cost: ${self.total_cost:.8f} ===")
        return "\n".join(lines)


class Agent:
    """A tool-using agent with mandatory tripwires.

    Args:
        llm: anything with ``.chat(messages, tools=...)`` returning an
             :class:`~mock_llm.LLMResponse`. The MockLLM satisfies this.
        tools: a :class:`~tools.ToolRegistry`. Defaults to the built-in one.
        config: tripwire settings.
    """

    def __init__(
        self,
        llm: Any,
        tools: ToolRegistry | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools if tools is not None else default_registry
        self.config = config if config is not None else AgentConfig()

    # ------------------------------------------------------------------
    def run(self, task: str, on_step: Callable[[AgentStep], None] | None = None) -> AgentRun:
        """Run the loop until it finishes, trips a tripwire, or errors out."""
        run = AgentRun()
        messages: list[dict] = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": task},
        ]
        schemas = self.tools.schemas()
        consecutive_errors = 0

        for iteration in range(1, self.config.max_iterations + 1):
            step = AgentStep(iteration=iteration)
            start = time.perf_counter()

            # --- call the model ---------------------------------------
            try:
                response: LLMResponse = self.llm.chat(messages, tools=schemas)
            except Exception as exc:  # noqa: BLE001
                run.stop_reason = StopReason.LLM_ERROR
                run.answer = f"The model call failed: {type(exc).__name__}: {exc}"
                run.iterations = iteration - 1
                return run

            step.llm_content = response.content
            step.prompt_tokens = response.prompt_tokens
            step.completion_tokens = response.completion_tokens
            step.cost = self._cost_of(response)

            # --- cost tripwire (check BEFORE running tools) -----------
            run.total_cost += step.cost
            run.total_tokens += response.prompt_tokens + response.completion_tokens
            if run.total_cost > self.config.max_cost_usd:
                run.steps.append(step)
                run.stop_reason = StopReason.COST_CEILING
                run.answer = (
                    f"Stopped: cost ceiling reached "
                    f"(${run.total_cost:.6f} > ${self.config.max_cost_usd:.6f})."
                )
                run.iterations = iteration
                if on_step:
                    on_step(step)
                return run

            # --- no tool calls -> we are done ------------------------
            if not response.tool_calls:
                step.elapsed_ms = (time.perf_counter() - start) * 1000
                run.steps.append(step)
                run.answer = response.content
                run.stop_reason = StopReason.COMPLETED
                run.iterations = iteration
                if on_step:
                    on_step(step)
                return run

            # --- run the tools ---------------------------------------
            messages.append(
                {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": response.tool_calls,
                }
            )
            step.tool_calls = list(response.tool_calls)

            for call in response.tool_calls:
                fn = call.get("function", {})
                name = fn.get("name", "")
                raw_args = fn.get("arguments", "{}")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args or {})
                except json.JSONDecodeError as exc:
                    args = {}
                    result = ToolResult(
                        tool=name, ok=False, error=f"invalid JSON arguments: {exc}"
                    )
                else:
                    result = self.tools.dispatch(name, args)

                step.tool_results.append(result)
                messages.append(
                    {
                        "role": "tool",
                        "name": name,
                        "tool_call_id": call.get("id", ""),
                        "content": json.dumps(result.to_dict()),
                    }
                )

                if not result.ok:
                    run.tool_errors += 1
                    consecutive_errors += 1
                else:
                    consecutive_errors = 0

            step.elapsed_ms = (time.perf_counter() - start) * 1000
            run.steps.append(step)
            if on_step:
                on_step(step)

            # --- tool-error tripwire ---------------------------------
            if consecutive_errors >= self.config.max_tool_errors:
                run.stop_reason = StopReason.TOOL_ERROR
                run.answer = (
                    f"Stopped: {consecutive_errors} consecutive tool errors. "
                    "The tools are not working."
                )
                run.iterations = iteration
                return run

            if self.config.stop_on_tool_error and step.tool_results and not all(
                r.ok for r in step.tool_results
            ):
                run.stop_reason = StopReason.TOOL_ERROR
                run.answer = "Stopped: a tool call failed."
                run.iterations = iteration
                return run

        # --- iteration tripwire -------------------------------------
        run.stop_reason = StopReason.MAX_ITERATIONS
        run.iterations = self.config.max_iterations
        run.answer = (
            f"Stopped: reached the maximum of {self.config.max_iterations} iterations "
            "without finishing."
        )
        return run

    # ------------------------------------------------------------------
    def _cost_of(self, response: LLMResponse) -> float:
        """Cost of one response, using the LLM's own price table if it has one."""
        per_1k_prompt = getattr(self.llm, "cost_per_1k_prompt_tokens", 0.0005)
        per_1k_completion = getattr(self.llm, "price_per_1k_completion_tokens", 0.0015)
        return (
            response.prompt_tokens / 1000.0 * per_1k_prompt
            + response.completion_tokens / 1000.0 * per_1k_completion
        )


def _smoke_test() -> None:
    from mock_llm import MockLLM

    agent = Agent(MockLLM(), config=AgentConfig(max_iterations=6))
    run = agent.run("What is the weather in Stockton?")
    print(run.render_log())
    print()
    print("ANSWER:", run.answer)
    print("succeeded:", run.succeeded)


if __name__ == "__main__":
    _smoke_test()
