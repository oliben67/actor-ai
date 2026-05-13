"""Deterministic fake LLM providers for running examples without API keys.

To switch any example to a real LLM, replace the provider assignment with one
of the real providers exported by actor_ai, e.g.:

    from actor_ai import Claude, GPT, Gemini, Mistral, DeepSeek, LiteLLM

    provider = Claude()                     # requires ANTHROPIC_API_KEY
    provider = GPT("gpt-4o")               # requires OPENAI_API_KEY
    provider = Gemini("gemini-2.0-flash")  # requires GOOGLE_API_KEY
    provider = LiteLLM("openai/gpt-4o")   # requires OPENAI_API_KEY
"""

# Future imports (must occur at the beginning of the file):
from __future__ import annotations

# Standard library imports:
from collections.abc import Callable

# Local imports:
from actor_ai.accounting import MonitoringContext, UsageSummary
from actor_ai.providers.base import LLMProvider


class ScriptedProvider(LLMProvider):
    """Returns preset replies in order; repeats the last reply once exhausted.

    All calls are recorded in ``self.calls`` for inspection.
    """

    def __init__(
        self,
        replies: list[str],
        usage: UsageSummary | None = None,
    ) -> None:
        self.model = "scripted/demo"
        self._replies = list(replies)
        self._index = 0
        self._usage = usage or UsageSummary(input_tokens=50, output_tokens=30)
        self.calls: list[dict] = []

    def run(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
        dispatcher: Callable[[str, dict], object],
        max_tokens: int,
        *,
        on_usage: Callable[[UsageSummary], None] | None = None,
        monitoring_context: MonitoringContext | None = None,
    ) -> str:
        self.calls.append(
            {
                "system": system,
                "messages": list(messages),
                "monitoring_context": monitoring_context,
            }
        )
        if on_usage is not None:
            on_usage(self._usage)
        idx = min(self._index, len(self._replies) - 1)
        reply = self._replies[idx]
        self._index += 1
        return reply


class ToolCallingProvider(LLMProvider):
    """Calls a named tool once, then returns a preset final reply.

    Simulates the LLM dispatching a single tool call and then composing a
    natural-language reply from the tool result.  ``tool_result`` holds the
    value returned by the tool after the call completes.
    """

    def __init__(
        self,
        tool_name: str,
        tool_args: dict,
        final_reply: str,
        usage: UsageSummary | None = None,
    ) -> None:
        self.model = "scripted/demo"
        self._tool_name = tool_name
        self._tool_args = tool_args
        self._final_reply = final_reply
        self._dispatched = False
        self.tool_result: object = None
        self._usage = usage or UsageSummary(input_tokens=80, output_tokens=40)

    def run(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
        dispatcher: Callable[[str, dict], object],
        max_tokens: int,
        *,
        on_usage: Callable[[UsageSummary], None] | None = None,
        monitoring_context: MonitoringContext | None = None,
    ) -> str:
        if not self._dispatched:
            self._dispatched = True
            self.tool_result = dispatcher(self._tool_name, self._tool_args)
        if on_usage is not None:
            on_usage(self._usage)
        return self._final_reply
