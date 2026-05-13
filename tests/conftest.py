"""Shared fixtures and test helpers."""

# Future imports (must occur at the beginning of the file):
from __future__ import annotations

# Standard library imports:
import contextlib
from collections.abc import Callable

# Third party imports:
import pykka
import pytest

# Local imports:
from actor_ai.accounting import MonitoringContext, UsageSummary
from actor_ai.providers.base import LLMProvider


class FakeProvider(LLMProvider):
    """Deterministic fake provider — returns preset replies, records every call."""

    def __init__(
        self,
        replies: list[str] | None = None,
        usage: UsageSummary | None = None,
    ) -> None:
        self.model = "fake-model"
        self._replies = list(replies) if replies else ["fake reply"]
        self._index = 0
        self.calls: list[dict] = []
        # Usage to report via on_usage callback (simulates token counts)
        self._usage = usage or UsageSummary(input_tokens=10, output_tokens=5)

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
                "tools": list(tools),
                "max_tokens": max_tokens,
                "monitoring_context": monitoring_context,
            }
        )
        if on_usage is not None:
            on_usage(self._usage)
        idx = min(self._index, len(self._replies) - 1)
        reply = self._replies[idx]
        self._index += 1
        return reply


class ToolCallingFakeProvider(LLMProvider):
    """Fake provider that calls one named tool once, then returns a final reply."""

    def __init__(
        self,
        tool_name: str,
        tool_args: dict,
        final_reply: str = "tool result processed",
        usage: UsageSummary | None = None,
    ) -> None:
        self.model = "fake-model"
        self._tool_name = tool_name
        self._tool_args = tool_args
        self._final_reply = final_reply
        self._dispatched = False
        self.tool_result: object = None
        self._usage = usage or UsageSummary(input_tokens=20, output_tokens=10)

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


@pytest.fixture
def actor_factory():
    """Factory that starts AIActor subclasses and stops them after the test."""
    refs: list[pykka.ActorRef] = []

    def create(actor_cls, **start_kwargs):
        ref = actor_cls.start(**start_kwargs)
        refs.append(ref)
        return ref

    yield create

    for ref in refs:
        with contextlib.suppress(pykka.ActorDeadError):
            ref.stop()


@pytest.fixture(autouse=True)
def stop_all_actors():
    """Safety net: stop any actors that leaked from a test."""
    yield
    pykka.ActorRegistry.stop_all()
