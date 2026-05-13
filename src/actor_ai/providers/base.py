# Future imports (must occur at the beginning of the file):
from __future__ import annotations

# Standard library imports:
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..accounting import MonitoringContext, UsageSummary


class LLMProvider(ABC):
    """Protocol for LLM backends.

    Subclasses implement the full agentic loop for a specific provider,
    accepting a normalized message history and returning the final text reply.
    Tool specs are passed in Anthropic's canonical format; each provider
    converts them internally.

    The optional *on_usage* keyword argument accepts a callable that will be
    invoked once per underlying API call with a ``UsageSummary`` containing
    the token counts reported by the provider.  ``AIActor`` uses this to
    accumulate usage across all API calls within a single ``instruct()`` and
    forward the total to the attached ``Ledger``.

    The optional *monitoring_context* keyword argument is forwarded to
    ``LiteLLM`` providers for traffic monitoring.  Other providers silently
    ignore it.
    """

    model: str

    @abstractmethod
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
        """Run the full agentic loop and return the final text response.

        Args:
            system: The system prompt.
            messages: Conversation so far as ``[{"role": ..., "content": ...}]``.
                      Provider works on a copy; the original is not mutated.
            tools: Tool specs in Anthropic's canonical format (``input_schema``
                   key).  Providers convert to their own format internally.
            dispatcher: Callable that executes a named tool with the given args.
            max_tokens: Maximum tokens for the completion.
            on_usage: Optional callback invoked after each underlying API call
                      with the token counts for that call.
            monitoring_context: Optional metadata forwarded to LiteLLM for
                      monitoring.  Ignored by all other providers.
        """
