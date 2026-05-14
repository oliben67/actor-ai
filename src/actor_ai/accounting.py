"""Token-level accounting for AIActor usage and cost calculation."""

# Future imports (must occur at the beginning of the file):
from __future__ import annotations

# Standard library imports:
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Self

# ---------------------------------------------------------------------------
# Monitoring context
# ---------------------------------------------------------------------------


@dataclass
class MonitoringContext:
    """Metadata forwarded to LiteLLM (and ignored by other providers) for monitoring."""

    actor_name: str
    session_id: str
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Usage primitives
# ---------------------------------------------------------------------------


@dataclass
class UsageSummary:
    """Aggregated token counts for one or more LLM calls."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: UsageSummary) -> UsageSummary:
        return UsageSummary(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
        )

    def __iadd__(self, other: UsageSummary) -> Self:
        return self.__add__(other) # type: ignore


# ---------------------------------------------------------------------------
# Ledger entries
# ---------------------------------------------------------------------------


@dataclass
class LedgerEntry:
    """One record per ``instruct()`` call, storing token counts and metadata."""

    actor_name: str
    model: str
    input_tokens: int
    output_tokens: int
    timestamp: datetime
    session_id: str | None = None

    @property
    def usage(self) -> UsageSummary:
        return UsageSummary(self.input_tokens, self.output_tokens)


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


class Ledger:
    """Thread-safe accumulator of LLM token usage across actors and sessions.

    A single ``Ledger`` instance can be shared across multiple actors so that
    you get a unified view of all spending::

        from actor_ai import AIActor, Ledger, Rates

        ledger = Ledger()

        class AgentA(AIActor):
            ledger = ledger

        class AgentB(AIActor):
            ledger = ledger

        # After some calls…
        print(ledger.total_usage())
        print(ledger.total_cost(Rates.default()))
        print(ledger.cost_by_actor(Rates.default()))
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: list[LedgerEntry] = []

    # ------------------------------------------------------------------ #
    # Write                                                                #
    # ------------------------------------------------------------------ #

    def record(
        self,
        actor_name: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        session_id: str | None = None,
    ) -> None:
        """Append one entry. Called automatically by ``AIActor``."""
        entry = LedgerEntry(
            actor_name=actor_name,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            timestamp=datetime.now(UTC),
            session_id=session_id,
        )
        with self._lock:
            self._entries.append(entry)

    def clear(self) -> None:
        """Remove all recorded entries."""
        with self._lock:
            self._entries.clear()

    # ------------------------------------------------------------------ #
    # Read — entries                                                        #
    # ------------------------------------------------------------------ #

    def entries(self) -> list[LedgerEntry]:
        """Return a snapshot of all recorded entries."""
        with self._lock:
            return list(self._entries)

    def entries_for_actor(self, actor_name: str) -> list[LedgerEntry]:
        return [e for e in self.entries() if e.actor_name == actor_name]

    def entries_for_session(self, session_id: str) -> list[LedgerEntry]:
        return [e for e in self.entries() if e.session_id == session_id]

    def entries_for_model(self, model: str) -> list[LedgerEntry]:
        return [e for e in self.entries() if e.model == model]

    # ------------------------------------------------------------------ #
    # Read — usage aggregates                                              #
    # ------------------------------------------------------------------ #

    def total_usage(self) -> UsageSummary:
        """Total token consumption across all entries."""
        return _sum_usage(self.entries())

    def usage_by_actor(self) -> dict[str, UsageSummary]:
        """Token consumption grouped by actor name."""
        return _group_usage(self.entries(), key=lambda e: e.actor_name)

    def usage_by_model(self) -> dict[str, UsageSummary]:
        """Token consumption grouped by model name."""
        return _group_usage(self.entries(), key=lambda e: e.model)

    def usage_by_session(self) -> dict[str, UsageSummary]:
        """Token consumption grouped by session ID."""
        return _group_usage(self.entries(), key=lambda e: e.session_id or "no_session")

    # ------------------------------------------------------------------ #
    # Read — cost aggregates                                               #
    # ------------------------------------------------------------------ #

    def total_cost(self, rates: Rates) -> float:
        """Total spend across all entries."""
        return sum(rates.cost(e.model, e.usage) for e in self.entries())

    def cost_by_actor(self, rates: Rates) -> dict[str, float]:
        """Spend grouped by actor name."""
        return _group_cost(self.entries(), rates, key=lambda e: e.actor_name)

    def cost_by_model(self, rates: Rates) -> dict[str, float]:
        """Spend grouped by model name."""
        return _group_cost(self.entries(), rates, key=lambda e: e.model)

    def cost_by_session(self, rates: Rates) -> dict[str, float]:
        """Spend grouped by session ID."""
        return _group_cost(self.entries(), rates, key=lambda e: e.session_id or "no_session")

    # ------------------------------------------------------------------ #
    # Convenience                                                          #
    # ------------------------------------------------------------------ #

    def summary(self, rates: Rates | None = None) -> dict:
        """Return a human-readable summary dict, optionally with cost figures."""
        usage = self.total_usage()
        out: dict = {
            "entries": len(self.entries()),
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
            "by_actor": {k: v.total_tokens for k, v in self.usage_by_actor().items()},
            "by_model": {k: v.total_tokens for k, v in self.usage_by_model().items()},
        }
        if rates is not None:
            out["total_cost_usd"] = self.total_cost(rates)
            out["cost_by_actor_usd"] = self.cost_by_actor(rates)
            out["cost_by_model_usd"] = self.cost_by_model(rates)
        return out

    def __len__(self) -> int:
        return len(self.entries())

    def __repr__(self) -> str:
        u = self.total_usage()
        return f"Ledger(entries={len(self)}, input={u.input_tokens}, output={u.output_tokens})"


# ---------------------------------------------------------------------------
# Rates
# ---------------------------------------------------------------------------

# Approximate pricing (USD per million tokens) as of mid-2025.
# Verify current rates at each provider's pricing page before use in production.
_DEFAULT_RATES: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-7": (15.00, 75.00),
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gemini-2.0-flash": (0.075, 0.30),
    "gemini-1.5-pro": (1.25, 5.00),
    "mistral-large-latest": (2.00, 6.00),
    "mistral-small-latest": (0.10, 0.30),
    "deepseek-chat": (0.14, 0.28),
    "deepseek-reasoner": (0.55, 2.19),
}


@dataclass(frozen=True)
class ModelRate:
    """Pricing for one model in USD per million tokens."""

    input_per_million: float
    output_per_million: float

    def cost(self, usage: UsageSummary) -> float:
        return (
            usage.input_tokens * self.input_per_million / 1_000_000
            + usage.output_tokens * self.output_per_million / 1_000_000
        )


class Rates:
    """Configurable per-model pricing table.

    Usage::

        # Use built-in approximate rates:
        rates = Rates.default()

        # Override or extend with custom rates:
        rates = Rates.from_dict({
            "my-fine-tune": {"input": 1.00, "output": 2.00},
        })

        # Override a single model at runtime:
        rates.set("gpt-4o", input_per_million=2.50, output_per_million=10.00)

        # Calculate cost for a usage object:
        cost = rates.cost("claude-sonnet-4-6", usage)
    """

    def __init__(self, rates: dict[str, ModelRate] | None = None) -> None:
        self._rates: dict[str, ModelRate] = (
            dict(rates)
            if rates is not None
            else {m: ModelRate(i, o) for m, (i, o) in _DEFAULT_RATES.items()}
        )

    @classmethod
    def default(cls) -> Rates:
        """Build a ``Rates`` instance pre-populated with known model prices."""
        return cls()

    @classmethod
    def from_dict(cls, data: dict[str, dict[str, float]]) -> Rates:
        """Build from a plain dict: ``{model: {"input": X, "output": Y}}``."""
        return cls({m: ModelRate(r["input"], r["output"]) for m, r in data.items()})

    def set(self, model: str, *, input_per_million: float, output_per_million: float) -> None:
        """Add or replace the rate for *model*."""
        self._rates[model] = ModelRate(input_per_million, output_per_million)

    def cost(self, model: str, usage: UsageSummary) -> float:
        """Return USD cost for *usage* on *model*; returns 0.0 if model unknown."""
        rate = self._rates.get(model)
        return rate.cost(usage) if rate is not None else 0.0

    def get_rate(self, model: str) -> ModelRate | None:
        return self._rates.get(model)

    def models(self) -> list[str]:
        """Return all models with configured rates."""
        return list(self._rates.keys())

    def __contains__(self, model: str) -> bool:
        return model in self._rates

    def __repr__(self) -> str:
        return f"Rates(models={self.models()})"


# ---------------------------------------------------------------------------
# Session ID helper
# ---------------------------------------------------------------------------


def new_session_id() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sum_usage(entries: list[LedgerEntry]) -> UsageSummary:
    total = UsageSummary()
    for e in entries:
        total += e.usage
    return total


def _group_usage(entries: list[LedgerEntry], *, key) -> dict[str, UsageSummary]:
    result: dict[str, UsageSummary] = {}
    for e in entries:
        k = key(e)
        result[k] = result.get(k, UsageSummary()) + e.usage
    return result


def _group_cost(entries: list[LedgerEntry], rates: Rates, *, key) -> dict[str, float]:
    result: dict[str, float] = {}
    for e in entries:
        k = key(e)
        result[k] = result.get(k, 0.0) + rates.cost(e.model, e.usage)
    return result
