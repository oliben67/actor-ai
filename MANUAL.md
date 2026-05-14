# actor-ai — User Manual

## Table of Contents

1. [Core concepts](#1-core-concepts)
2. [AIActor](#2-aiactor)
3. [Session management](#3-session-management)
4. [Memory](#4-memory)
5. [Tool calling](#5-tool-calling)
6. [Providers](#6-providers)
7. [Chorus](#7-chorus)
8. [Accounting](#8-accounting)
9. [Monitoring with LiteLLM](#9-monitoring-with-litellm)
10. [Message API](#10-message-api)
11. [Complete reference](#11-complete-reference)

---

## 1. Core concepts

`actor-ai` is built on [pykka](https://github.com/jodal/pykka), a Python actor framework. Every AI agent is a `ThreadingActor` — it runs in its own thread, processes messages from a queue, and exposes a proxy API for thread-safe interaction.

```
your code
   │
   │  ref.proxy().instruct("…").get()
   ▼
AIActor (thread)
   │
   │  provider.run(system, messages, tools, …)
   ▼
LLM API (Claude / GPT / Gemini / …)
```

**pykka basics you need to know:**

| Pattern | Meaning |
|---|---|
| `ref = MyActor.start()` | Start the actor; returns an `ActorRef` |
| `ref.proxy().method(args).get()` | Call a method and wait for the result |
| `ref.proxy().attribute.get()` | Read a class/instance attribute |
| `ref.proxy().attribute = value` | Set an attribute on the running actor |
| `ref.ask(Message())` | Send a message object and block for the reply |
| `ref.tell(Message())` | Fire-and-forget message send |
| `ref.stop()` | Stop the actor |

---

## 2. AIActor

`AIActor` is the base class for all AI agents.

```python
from actor_ai import AIActor, Claude

class MyAgent(AIActor):
    system_prompt = "You are a helpful assistant."
    provider      = Claude()
    max_tokens    = 4096
    max_history   = 0       # 0 = unlimited session history
    ledger        = None    # attach a Ledger to enable accounting
    actor_name    = None    # human label used in accounting reports
    monitoring    = False   # set True to enable LiteLLM monitoring
```

### Class attributes

| Attribute | Type | Default | Description |
|---|---|---|---|
| `system_prompt` | `str` | `"You are a helpful AI agent."` | System prompt sent on every call |
| `provider` | `LLMProvider` | `Claude()` | Which LLM backend to use |
| `max_tokens` | `int` | `4096` | Maximum completion tokens |
| `max_history` | `int` | `0` | Keep last N turns; 0 = unlimited |
| `ledger` | `Ledger \| None` | `None` | Attach for token accounting |
| `actor_name` | `str \| None` | `None` | Label in accounting (default: class name) |
| `monitoring` | `bool` | `False` | Forward metadata to LiteLLM |

### Starting and stopping

```python
ref = MyAgent.start()           # starts the actor thread
ref.proxy().instruct("hi").get()
ref.stop()                      # joins the thread
```

Use a `try/finally` block to ensure cleanup:

```python
ref = MyAgent.start()
try:
    reply = ref.proxy().instruct("Hello!").get()
finally:
    ref.stop()
```

### instruct()

```python
reply: str = ref.proxy().instruct(
    instruction,            # str: the user message
    history=None,           # list[dict] | None: explicit message list
    use_session=True,       # bool: accumulate in rolling session
).get()
```

- When `use_session=True` and `history=None` (default), the actor appends the turn to its session.
- When `use_session=False`, the call is stateless — the session is not read or written.
- When `history` is provided, it is used as-is (the session is ignored).

---

## 3. Session management

The actor maintains a rolling list of `{"role": "user"/"assistant", "content": "…"}` messages. This list is prepended to every `instruct()` call so the LLM has full conversational context.

```python
class ChatBot(AIActor):
    system_prompt = "You are a friendly assistant."
    provider      = Claude()
    max_history   = 10      # keep last 10 turns (20 messages)

ref = ChatBot.start()
proxy = ref.proxy()

proxy.instruct("My name is Alice.").get()
proxy.instruct("Do you remember my name?").get()   # → "Yes, you told me your name is Alice."

session = proxy.get_session().get()   # list[dict]
print(len(session))                   # 4 (2 turns × 2 messages each)
```

### max_history

When `max_history > 0`, the actor keeps only the last `max_history` turns (2 × max_history messages). Older turns are dropped automatically after each call.

```python
class TightMemory(AIActor):
    max_history = 3   # 3-turn sliding window
```

### clear_session()

Discards all session messages and generates a new session ID. The actor's long-term memory is unaffected.

```python
old_id = proxy.get_session_id().get()
proxy.clear_session().get()
new_id = proxy.get_session_id().get()
assert old_id != new_id
```

### get_session() / get_session_id()

```python
messages:   list[dict] = proxy.get_session().get()
session_id: str        = proxy.get_session_id().get()
```

---

## 4. Memory

Memory stores named facts that persist across sessions. Facts are automatically appended to the system prompt so the LLM sees them on every call.

```python
proxy.remember("name",    "Alice").get()
proxy.remember("company", "Acme Corp").get()

# System prompt seen by the LLM becomes:
# "You are a helpful assistant.
#
# Known facts:
# - name: Alice
# - company: Acme Corp"

proxy.forget("company").get()

memory: dict[str, str] = proxy.get_memory().get()
```

**Memory vs session:**

| | Session | Memory |
|---|---|---|
| Scope | Single conversation thread | Persistent across sessions |
| Content | Full message history | Named key-value facts |
| Cleared by | `clear_session()` | `forget(key)` |
| Sent to LLM | As message history | Injected into system prompt |

---

## 5. Tool calling

Methods decorated with `@tool` are exposed to the LLM as callable functions. The actor dispatches calls automatically and feeds results back to the LLM.

```python
from actor_ai import AIActor, tool, Claude

class Calculator(AIActor):
    system_prompt = "Use tools for every computation. Never guess."
    provider      = Claude()

    @tool("Add two integers.")
    def add(self, x: int, y: int) -> int:
        return x + y

    @tool
    def power(self, base: float, exponent: float) -> float:
        """Raise base to the power of exponent."""
        return base ** exponent
```

### `@tool` decorator

```python
# Form 1: decorator with docstring description
@tool
def my_method(self, x: int) -> str:
    """Description seen by the LLM."""
    ...

# Form 2: explicit description string (overrides docstring)
@tool("Explicit description.")
def my_method(self, x: int) -> str:
    ...
```

### Type annotations and JSON Schema

Parameter types are converted to JSON Schema types automatically:

| Python | JSON Schema |
|---|---|
| `str` | `"string"` |
| `int` | `"integer"` |
| `float` | `"number"` |
| `bool` | `"boolean"` |
| `list` | `"array"` |
| `dict` | `"object"` |
| anything else | `"string"` |

Parameters with default values are marked as **optional** in the schema:

```python
@tool
def greet(self, name: str, formal: bool = False) -> str:
    """Greet a person, optionally in formal style."""
    ...
# name is required; formal is optional
```

### How tool calling works

1. The LLM receives the tool specs alongside the conversation.
2. If the LLM wants to call a tool, it emits a tool-use response.
3. `AIActor` dispatches the call to the decorated method.
4. The return value (converted to `str`) is fed back to the LLM.
5. The loop repeats until the LLM returns a final text reply.

This loop runs entirely inside `provider.run()` — from the caller's perspective, `instruct()` is still a single call that returns the final reply.

---

## 6. Providers

### Claude (Anthropic)

```python
from actor_ai import Claude

class MyAgent(AIActor):
    provider = Claude()                              # defaults
    provider = Claude("claude-opus-4-7",
                      temperature=0.2,
                      top_p=0.9,
                      top_k=40,
                      timeout=30.0,
                      stop_sequences=["STOP"])
```

| Parameter | Type | Description |
|---|---|---|
| `model` | `str` | Default: `"claude-sonnet-4-6"` |
| `api_key` | `str \| None` | Overrides `ANTHROPIC_API_KEY` |
| `temperature` | `float \| None` | Sampling temperature `[0, 1]` |
| `top_p` | `float \| None` | Nucleus sampling. Mutually exclusive with `top_k` |
| `top_k` | `int \| None` | Top-K sampling |
| `timeout` | `float \| None` | HTTP timeout in seconds |
| `stop_sequences` | `list[str] \| None` | Additional stop sequences |

### GPT (OpenAI)

```python
from actor_ai import GPT

provider = GPT()                                    # gpt-4o
provider = GPT("gpt-4o-mini",
               temperature=0.0,
               seed=42,
               frequency_penalty=0.3,
               response_format={"type": "json_object"})
```

| Parameter | Type | Description |
|---|---|---|
| `model` | `str` | Default: `"gpt-4o"` |
| `api_key` | `str \| None` | Overrides `OPENAI_API_KEY` |
| `temperature` | `float \| None` | `[0, 2]` |
| `top_p` | `float \| None` | |
| `frequency_penalty` | `float \| None` | |
| `presence_penalty` | `float \| None` | |
| `timeout` | `float \| None` | |
| `stop` | `list[str] \| str \| None` | |
| `seed` | `int \| None` | Deterministic sampling |
| `response_format` | `dict \| None` | e.g. `{"type": "json_object"}` |

### Gemini (Google AI)

```python
from actor_ai import Gemini

provider = Gemini()                                 # gemini-2.0-flash
provider = Gemini("gemini-1.5-pro", temperature=0.4, top_p=0.95)
```

Reads `GOOGLE_API_KEY`. Uses Google's OpenAI-compatible endpoint.

### Mistral

```python
from actor_ai import Mistral

provider = Mistral()                                # mistral-large-latest
provider = Mistral("mistral-small-latest", temperature=0.7)
```

Reads `MISTRAL_API_KEY`.

### DeepSeek

```python
from actor_ai import DeepSeek

provider = DeepSeek()                               # deepseek-chat
provider = DeepSeek("deepseek-reasoner", temperature=0.0, seed=0)
```

Reads `DEEPSEEK_API_KEY`.

### Copilot (GitHub)

Routes requests through GitHub Copilot's OpenAI-compatible endpoint (`https://api.githubcopilot.com`). Requires a GitHub account with an active Copilot subscription (Individual, Business, or Enterprise).

Token resolution order (first match wins):

1. `api_key` constructor argument
2. `GITHUB_TOKEN` environment variable
3. `gh auth token` CLI — works automatically when the [GitHub CLI](https://cli.github.com/) is authenticated (`gh auth login`), no env var required

```python
from actor_ai import AIActor, Copilot

class Assistant(AIActor):
    system_prompt = "You are a helpful coding assistant."
    provider = Copilot()                        # gpt-4o (default)
    provider = Copilot("claude-sonnet-4-5")     # Claude via Copilot
    provider = Copilot("gemini-2.0-flash",
                       temperature=0.2,
                       seed=42)

# Inspect valid models at runtime
print(Copilot.MODELS)   # frozenset of valid model strings
```

#### Valid models

| Model | Notes |
|---|---|
| `gpt-4o` | Default |
| `gpt-4o-mini` | |
| `o1` | |
| `o1-mini` | |
| `o3-mini` | |
| `claude-sonnet-4-5` | Anthropic Claude via Copilot |
| `gemini-2.0-flash` | Google Gemini via Copilot |

Passing any other model string raises `ValueError` immediately at construction time — before any network call. Use `Copilot.MODELS` (a `frozenset[str]`) for runtime validation, or rely on the `CopilotModel` `Literal` type for IDE autocompletion.

```python
from actor_ai import CopilotModel   # Literal["gpt-4o", "gpt-4o-mini", ...]

try:
    Copilot("gpt-3.5-turbo")
except ValueError as exc:
    print(exc)   # Unsupported Copilot model 'gpt-3.5-turbo'. Valid models: ...
```

#### Parameters

| Parameter | Type | Description |
|---|---|---|
| `model` | `CopilotModel` | Default: `"gpt-4o"` |
| `api_key` | `str \| None` | GitHub token; overrides env var and `gh` CLI |
| `temperature` | `float \| None` | Sampling temperature |
| `top_p` | `float \| None` | Nucleus-sampling probability mass |
| `timeout` | `float \| None` | HTTP timeout in seconds |
| `stop` | `list[str] \| str \| None` | Stop sequences |
| `seed` | `int \| None` | Deterministic sampling (best-effort) |

The `Copilot-Integration-Id: vscode-chat` header is sent automatically so GitHub's backend routes the request correctly.

### LiteLLM

See [Section 9 — Monitoring with LiteLLM](#9-monitoring-with-litellm) for full details.

```python
from actor_ai import LiteLLM

provider = LiteLLM("openai/gpt-4o")
provider = LiteLLM("anthropic/claude-sonnet-4-6",
                   temperature=0.2,
                   max_retries=3,
                   success_callbacks=["langfuse"])
```

### Swapping providers at runtime

Because pykka's proxy supports attribute assignment, you can replace the provider on a running actor:

```python
ref.proxy().provider = GPT("gpt-4o-mini")
```

The next `instruct()` call will use the new provider.

---

## 7. Chorus

`Chorus` manages a named group of `AIActor` instances and provides three coordination patterns.

```python
from actor_ai import AIActor, Chorus, Claude, GPT

class Researcher(AIActor):
    system_prompt = "You are a research specialist."
    provider      = Claude()

class Writer(AIActor):
    system_prompt = "You are a creative writer."
    provider      = GPT("gpt-4o")

researcher_ref = Researcher.start()
writer_ref     = Writer.start()

chorus_ref = Chorus.start(
    agents={"researcher": researcher_ref, "writer": writer_ref}
)
chorus = chorus_ref.proxy()
```

### Coordination patterns

**instruct** — send to one agent:
```python
reply: str = chorus.instruct("researcher", "Find facts about Mars.").get()
```

**broadcast** — send to all agents in parallel:
```python
replies: dict[str, str] = chorus.broadcast("Introduce yourself.").get()
# {"researcher": "I am …", "writer": "I am …"}
```

**pipeline** — chain agents sequentially; each receives the previous reply:
```python
final: str = chorus.pipeline(
    ["researcher", "writer"],
    "Write a report about Mars.",
).get()
```

### Agent management

```python
# Add / remove at runtime
chorus.add("editor", editor_ref).get()
chorus.remove("editor").get()

# List agents
names: list[str] = chorus.agents().get()

# Stop specific agents (also removes from chorus)
chorus.stop_agents(names=["reviewer"]).get()

# Stop all agents
chorus.stop_agents().get()
```

### Memory broadcast

```python
# Store a fact in all agents
chorus.remember("audience", "general public").get()

# Store in specific agents only
chorus.remember("style", "academic", names=["researcher"]).get()

# Forget from all agents
chorus.forget("style").get()

# Forget from specific agents
chorus.forget("audience", names=["writer"]).get()
```

### Stopping

`chorus_ref.stop()` stops the Chorus actor itself. Registered sub-agents are **not** stopped automatically — call `chorus.stop_agents()` first or stop them individually.

---

## 8. Accounting

The accounting layer records every `instruct()` call and calculates token spend.

### Ledger

```python
from actor_ai import Ledger, Rates

ledger = Ledger()

class MyAgent(AIActor):
    provider   = Claude()
    ledger     = ledger          # attach the shared ledger
    actor_name = "my-agent"      # label in reports (default: class name)
```

Each completed `instruct()` call appends one `LedgerEntry`:
- `actor_name`, `model`, `input_tokens`, `output_tokens`, `timestamp`, `session_id`

#### Reading the ledger

```python
# All entries
entries: list[LedgerEntry] = ledger.entries()

# Filtered
entries_for_actor:   list[LedgerEntry] = ledger.entries_for_actor("my-agent")
entries_for_session: list[LedgerEntry] = ledger.entries_for_session(session_id)
entries_for_model:   list[LedgerEntry] = ledger.entries_for_model("claude-sonnet-4-6")

# Aggregate usage
total: UsageSummary                    = ledger.total_usage()
by_actor:  dict[str, UsageSummary]     = ledger.usage_by_actor()
by_model:  dict[str, UsageSummary]     = ledger.usage_by_model()
by_session: dict[str, UsageSummary]    = ledger.usage_by_session()

# Cost
rates = Rates.default()
total_cost:    float                   = ledger.total_cost(rates)
by_actor_cost: dict[str, float]        = ledger.cost_by_actor(rates)
by_model_cost: dict[str, float]        = ledger.cost_by_model(rates)

# Summary dict
summary: dict = ledger.summary(rates)

# Misc
n: int = len(ledger)
ledger.clear()
```

### Rates

```python
from actor_ai import Rates, ModelRate

# Pre-populated with ~2025 rates for all built-in models
rates = Rates.default()

# Build from a dict
rates = Rates.from_dict({
    "my-model": {"input": 1.50, "output": 3.00},
})

# Override a single model at runtime
rates.set("gpt-4o", input_per_million=2.50, output_per_million=10.00)

# Look up a rate
rate: ModelRate | None = rates.get_rate("gpt-4o")

# Calculate cost for a given usage
cost: float = rates.cost("gpt-4o", usage_summary)

# Check membership
"gpt-4o" in rates   # True
```

### UsageSummary

```python
from actor_ai import UsageSummary

u = UsageSummary(input_tokens=100, output_tokens=50)
u.total_tokens   # 150

# Supports addition
total = u1 + u2
total += u3
```

### LedgerEntry

```python
entry.actor_name     # str
entry.model          # str
entry.input_tokens   # int
entry.output_tokens  # int
entry.timestamp      # datetime (UTC)
entry.session_id     # str | None
entry.usage          # UsageSummary
```

### Default rates (approximate USD / million tokens, mid-2025)

| Model | Input | Output |
|---|---|---|
| `claude-sonnet-4-6` | $3.00 | $15.00 |
| `claude-opus-4-7` | $15.00 | $75.00 |
| `claude-haiku-4-5-20251001` | $0.80 | $4.00 |
| `gpt-4o` | $2.50 | $10.00 |
| `gpt-4o-mini` | $0.15 | $0.60 |
| `gemini-2.0-flash` | $0.075 | $0.30 |
| `gemini-1.5-pro` | $1.25 | $5.00 |
| `mistral-large-latest` | $2.00 | $6.00 |
| `mistral-small-latest` | $0.10 | $0.30 |
| `deepseek-chat` | $0.14 | $0.28 |
| `deepseek-reasoner` | $0.55 | $2.19 |

Always verify current rates at each provider's pricing page before use in production.

---

## 9. Monitoring with LiteLLM

When `monitoring = True`, the actor creates a `MonitoringContext` for every `instruct()` call and passes it to the provider. The `LiteLLM` provider forwards this context as the `metadata` kwarg to `litellm.completion()`, making it visible to any registered callbacks (Langfuse, Helicone, custom, etc.).

### Enabling monitoring

```python
from actor_ai import AIActor, LiteLLM

class MonitoredAgent(AIActor):
    system_prompt = "You are a helpful assistant."
    provider      = LiteLLM("openai/gpt-4o")
    actor_name    = "support-bot"
    monitoring    = True
```

When `monitoring = True`, every `instruct()` call forwards this to `litellm.completion()`:

```python
metadata = {
    "actor_name": "support-bot",       # actor_name or class name
    "session_id": "<uuid>",            # current session ID
    # + any extra fields in MonitoringContext.metadata
}
```

### MonitoringContext

```python
from actor_ai import MonitoringContext

ctx = MonitoringContext(
    actor_name = "my-agent",
    session_id = "sess-abc123",
    metadata   = {"user_id": "u42", "environment": "prod"},
)
```

`metadata` is merged into the dict forwarded to LiteLLM, so all three fields — `actor_name`, `session_id`, and any custom keys — arrive in the same flat dict.

### Registering callbacks

Pass callback names (for built-in LiteLLM integrations) or callables directly to the `LiteLLM` constructor:

```python
def my_callback(kwargs, response, start_time, end_time):
    meta = kwargs.get("metadata", {})
    print(f"[{meta.get('actor_name')}] {response.usage}")

provider = LiteLLM(
    "openai/gpt-4o",
    success_callbacks=[my_callback, "langfuse"],
    failure_callbacks=["sentry"],
)
```

### LiteLLM provider parameters

| Parameter | Type | Description |
|---|---|---|
| `model` | `str` | LiteLLM model string, e.g. `"openai/gpt-4o"` |
| `api_key` | `str \| None` | Explicit API key |
| `temperature` | `float \| None` | |
| `top_p` | `float \| None` | |
| `timeout` | `float \| None` | |
| `max_retries` | `int \| None` | Passed as `num_retries` to LiteLLM |
| `success_callbacks` | `list \| None` | LiteLLM success callbacks |
| `failure_callbacks` | `list \| None` | LiteLLM failure callbacks |

### Monitoring with other providers

The `monitoring_context` parameter is passed to every provider's `run()` method. Non-LiteLLM providers (`Claude`, `GPT`, etc.) silently ignore it — so you can safely set `monitoring = True` on an actor using any provider and switch later.

---

## 10. Message API

As a pykka actor, `AIActor` accepts raw message objects in addition to the proxy API.

### Message types

```python
from actor_ai import Instruct, Remember, Forget

Instruct(
    instruction: str,
    history:     list[dict] = [],    # explicit message list
    use_session: bool       = True,
)

Remember(key: str, value: str)

Forget(key: str)
```

### ask() — synchronous

```python
reply = ref.ask(Instruct("What is 2 + 2?"))              # → "4"
ref.ask(Remember("name", "Alice"))                        # → None
ref.ask(Forget("name"))                                   # → None
```

### tell() — fire-and-forget

```python
ref.tell(Remember("name", "Alice"))    # does not block
ref.tell(Forget("name"))               # does not block
```

pykka processes messages in FIFO order, so a `tell(Remember(…))` followed by `ask(Instruct(…))` is guaranteed to process the `Remember` first.

### Proxy API equivalents

| Message API | Proxy API |
|---|---|
| `ref.ask(Instruct("…"))` | `ref.proxy().instruct("…").get()` |
| `ref.ask(Remember("k","v"))` | `ref.proxy().remember("k","v").get()` |
| `ref.ask(Forget("k"))` | `ref.proxy().forget("k").get()` |
| `ref.tell(Remember("k","v"))` | `ref.proxy().remember("k","v")` *(no .get())* |
| `ref.tell(Forget("k"))` | `ref.proxy().forget("k")` *(no .get())* |

---

## 11. Complete reference

### AIActor public API

```python
# Instructions
instruct(instruction, history=None, use_session=True) -> str

# Memory
remember(key: str, value: str) -> None
forget(key: str) -> None
get_memory() -> dict[str, str]

# Session
clear_session() -> None
get_session() -> list[dict]
get_session_id() -> str
```

### Chorus public API

```python
# Agent management
add(name: str, ref: ActorRef) -> None
remove(name: str) -> None
agents() -> list[str]
stop_agents(names: list[str] | None = None) -> None

# Coordination
instruct(name: str, instruction: str, **kwargs) -> str
broadcast(instruction: str, **kwargs) -> dict[str, str]
pipeline(names: list[str], instruction: str, **kwargs) -> str

# Memory broadcast
remember(key: str, value: str, names: list[str] | None = None) -> None
forget(key: str, names: list[str] | None = None) -> None
```

### Ledger public API

```python
# Write
record(actor_name, model, input_tokens, output_tokens, session_id=None) -> None
clear() -> None

# Read — entries
entries() -> list[LedgerEntry]
entries_for_actor(actor_name) -> list[LedgerEntry]
entries_for_session(session_id) -> list[LedgerEntry]
entries_for_model(model) -> list[LedgerEntry]

# Read — usage
total_usage() -> UsageSummary
usage_by_actor() -> dict[str, UsageSummary]
usage_by_model() -> dict[str, UsageSummary]
usage_by_session() -> dict[str, UsageSummary]

# Read — cost
total_cost(rates: Rates) -> float
cost_by_actor(rates: Rates) -> dict[str, float]
cost_by_model(rates: Rates) -> dict[str, float]
cost_by_session(rates: Rates) -> dict[str, float]

# Summary
summary(rates: Rates | None = None) -> dict

# Misc
__len__() -> int
```

### Rates public API

```python
default() -> Rates                           # classmethod
from_dict(data: dict) -> Rates               # classmethod
set(model, *, input_per_million, output_per_million) -> None
cost(model: str, usage: UsageSummary) -> float
get_rate(model: str) -> ModelRate | None
models() -> list[str]
__contains__(model: str) -> bool
```

### @tool decorator

```python
@tool                           # description from docstring
@tool("Explicit description")   # explicit description
```

The decorated method must be a regular instance method on an `AIActor` subclass. It will not appear in `extract_tools()` results for plain Python objects; only methods on running actors are dispatched.

### Exported symbols

```python
from actor_ai import (
    # Actors
    AIActor, Chorus,
    # Providers
    Claude, Copilot, CopilotModel, GPT, Gemini, Mistral, DeepSeek, LiteLLM, LLMProvider,
    # Messages
    Instruct, Remember, Forget,
    # Accounting
    Ledger, LedgerEntry, ModelRate, MonitoringContext, Rates, UsageSummary,
    # Decorators
    tool,
)
```
