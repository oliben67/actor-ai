# actor-ai — User Manual

## Table of Contents

1. [Core concepts](#1-core-concepts)
2. [AIActor](#2-aiactor)
3. [Session management](#3-session-management)
4. [Memory](#4-memory)
5. [Tool calling](#5-tool-calling)
6. [Providers](#6-providers)
7. [Chorus](#7-chorus)
8. [Workflow](#8-workflow)
9. [Accounting](#9-accounting)
10. [Monitoring with LiteLLM](#10-monitoring-with-litellm)
11. [Message API](#11-message-api)
12. [Complete reference](#12-complete-reference)

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
| `provider` | `LLMProvider \| None` | `None` | Which LLM backend to use; `None` means no provider |
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
- Raises `RuntimeError` if `provider` is `None`.

### AIActor without a provider

`provider` defaults to `None`. An actor without a provider behaves like a plain pykka `ThreadingActor` — it can receive messages, manage memory, track session, and communicate with other actors. Calling `instruct()` raises `RuntimeError`.

```python
from actor_ai import AIActor

class DataActor(AIActor):
    pass  # no provider — pure actor, no LLM needed

ref = DataActor.start()

# Memory and session still work:
ref.proxy().remember("context", "quarterly report").get()
ref.proxy().get_memory().get()   # → {"context": "quarterly report"}

# instruct() raises RuntimeError without a provider:
try:
    ref.proxy().instruct("summarise").get()
except RuntimeError as exc:
    print(exc)   # "No provider configured. Set a provider class attribute: …"

ref.stop()
```

This is useful for pure coordination actors, data-processing nodes, or actors that act as sinks in a `Chorus` pipeline without invoking an LLM.

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

See [Section 10 — Monitoring with LiteLLM](#10-monitoring-with-litellm) for full details.

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

`Chorus` manages a named group of actors and provides three coordination patterns. Members can be `AIActor` instances, other `Chorus` instances, or any plain pykka actor that exposes an `instruct()` method.

```python
from actor_ai import AIActor, Chorus, ChorusType, Claude, GPT

class Researcher(AIActor):
    system_prompt = "You are a research specialist."
    provider      = Claude()

class Writer(AIActor):
    system_prompt = "You are a creative writer."
    provider      = GPT("gpt-4o")

researcher_ref = Researcher.start()
writer_ref     = Writer.start()

chorus_ref = Chorus.start(
    agents={"researcher": researcher_ref, "writer": writer_ref},
    type="team",
)
chorus = chorus_ref.proxy()
```

### ChorusType

`ChorusType` is a `Literal` type that describes the role of a Chorus:

```python
from actor_ai import ChorusType

# Valid values
"system"      # system-level coordination layer
"project"     # project-scoped group
"team"        # functional team
"department"  # department-level grouping
"custom"      # default — any other purpose
```

Read the type via the proxy:

```python
t: ChorusType = chorus.type.get()   # → "team"
```

### Coordination patterns

**instruct** — send to one agent:
```python
reply: str = chorus.instruct("researcher", "Find facts about Mars.").get()
```

**instruct (broadcast form)** — when called with a single argument, broadcasts to all members and returns a formatted `"name: reply\n…"` string:
```python
combined: str = chorus.instruct("Introduce yourself.").get()
# → "researcher: I am a research specialist …\nwriter: I am a creative writer …"
```

**broadcast** — send to all agents in parallel, returns a dict:
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
# Add / remove at construction or runtime
chorus.add("editor", editor_ref).get()
chorus.remove("editor").get()

# Semantic aliases for add / remove
chorus.join("reviewer", reviewer_ref).get()   # same as add()
chorus.leave("reviewer").get()                # same as remove()

# List agents
names: list[str] = chorus.agents().get()

# Stop specific agents (also removes from chorus)
chorus.stop_agents(names=["reviewer"]).get()

# Stop all agents
chorus.stop_agents().get()
```

### Memory broadcast

`remember()` / `forget()` calls propagate to **all members** automatically, including nested choruses and non-AI actors (if they expose the matching method):

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

Memory propagation also works via the message API: `ref.tell(Remember("key", "val"))` delivered to a Chorus is forwarded to all members.

### Nested choruses

A `Chorus` can be a member of another `Chorus`. `broadcast()` and memory propagation cascade correctly through nesting:

```python
inner = Chorus.start(agents={"a": a_ref, "b": b_ref}, type="team")
outer = Chorus.start(agents={"inner": inner, "c": c_ref}, type="department")

# Broadcasts to inner (which broadcasts to a and b) and to c
outer.proxy().broadcast("Announce yourselves.").get()

# Memory flows to a, b, and c
outer.proxy().remember("project", "Atlas").get()
```

### Non-AI actors as members

Any pykka actor that exposes an `instruct(instruction)` method can be a Chorus member. Plain actors do not need to extend `AIActor`:

```python
import pykka

class LoggingActor(pykka.ThreadingActor):
    def instruct(self, instruction: str) -> str:
        print(f"[LOG] {instruction}")
        return "logged"

logger_ref = LoggingActor.start()
chorus_ref = Chorus.start(agents={"logger": logger_ref, "writer": writer_ref})
```

### Stopping

`chorus_ref.stop()` stops the Chorus actor itself. Registered sub-agents are **not** stopped automatically — call `chorus.stop_agents()` first or stop them individually.

---

## 8. Workflow

`Workflow` implements a state machine that orchestrates actors and Choruses. Transitions fire either when a reply matches a guard predicate or when a named event is dispatched. States and transitions can be added, replaced, or removed at runtime on a live workflow.

### Building blocks

**WorkflowState** maps a state name to an actor (or group of parallel actors) plus an instruction template:

```python
from actor_ai import WorkflowState

# Single actor or Chorus
WorkflowState(chorus=review_chorus, instruction="Review this:\n{output}")

# Parallel actors — all fired simultaneously, replies combined as "name: reply\n…"
WorkflowState(
    chorus={"researcher": researcher_ref, "critic": critic_ref},
    instruction="Analyse: {input}",
)
```

**Instruction templates** support two placeholders:

| Placeholder | Expands to |
|---|---|
| `{input}` | The original instruction passed to `run()` / `step()` |
| `{output}` | The reply produced by the immediately preceding state |

Literal braces that are not template markers must be doubled (`{{` / `}}`).

**WorkflowTransition** is a directed edge:

```python
from actor_ai import WorkflowTransition

# Guard — fires automatically when the reply matches a predicate
WorkflowTransition("draft", "review", guard=lambda r: "ready" in r)

# Event — fires when Workflow.event(name) is called
WorkflowTransition("review", "draft", on_event="reject")

# Both — first matching condition wins
WorkflowTransition("analyse", "approve",
                   guard=lambda r: "approved" in r,
                   on_event="force_approve")
```

### Creating a workflow

```python
from actor_ai import Workflow, WorkflowState, WorkflowTransition

wf = Workflow.start(
    states={
        "draft":  WorkflowState(draft_chorus,  instruction="{input}"),
        "review": WorkflowState(review_chorus, instruction="Review:\n{output}"),
    },
    transitions=[
        WorkflowTransition("draft", "review", guard=lambda r: "ready" in r),
        WorkflowTransition("review", "draft", on_event="reject"),
    ],
    initial_state="draft",
)
```

### Blocking execution — run() and step()

**`run(instruction)`** — executes from the current state and follows guard transitions until no guard matches (terminal state). Event-only transitions do not block `run()`. Blocks the calling thread until complete.

```python
output: str = wf.proxy().run("Draft a proposal.").get()
```

**`step(instruction)`** — executes the current state exactly once and applies the first matching guard transition. Does not loop.

```python
out1 = wf.proxy().step("Begin the process.").get()
# current state may have advanced via guard
out2 = wf.proxy().step().get()   # instruction=None → uses last output as {input}
```

### Event transitions

Fire a named event to trigger an event-based transition from the current state:

```python
fired: bool = wf.proxy().event("reject").get()
# True if a matching transition was found; False otherwise
```

### Non-blocking execution — run_detached()

`run_detached()` launches the workflow loop in a background OS thread so the workflow actor's mailbox stays free for `event()` and management calls during execution.

```python
import threading

done = threading.Event()

wf.proxy().run_detached(
    "Draft a proposal.",
    on_complete=lambda output: done.set(),
    on_error=lambda exc: print("Error:", exc),
).get()

# Actor mailbox is free — fire events while the run is in progress
wf.proxy().event("approve").get()

done.wait(timeout=30)
```

The three-phase protocol used internally:

1. **Prepare** (`prepare_step`) — actor atomically reads the current state and formats the instruction.
2. **Execute** — the actor(s) are invoked outside the workflow thread (slow, actor is free).
3. **Commit** (`commit_step`) — actor atomically stores the output and advances the state.

You can use `prepare_step` / `commit_step` directly for custom orchestration (see [Section 8 — Complete reference](#12-complete-reference)).

### Runtime modification

States and transitions can be added or replaced on a running workflow:

```python
# Add a new state
wf.proxy().add_state(
    "approve",
    WorkflowState(approve_chorus, instruction="Finalise:\n{output}")
).get()

# Add a new transition
wf.proxy().add_transition(
    WorkflowTransition("review", "approve", on_event="approve")
).get()

# Remove a state
wf.proxy().remove_state("old_state").get()

# Remove transitions from a source
wf.proxy().remove_transitions("review").get()               # all from "review"
wf.proxy().remove_transitions("review", "draft").get()      # only review→draft
```

### State inspection and control

```python
# Force-jump to any registered state (raises KeyError if unknown)
wf.proxy().set_state("review").get()

# Read current state and last output
state: str | None = wf.proxy().current_state().get()
output: str       = wf.proxy().last_output().get()

# List all registered state names
names: list[str] = wf.proxy().states().get()
```

### Parallel actor states

Pass a `dict[str, ActorRef]` as `chorus` to fire multiple actors simultaneously. All actors receive the same instruction; their replies are combined as `"name: reply\n…"`:

```python
WorkflowState(
    chorus={"researcher": r_ref, "critic": c_ref},
    instruction="Analyse: {input}",
)
# Combined output: "researcher: …\ncritic: …"
```

Guard predicates receive the combined string, so you can match keywords from any actor:

```python
WorkflowTransition(
    "analyse", "summarise",
    guard=lambda r: "approved" in r   # matches if any actor replied "approved"
)
```

### Complete workflow example

```python
from actor_ai import AIActor, Chorus, Claude, Workflow, WorkflowState, WorkflowTransition

class Drafter(AIActor):
    system_prompt = "You draft documents."
    provider = Claude()

class Reviewer(AIActor):
    system_prompt = "You review documents for quality."
    provider = Claude()

class Approver(AIActor):
    system_prompt = "You give final sign-off."
    provider = Claude()

drafter_ref  = Drafter.start()
reviewer_ref = Reviewer.start()
approver_ref = Approver.start()

draft_chorus  = Chorus.start(agents={"drafter":  drafter_ref})
review_chorus = Chorus.start(agents={"reviewer": reviewer_ref})
approve_chorus= Chorus.start(agents={"approver": approver_ref})

wf = Workflow.start(
    states={
        "draft":   WorkflowState(draft_chorus,   instruction="{input}"),
        "review":  WorkflowState(review_chorus,  instruction="Review:\n{output}"),
        "approve": WorkflowState(approve_chorus, instruction="Finalise:\n{output}"),
    },
    transitions=[
        WorkflowTransition("draft",  "review",  guard=lambda r: "ready" in r),
        WorkflowTransition("review", "approve", guard=lambda r: "approved" in r),
        WorkflowTransition("review", "draft",   on_event="reject"),
    ],
    initial_state="draft",
)

# Blocking run — follows guard chain until terminal
output = wf.proxy().run("Please draft a Q3 proposal.").get()

wf.stop()
```

---

## 9. Accounting

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

## 10. Monitoring with LiteLLM

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

## 11. Message API

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

## 12. Complete reference

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
join(name: str, ref: ActorRef) -> None    # alias for add()
leave(name: str) -> None                  # alias for remove()
agents() -> list[str]
stop_agents(names: list[str] | None = None) -> None

# Coordination
instruct(instruction: str) -> str                          # broadcast form (single arg)
instruct(name: str, instruction: str, **kwargs) -> str     # single-agent form (two args)
broadcast(instruction: str, **kwargs) -> dict[str, str]
pipeline(names: list[str], instruction: str, **kwargs) -> str

# Memory broadcast
remember(key: str, value: str, names: list[str] | None = None) -> None
forget(key: str, names: list[str] | None = None) -> None

# Type
type: ChorusType   # attribute (readable via proxy)
```

### Workflow public API

```python
# State machine management
add_state(name: str, state: WorkflowState) -> None
remove_state(name: str) -> None
add_transition(transition: WorkflowTransition) -> None
remove_transitions(source: str, target: str | None = None) -> None
set_state(name: str) -> None              # raises KeyError if unknown

# Inspection
states() -> list[str]
current_state() -> str | None
last_output() -> str

# Blocking execution
step(instruction: str | None = None) -> str
run(instruction: str | None = None) -> str

# Non-blocking execution
run_detached(
    instruction: str | None = None,
    on_complete: Callable[[str], None] | None = None,
    on_error: Callable[[Exception], None] | None = None,
) -> None

# Event dispatch
event(name: str) -> bool

# Low-level coordination (used by run_detached; available for custom orchestration)
prepare_step(instruction: str | None) -> tuple[ActorRef | dict[str, ActorRef], str] | None
commit_step(output: str) -> bool
```

### WorkflowState

```python
@dataclass
class WorkflowState:
    chorus: pykka.ActorRef | dict[str, pykka.ActorRef]
    instruction: str = "{output}"
```

### WorkflowTransition

```python
@dataclass
class WorkflowTransition:
    source: str
    target: str
    on_event: str | None = None
    guard: Callable[[str], bool] | None = None
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
    AIActor, Chorus, ChorusType,
    # Workflow
    Workflow, WorkflowState, WorkflowTransition,
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
