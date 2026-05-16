# actor-ai — User Manual

## Table of Contents

1. [Core concepts](#1-core-concepts)
2. [Defining agents](#2-defining-agents)
3. [Session management](#3-session-management)
4. [Memory](#4-memory)
5. [Tool calling and sub-agents](#5-tool-calling-and-sub-agents)
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
   │  proxy.instruct("…").get()
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
| `ref = MyAgent.start()` | Start the actor; returns an `ActorRef` |
| `ref.proxy().method(args).get()` | Call a method and wait for the result |
| `ref.proxy().attribute.get()` | Read a class/instance attribute |
| `ref.proxy().attribute = value` | Set an attribute on the running actor |
| `ref.ask(Message())` | Send a message object and block for the reply |
| `ref.tell(Message())` | Fire-and-forget message send |
| `ref.stop()` | Stop the actor |

---

## 2. Defining agents

### 2.1 The recommended approach: `make_agent()`

`make_agent()` creates a fully configured agent class in a single call — no subclassing needed. It is the preferred way to define agents because it is concise, composable, and makes agent hierarchies explicit.

```python
from actor_ai import make_agent, Claude, GPT

Researcher = make_agent(
    "Researcher",
    "You are a deep research specialist. Cite sources.",
    Claude(),
)

Writer = make_agent(
    "Writer",
    "You write clear, concise summaries for a general audience.",
    GPT("gpt-4o"),
)
```

Use the returned class exactly like a hand-written `AIActor` subclass:

```python
# Manual start / stop
ref = Researcher.start()
reply = ref.proxy().instruct("Tell me about Mars.").get()
ref.stop()

# Synchronous context manager (recommended)
with Researcher.get_proxy() as proxy:
    reply = proxy.instruct("Tell me about Mars.").get()

# Async context manager
async with Researcher.aget_proxy() as proxy:
    reply = await asyncio.to_thread(proxy.instruct("Tell me about Mars.").get)
```

### `make_agent()` parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | *required* | Class name; also the default `actor_name` label in accounting |
| `system_prompt` | `str` | *required* | System prompt sent on every `instruct()` call |
| `provider` | `LLMProvider \| None` | `None` | LLM backend; `None` means no provider (pure actor) |
| `tools` | `list[Callable] \| None` | `None` | Functions to expose to the LLM (auto-decorated if not already) |
| `sub_agents` | `dict[str, type[AIActor]] \| None` | `None` | Agent classes auto-wired as `@tool` delegation methods |
| `max_tokens` | `int` | `4096` | Maximum completion tokens |
| `max_history` | `int` | `0` | Rolling session window in turns; `0` = unlimited |
| `ledger` | `Ledger \| None` | `None` | Attach for token accounting |
| `actor_name` | `str \| None` | `None` | Explicit accounting label; defaults to `name` |
| `monitoring` | `bool` | `False` | Forward metadata to LiteLLM |

### 2.2 Agent hierarchies with `sub_agents`

The most powerful feature of `make_agent()` is `sub_agents`. Each entry is automatically wired as a `@tool` method: when the LLM calls it, a fresh instance of the named agent class is started, instructed, and stopped — with no routing code needed.

```python
from actor_ai import make_agent, Claude

Researcher = make_agent(
    "Researcher",
    "You are a research specialist. Return concise, sourced facts.",
    Claude(),
)

Writer = make_agent(
    "Writer",
    "You transform raw facts into polished prose.",
    Claude(),
)

Critic = make_agent(
    "Critic",
    "You review drafts and suggest concrete improvements.",
    Claude(),
)

# The LLM sees researcher, writer, and critic as callable tools.
# It decides when and in what order to invoke them.
Orchestrator = make_agent(
    "Orchestrator",
    (
        "You coordinate research and writing. "
        "Use researcher to gather facts, writer to draft, critic to review. "
        "Return the final polished text."
    ),
    Claude(),
    sub_agents={
        "researcher": Researcher,
        "writer": Writer,
        "critic": Critic,
    },
)

with Orchestrator.get_proxy() as proxy:
    report = proxy.instruct("Write a report on climate change.").get()
```

Each sub-agent call is stateless (fresh start/stop per call). For sub-agents that must maintain a session, use the class-based approach and hold a reference to a running actor inside a `@tool` method.

### 2.3 Injecting tools

Pass plain callables or `@tool`-decorated functions via the `tools` parameter:

```python
from actor_ai import make_agent, tool, Claude

@tool("Count words in a text.")
def word_count(self, text: str) -> int:
    return len(text.split())

def celsius_to_fahrenheit(self, celsius: float) -> float:
    "Convert Celsius to Fahrenheit."
    return celsius * 9 / 5 + 32

Analyst = make_agent(
    "Analyst",
    "Use tools for all calculations. Never guess.",
    Claude(),
    tools=[word_count, celsius_to_fahrenheit],   # auto-decorated if missing @tool
)
```

### 2.4 Class-based approach (for complex agents)

Use a full subclass when you need lifecycle hooks (`on_start`), access to `self` state inside tools, or logic that does not fit a simple function:

```python
from actor_ai import AIActor, tool, Claude

class StatefulAnalyst(AIActor):
    system_prompt = "You analyse data. Use tools."
    provider      = Claude()

    def on_start(self) -> None:
        super().on_start()
        self._cache: dict = {}    # per-instance state

    @tool("Look up a value, caching results.")
    def lookup(self, key: str) -> str:
        if key not in self._cache:
            self._cache[key] = expensive_lookup(key)
        return self._cache[key]
```

**Rule of thumb:** reach for `make_agent()` first. Switch to a class only when `self` state or `on_start()` is genuinely needed.

### 2.5 Agent without a provider

`provider` defaults to `None`. An agent without a provider behaves like a plain pykka `ThreadingActor` — it can receive messages, manage memory, and participate in a `Chorus` without ever calling an LLM. Calling `instruct()` raises `RuntimeError`.

```python
DataNode = make_agent("DataNode", "Pure actor — no LLM.")

ref = DataNode.start()
ref.proxy().remember("context", "quarterly report").get()
ref.stop()
```

### 2.6 AIActor class attributes

When using the class-based approach, the following class attributes configure the agent:

| Attribute | Type | Default | Description |
|---|---|---|---|
| `system_prompt` | `str` | `"You are a helpful AI agent."` | System prompt sent on every call |
| `provider` | `LLMProvider \| None` | `None` | Which LLM backend to use |
| `max_tokens` | `int` | `4096` | Maximum completion tokens |
| `max_history` | `int` | `0` | Keep last N turns; 0 = unlimited |
| `ledger` | `Ledger \| None` | `None` | Attach for token accounting |
| `actor_name` | `str \| None` | `None` | Label in accounting (default: class name) |
| `monitoring` | `bool` | `False` | Forward metadata to LiteLLM |

### 2.7 instruct()

```python
reply: str = proxy.instruct(
    instruction,            # str | Path | IO[str]: the user message
    history=None,           # list[dict] | None: explicit message list
    use_session=True,       # bool: accumulate in rolling session
).get()
```

`instruction` accepts three input forms:

| Input | Behaviour |
|---|---|
| `str` | Used directly as the message text |
| `pathlib.Path` (or any `os.PathLike`) | File is read with UTF-8 encoding |
| Readable stream (`IO[str]` / `IO[bytes]`) | Stream is read; bytes are decoded as UTF-8 |

```python
from pathlib import Path
import io

proxy.instruct("What is the capital of France?").get()          # plain string
proxy.instruct(Path("prompt.txt")).get()                        # file path
proxy.instruct(open("prompt.txt")).get()                        # text file object
proxy.instruct(io.StringIO("Summarise this.")).get()            # StringIO
proxy.instruct(io.BytesIO(b"Summarise this.")).get()            # BytesIO
```

- When `use_session=True` and `history=None` (default), the actor appends the turn to its session.
- When `use_session=False`, the call is stateless — the session is not read or written.
- When `history` is provided, it is used as-is (the session is ignored).
- Raises `RuntimeError` if `provider` is `None`.
- Raises `TypeError` if `instruction` is not a str, path-like, or readable stream.

---

## 3. Session management

The actor maintains a rolling list of `{"role": "user"/"assistant", "content": "…"}` messages. This list is prepended to every `instruct()` call so the LLM has full conversational context.

```python
ChatBot = make_agent(
    "ChatBot",
    "You are a friendly assistant.",
    Claude(),
    max_history=10,     # keep last 10 turns (20 messages)
)

with ChatBot.get_proxy() as proxy:
    proxy.instruct("My name is Alice.").get()
    reply = proxy.instruct("Do you remember my name?").get()
    # → "Yes, you told me your name is Alice."

    session = proxy.get_session().get()   # list[dict]
    print(len(session))                   # 4 (2 turns × 2 messages)
```

### max_history

When `max_history > 0`, the actor keeps only the last `max_history` turns (2 × max_history messages). Older turns are dropped automatically after each call.

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

`AIActor` has three memory tiers, each with a different scope and lifetime:

| Tier | Methods | Scope | Injected as |
|---|---|---|---|
| Short-term | session history | Current conversation | Message history |
| Working | `remember_working` / `forget_working` | Current task / session | `Working memory:` in system prompt |
| Long-term | `remember` / `forget` | Durable across sessions | `Known facts:` in system prompt |

### 4.1 Long-term memory

Long-term memory stores facts that persist across sessions. Use it for durable user context: name, preferences, role.

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

Long-term memory **survives `clear_session()`**.

### 4.2 Working memory

Working memory stores task-scoped facts. Use it for ephemeral context that applies to a specific task: current goal, intermediate results, document being processed.

```python
proxy.remember_working("task",    "write Q3 summary").get()
proxy.remember_working("context", "cloud revenue up 12%").get()

# System prompt also contains:
# "Working memory:
# - task: write Q3 summary
# - context: cloud revenue up 12%"

proxy.forget_working("context").get()
proxy.get_working_memory()        # -> dict[str, str]
proxy.clear_working_memory()      # remove all working facts without resetting session
```

Working memory is **cleared by `clear_session()`**.

### 4.3 Session history (short-term)

Session history is the rolling message log automatically maintained by `instruct()`. It is covered in [Section 3](#3-session-management).

### 4.4 Summary

```python
# Long-term memory
proxy.remember("name", "Alice").get()       # store
proxy.forget("name").get()                  # remove one key
proxy.get_memory().get()                    # inspect -> dict

# Working memory
proxy.remember_working("task", "draft Q3").get()  # store
proxy.forget_working("task").get()                # remove one key
proxy.get_working_memory().get()                  # inspect -> dict
proxy.clear_working_memory().get()                # clear all

# clear_session() resets conversation AND working memory; long-term persists
proxy.clear_session().get()
```

---

## 5. Tool calling and sub-agents

### 5.1 Sub-agents via `make_agent()`

The simplest way to wire sub-agents is the `sub_agents` parameter of `make_agent()` (see [Section 2.2](#22-agent-hierarchies-with-sub_agents)). Each named entry becomes a `@tool` method that the LLM can call to delegate work.

### 5.2 `@tool` decorator

For the class-based approach, decorate methods with `@tool`:

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

```python
# Form 1: description from docstring
@tool
def my_method(self, x: int) -> str:
    """Description seen by the LLM."""
    ...

# Form 2: explicit description string
@tool("Explicit description.")
def my_method(self, x: int) -> str:
    ...
```

### 5.3 Sub-agents as `@tool` methods (class-based)

When you need persistent sub-agents that maintain their own session, start them before the orchestrator and capture their refs in `@tool` closures or `on_start()`:

```python
class Orchestrator(AIActor):
    system_prompt = "Coordinate research and writing."
    provider      = Claude()

    def on_start(self) -> None:
        super().on_start()
        self._researcher = Researcher.start()
        self._writer     = Writer.start()

    @tool("Delegate a research question to the research specialist.")
    def research(self, query: str) -> str:
        return self._researcher.proxy().instruct(query).get()

    @tool("Delegate a writing task to the writing specialist.")
    def write(self, content: str) -> str:
        return self._writer.proxy().instruct(content).get()
```

### 5.4 Type annotations and JSON Schema

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

### 5.5 How tool calling works

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

Researcher = make_agent("Researcher", "You research.", Claude())
Researcher = make_agent("Researcher", "You research.",
    Claude("claude-opus-4-7", temperature=0.2, top_p=0.9, timeout=30.0))
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

Writer = make_agent("Writer", "You write.", GPT())
Writer = make_agent("Writer", "You write.",
    GPT("gpt-4o-mini", temperature=0.0, seed=42,
        frequency_penalty=0.3, response_format={"type": "json_object"}))
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

agent = make_agent("Agent", "You assist.", Gemini())
agent = make_agent("Agent", "You assist.",
    Gemini("gemini-1.5-pro", temperature=0.4, top_p=0.95))
```

Reads `GOOGLE_API_KEY`. Uses Google's OpenAI-compatible endpoint.

### Mistral

```python
from actor_ai import Mistral

agent = make_agent("Agent", "You assist.", Mistral())
agent = make_agent("Agent", "You assist.",
    Mistral("mistral-small-latest", temperature=0.7))
```

Reads `MISTRAL_API_KEY`.

### DeepSeek

```python
from actor_ai import DeepSeek

agent = make_agent("Agent", "You assist.", DeepSeek())
agent = make_agent("Agent", "You assist.",
    DeepSeek("deepseek-reasoner", temperature=0.0, seed=0))
```

Reads `DEEPSEEK_API_KEY`.

### Copilot (GitHub)

Routes requests through GitHub Copilot's OpenAI-compatible endpoint. Requires a GitHub account with an active Copilot subscription.

Token resolution order (first match wins):

1. `api_key` constructor argument
2. `GITHUB_TOKEN` environment variable
3. `gh auth token` CLI — works automatically when the [GitHub CLI](https://cli.github.com/) is authenticated

```python
from actor_ai import Copilot, CopilotModel

agent = make_agent("Agent", "You assist.", Copilot())
agent = make_agent("Agent", "You assist.", Copilot("claude-sonnet-4-5"))

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

Passing any other model string raises `ValueError` at construction time. Use `Copilot.MODELS` (a `frozenset[str]`) for runtime validation, or the `CopilotModel` `Literal` for IDE autocompletion.

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

### LiteLLM

See [Section 10 — Monitoring with LiteLLM](#10-monitoring-with-litellm) for full details.

```python
from actor_ai import LiteLLM

agent = make_agent("Agent", "You assist.", LiteLLM("openai/gpt-4o"))
agent = make_agent("Agent", "You assist.",
    LiteLLM("anthropic/claude-sonnet-4-6",
            temperature=0.2, max_retries=3,
            success_callbacks=["langfuse"]))
```

### Swapping providers at runtime

Because pykka's proxy supports attribute assignment, you can replace the provider on a running actor:

```python
ref.proxy().provider = GPT("gpt-4o-mini")
```

The next `instruct()` call will use the new provider.

---

## 7. Chorus

`Chorus` manages a named group of actors and provides three coordination patterns. Members can be agents created with `make_agent()`, class-based `AIActor` subclasses, other `Chorus` instances, or any plain pykka actor that exposes an `instruct()` method.

```python
from actor_ai import Chorus, make_agent, Claude, GPT

Researcher = make_agent("Researcher", "You research.", Claude())
Writer     = make_agent("Writer",     "You write.",    GPT("gpt-4o"))

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
chorus.add("editor", editor_ref).get()
chorus.remove("editor").get()

chorus.join("reviewer", reviewer_ref).get()   # alias for add()
chorus.leave("reviewer").get()                # alias for remove()

names: list[str] = chorus.agents().get()
chorus.stop_agents(names=["reviewer"]).get()  # stop specific agents
chorus.stop_agents().get()                    # stop all agents
```

### Memory broadcast

`remember()` / `forget()` calls propagate to **all members** automatically, including nested choruses and non-AI actors:

```python
chorus.remember("audience", "general public").get()
chorus.remember("style", "academic", names=["researcher"]).get()
chorus.forget("style").get()
chorus.forget("audience", names=["writer"]).get()
```

### Nested choruses

```python
inner = Chorus.start(agents={"a": a_ref, "b": b_ref}, type="team")
outer = Chorus.start(agents={"inner": inner, "c": c_ref}, type="department")

outer.proxy().broadcast("Announce yourselves.").get()
outer.proxy().remember("project", "Atlas").get()
```

### Non-AI actors as members

Any pykka actor that exposes an `instruct(instruction)` method can be a Chorus member:

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
from actor_ai import WorkflowState, make_agent, Claude

Draft  = make_agent("Drafter",  "You draft documents.", Claude())
Review = make_agent("Reviewer", "You review quality.",  Claude())

draft_ref  = Draft.start()
review_ref = Review.start()

WorkflowState(chorus=draft_ref,  instruction="{input}")
WorkflowState(chorus=review_ref, instruction="Review this:\n{output}")

# Parallel actors — all fired simultaneously, replies combined as "name: reply\n…"
WorkflowState(
    chorus={"researcher": r_ref, "critic": c_ref},
    instruction="Analyse: {input}",
)
```

**Instruction templates** support two placeholders:

| Placeholder | Expands to |
|---|---|
| `{input}` | The original instruction passed to `run()` / `step()` |
| `{output}` | The reply produced by the immediately preceding state |

**WorkflowTransition** is a directed edge:

```python
from actor_ai import WorkflowTransition

WorkflowTransition("draft", "review", guard=lambda r: "ready" in r)
WorkflowTransition("review", "draft", on_event="reject")
WorkflowTransition("analyse", "approve",
                   guard=lambda r: "approved" in r,
                   on_event="force_approve")
```

### Creating a workflow

```python
from actor_ai import Workflow, WorkflowState, WorkflowTransition, make_agent, Claude

Draft    = make_agent("Drafter",   "You draft documents.",     Claude())
Reviewer = make_agent("Reviewer",  "You review quality.",      Claude())
Approver = make_agent("Approver",  "You give final sign-off.", Claude())

wf = Workflow.start(
    states={
        "draft":   WorkflowState(Draft.start(),    instruction="{input}"),
        "review":  WorkflowState(Reviewer.start(), instruction="Review:\n{output}"),
        "approve": WorkflowState(Approver.start(), instruction="Finalise:\n{output}"),
    },
    transitions=[
        WorkflowTransition("draft",  "review",  guard=lambda r: "ready" in r),
        WorkflowTransition("review", "approve", guard=lambda r: "approved" in r),
        WorkflowTransition("review", "draft",   on_event="reject"),
    ],
    initial_state="draft",
)

output = wf.proxy().run("Please draft a Q3 proposal.").get()
wf.stop()
```

### Blocking execution — run() and step()

**`run(instruction)`** — executes from the current state and follows guard transitions until no guard matches (terminal state). Blocks the calling thread.

```python
output: str = wf.proxy().run("Draft a proposal.").get()
```

**`step(instruction)`** — executes the current state exactly once and applies the first matching guard transition.

```python
out1 = wf.proxy().step("Begin the process.").get()
out2 = wf.proxy().step().get()   # instruction=None → uses last output as {input}
```

### Event transitions

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

wf.proxy().event("approve").get()   # actor mailbox is free during detached run
done.wait(timeout=30)
```

### Runtime modification

```python
wf.proxy().add_state("polish", WorkflowState(polish_ref, instruction="Polish:\n{output}")).get()
wf.proxy().add_transition(WorkflowTransition("review", "polish", on_event="polish")).get()
wf.proxy().remove_state("old_state").get()
wf.proxy().remove_transitions("review").get()           # all from "review"
wf.proxy().remove_transitions("review", "draft").get()  # only review→draft
```

### State inspection and control

```python
wf.proxy().set_state("review").get()          # force-jump (raises KeyError if unknown)
state:  str | None = wf.proxy().current_state().get()
output: str        = wf.proxy().last_output().get()
names:  list[str]  = wf.proxy().states().get()
```

### Parallel actor states

Pass a `dict[str, ActorRef]` as `chorus` to fire multiple actors simultaneously:

```python
WorkflowState(
    chorus={"researcher": r_ref, "critic": c_ref},
    instruction="Analyse: {input}",
)
# Combined output: "researcher: …\ncritic: …"

WorkflowTransition(
    "analyse", "summarise",
    guard=lambda r: "approved" in r   # matches if any actor's reply contains it
)
```

---

## 9. Accounting

The accounting layer records every `instruct()` call and calculates token spend.

### Per-actor usage tracking

Every `AIActor` instance tracks token usage internally, independent of any `Ledger`. Use this for lightweight, per-actor counters without the overhead of a shared ledger.

```python
with MyAgent.get_proxy() as proxy:
    proxy.instruct("First call.").get()
    proxy.instruct("Second call.").get()

    usage = proxy.get_usage().get()
    print(usage.input_tokens)   # total input tokens across all calls
    print(usage.output_tokens)  # total output tokens across all calls
    print(usage.total_tokens)   # input + output

    proxy.reset_usage().get()   # reset counters to zero
    usage = proxy.get_usage().get()
    print(usage.total_tokens)   # 0
```

`get_usage()` returns a `UsageSummary` snapshot; it is not affected by subsequent calls until the next `get_usage()` call. `reset_usage()` resets only the counter — it does not affect the session or memory.

### Ledger

```python
from actor_ai import Ledger, Rates, make_agent, Claude

ledger = Ledger()

Agent = make_agent(
    "Agent",
    "You assist.",
    Claude(),
    ledger=ledger,
    actor_name="my-agent",
)
```

Each completed `instruct()` call appends one `LedgerEntry`:
- `actor_name`, `model`, `input_tokens`, `output_tokens`, `timestamp`, `session_id`

#### Reading the ledger

```python
entries: list[LedgerEntry]          = ledger.entries()
entries_for_actor:   list[LedgerEntry] = ledger.entries_for_actor("my-agent")
entries_for_session: list[LedgerEntry] = ledger.entries_for_session(session_id)
entries_for_model:   list[LedgerEntry] = ledger.entries_for_model("claude-sonnet-4-6")

total: UsageSummary                 = ledger.total_usage()
by_actor:  dict[str, UsageSummary]  = ledger.usage_by_actor()
by_model:  dict[str, UsageSummary]  = ledger.usage_by_model()
by_session: dict[str, UsageSummary] = ledger.usage_by_session()

rates = Rates.default()
total_cost:    float                = ledger.total_cost(rates)
by_actor_cost: dict[str, float]     = ledger.cost_by_actor(rates)
by_model_cost: dict[str, float]     = ledger.cost_by_model(rates)

summary: dict = ledger.summary(rates)
n: int = len(ledger)
ledger.clear()
```

### Rates

```python
from actor_ai import Rates, ModelRate

rates = Rates.default()
rates = Rates.from_dict({"my-model": {"input": 1.50, "output": 3.00}})
rates.set("gpt-4o", input_per_million=2.50, output_per_million=10.00)
rate: ModelRate | None = rates.get_rate("gpt-4o")
cost: float = rates.cost("gpt-4o", usage_summary)
"gpt-4o" in rates   # True
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

When `monitoring=True`, the actor creates a `MonitoringContext` for every `instruct()` call and passes it to the provider. The `LiteLLM` provider forwards this context as the `metadata` kwarg to `litellm.completion()`.

```python
from actor_ai import make_agent, LiteLLM

MonitoredAgent = make_agent(
    "SupportBot",
    "You are a helpful support assistant.",
    LiteLLM("openai/gpt-4o"),
    actor_name="support-bot",
    monitoring=True,
)
```

When `monitoring=True`, every `instruct()` call forwards this to `litellm.completion()`:

```python
metadata = {
    "actor_name": "support-bot",
    "session_id": "<uuid>",
}
```

### Registering callbacks

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

The `monitoring_context` parameter is passed to every provider's `run()` method. Non-LiteLLM providers silently ignore it, so you can safely set `monitoring=True` on any agent and switch providers later.

---

## 11. Message API

As a pykka actor, `AIActor` accepts raw message objects in addition to the proxy API.

```python
from actor_ai import Instruct, Remember, Forget

Instruct(
    instruction: str,
    history:     list[dict] = [],
    use_session: bool       = True,
)
Remember(key: str, value: str)
Forget(key: str)
```

### ask() — synchronous

```python
reply = ref.ask(Instruct("What is 2 + 2?"))
ref.ask(Remember("name", "Alice"))
ref.ask(Forget("name"))
```

### tell() — fire-and-forget

```python
ref.tell(Remember("name", "Alice"))
ref.tell(Forget("name"))
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

### `make_agent()` — agent factory

```python
make_agent(
    name: str,
    system_prompt: str,
    provider: LLMProvider | None = None,
    *,
    tools: list[Callable] | None = None,
    sub_agents: dict[str, type[AIActor]] | None = None,
    max_tokens: int = 4096,
    max_history: int = 0,
    ledger: Ledger | None = None,
    actor_name: str | None = None,
    monitoring: bool = False,
) -> type[AIActor]
```

### AIActor public API

```python
# Instructions
instruct(instruction: str | Path | IO[str], history=None, use_session=True) -> str

# Long-term memory
remember(key: str, value: str) -> None
forget(key: str) -> None
get_memory() -> dict[str, str]

# Working memory (task-scoped; cleared by clear_session())
remember_working(key: str, value: str) -> None
forget_working(key: str) -> None
get_working_memory() -> dict[str, str]
clear_working_memory() -> None

# Session
clear_session() -> None      # also clears working memory; preserves long-term memory
get_session() -> list[dict]
get_session_id() -> str

# Token usage tracking
get_usage() -> UsageSummary   # accumulated since last reset_usage()
reset_usage() -> None

# Context managers (classmethods)
get_proxy()   -> Generator[ActorProxy]       # synchronous context manager
aget_proxy()  -> AsyncGenerator[ActorProxy]  # async context manager
```

### Chorus public API

```python
add(name: str, ref: ActorRef) -> None
remove(name: str) -> None
join(name: str, ref: ActorRef) -> None    # alias for add()
leave(name: str) -> None                  # alias for remove()
agents() -> list[str]
stop_agents(names: list[str] | None = None) -> None

instruct(instruction: str) -> str                          # broadcast (single arg)
instruct(name: str, instruction: str, **kwargs) -> str     # single agent (two args)
broadcast(instruction: str, **kwargs) -> dict[str, str]
pipeline(names: list[str], instruction: str, **kwargs) -> str

remember(key: str, value: str, names: list[str] | None = None) -> None
forget(key: str, names: list[str] | None = None) -> None

type: ChorusType   # attribute (readable via proxy)
```

### Workflow public API

```python
add_state(name: str, state: WorkflowState) -> None
remove_state(name: str) -> None
add_transition(transition: WorkflowTransition) -> None
remove_transitions(source: str, target: str | None = None) -> None
set_state(name: str) -> None

states() -> list[str]
current_state() -> str | None
last_output() -> str

step(instruction: str | None = None) -> str
run(instruction: str | None = None) -> str

run_detached(
    instruction: str | None = None,
    on_complete: Callable[[str], None] | None = None,
    on_error: Callable[[Exception], None] | None = None,
) -> None

event(name: str) -> bool

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
record(actor_name, model, input_tokens, output_tokens, session_id=None) -> None
clear() -> None

entries() -> list[LedgerEntry]
entries_for_actor(actor_name) -> list[LedgerEntry]
entries_for_session(session_id) -> list[LedgerEntry]
entries_for_model(model) -> list[LedgerEntry]

total_usage() -> UsageSummary
usage_by_actor() -> dict[str, UsageSummary]
usage_by_model() -> dict[str, UsageSummary]
usage_by_session() -> dict[str, UsageSummary]

total_cost(rates: Rates) -> float
cost_by_actor(rates: Rates) -> dict[str, float]
cost_by_model(rates: Rates) -> dict[str, float]
cost_by_session(rates: Rates) -> dict[str, float]

summary(rates: Rates | None = None) -> dict
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

### `@tool` decorator

```python
@tool                           # description from docstring
@tool("Explicit description")   # explicit description
```

### Exported symbols

```python
from actor_ai import (
    # Agent factory (recommended)
    make_agent,
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
