# actor-ai

**Multi-provider AI agents built on [pykka](https://github.com/jodal/pykka).**

`actor-ai` extends pykka's actor framework so each agent can process natural-language instructions, call tools, maintain conversation sessions, remember facts, track token spend, monitor traffic, coordinate with other agents via Chorus, and execute multi-step state-machine workflows — all with a clean, provider-agnostic API.

## Installation

```bash
pip install actor-ai          # or: uv add actor-ai
```

Requires Python ≥ 3.14 and one (or more) provider API keys.

## Quick start

```python
from actor_ai import AIActor, Claude

class Assistant(AIActor):
    system_prompt = "You are a helpful assistant."
    provider = Claude()          # requires ANTHROPIC_API_KEY

ref = Assistant.start()
reply = ref.proxy().instruct("What is the capital of France?").get()
print(reply)   # → "The capital of France is Paris."
ref.stop()
```

`AIActor` can also run without a provider — it behaves as a plain pykka actor (memory, session, and message-passing still work; `instruct()` raises `RuntimeError` if called without a provider):

```python
class DataActor(AIActor):
    pass  # no provider — pure actor, no LLM needed
```

## Providers

| Class | Backend | Environment variable |
|---|---|---|
| `Claude` | Anthropic | `ANTHROPIC_API_KEY` |
| `GPT` | OpenAI | `OPENAI_API_KEY` |
| `Gemini` | Google AI (OpenAI-compat) | `GOOGLE_API_KEY` |
| `Mistral` | Mistral AI | `MISTRAL_API_KEY` |
| `DeepSeek` | DeepSeek | `DEEPSEEK_API_KEY` |
| `Copilot` | GitHub Copilot (OpenAI-compat) | `GITHUB_TOKEN` |
| `LiteLLM` | Any (100+ models) | depends on model |

`Copilot` routes requests through GitHub Copilot's OpenAI-compatible endpoint and supports multiple underlying models from a single token:

```python
from actor_ai import AIActor, Copilot

class Assistant(AIActor):
    system_prompt = "You are a helpful coding assistant."
    provider = Copilot()                    # gpt-4o (default)
    provider = Copilot("claude-sonnet-4-5") # Claude via Copilot
    provider = Copilot("gemini-2.0-flash")  # Gemini via Copilot

print(Copilot.MODELS)  # frozenset of valid model strings
```

Valid models: `gpt-4o`, `gpt-4o-mini`, `o1`, `o1-mini`, `o3-mini`, `claude-sonnet-4-5`, `gemini-2.0-flash`. Passing any other string raises `ValueError` at construction time.

The token is resolved in order: `api_key` argument → `GITHUB_TOKEN` env var → `gh auth token` CLI → OS keyring (GitHub CLI / VS Code).

## Chorus

`Chorus` groups named actors and lets you coordinate them at three levels.  Members can be `AIActor` instances, other `Chorus` instances, or any plain pykka actor that exposes an `instruct()` method.

```python
from actor_ai import Chorus, ChorusType

# Create a typed chorus
team = Chorus.start(type="team", agents={"writer": writer_ref, "editor": editor_ref})

# Single-agent instruct
reply = team.proxy().instruct("writer", "Draft a proposal.").get()

# Broadcast to all members in parallel
replies = team.proxy().broadcast("Introduce yourself.").get()

# Pipeline: chain output of each member to the next
final = team.proxy().pipeline(["writer", "editor"], "Write a report.").get()

# Actors can join and leave at runtime
team.proxy().join("reviewer", reviewer_ref).get()
team.proxy().leave("editor").get()
```

`ChorusType` is a `Literal["system", "project", "team", "department", "custom"]` — set at construction and readable via proxy.

Nested choruses: a `Chorus` can be a member of another `Chorus`. `broadcast()` and memory propagation cascade correctly through nesting levels.

## Workflow

`Workflow` implements a state-machine that orchestrates Chorus and actor instances. Transitions fire either when a reply matches a guard predicate or when a named event is dispatched. States and transitions can be added or replaced at runtime on a live workflow.

```python
from actor_ai import Workflow, WorkflowState, WorkflowTransition

wf = Workflow.start(
    states={
        "draft":  WorkflowState(draft_chorus,  instruction="{input}"),
        "review": WorkflowState(review_chorus, instruction="Review this:\n{output}"),
        # Parallel actors — fire simultaneously, replies combined:
        "analyse": WorkflowState(
            chorus={"researcher": r_ref, "critic": c_ref},
            instruction="Analyse: {input}",
        ),
    },
    transitions=[
        WorkflowTransition("draft",  "review",  guard=lambda r: "ready" in r),
        WorkflowTransition("review", "draft",   on_event="reject"),
    ],
    initial_state="draft",
)

# Blocking run — follows guard transitions until terminal:
output = wf.proxy().run("Draft a proposal.").get()

# Fire a named event:
wf.proxy().event("reject").get()

# Non-blocking run — actor mailbox stays free for events during execution:
wf.proxy().run_detached(
    "Draft again.",
    on_complete=lambda out: print("Done:", out),
).get()

# Add a state at runtime:
wf.proxy().add_state("approve", WorkflowState(approve_chorus, "{output}")).get()
wf.proxy().add_transition(WorkflowTransition("review", "approve", on_event="approve")).get()
```

## Key features

- **Multi-turn sessions** — rolling conversation history, configurable window (`max_history`)
- **Long-term memory** — `remember(key, value)` / `forget(key)` facts injected into every system prompt
- **Tool calling** — decorate methods with `@tool` to expose them to the LLM
- **Chorus** — group actors as a named team with `broadcast`, `pipeline`, `join`/`leave`, typed (`ChorusType`), nestable
- **Workflow** — state-machine orchestration with guard and event transitions, parallel actor states, runtime modification, non-blocking `run_detached()`
- **Accounting** — track token usage and cost per actor, model, and session (`Ledger`, `Rates`)
- **Monitoring** — forward per-call metadata to LiteLLM callbacks (Langfuse, Helicone, custom)

## Examples

The [`examples/`](examples/) directory contains self-contained, runnable scripts. Each example uses a fake scripted provider so no API key is required.

```
examples/
  01_hello_world.py         — One-turn interaction
  02_session.py             — Session history, max_history, clear_session
  03_memory.py              — remember() / forget()
  04_tools.py               — @tool decorator and tool calling
  05_providers.py           — All provider configurations
  06_chorus.py              — Chorus: instruct, broadcast, pipeline, memory
  07_accounting.py          — Ledger and Rates
  08_monitoring.py          — LiteLLM monitoring
  09_messages.py            — Instruct / Remember / Forget message objects
  10_advanced.py            — Full pipeline combining all features
  11_copilot.py             — Copilot provider with token auto-resolution
  12_chorus_advanced.py     — ChorusType, join/leave, nested choruses, non-AI actors
  13_workflow.py            — Workflow state machine: run, step, event, runtime modification
  14_workflow_parallel.py   — Parallel actor states and run_detached()
```

Run any example:

```bash
uv run python examples/01_hello_world.py
```

## User Manual

See [MANUAL.md](https://github.com/oliben67/actor-ai/blob/master/MANUAL.md) for the complete reference.

## License

MIT
