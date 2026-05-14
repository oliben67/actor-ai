# Workflow Diagrams

Visual state-machine diagrams for the workflow examples.
All diagrams use [Mermaid](https://mermaid.js.org/) syntax and render natively in GitHub, GitLab, and most Markdown previewers.

---

## Example 13 — Core Workflow Features

### 13-1: Basic `run()` with guard transition

A guard fires automatically when the draft reply contains `"ready"`.

```mermaid
stateDiagram-v2
    [*] --> draft
    draft --> review : [guard] ready in reply
    review --> [*]
```

---

### 13-2: Manual `step()` stepping

Two states connected by a guard on `"complete"`.
Each `step()` call executes the current state exactly once and advances if the guard matches.

```mermaid
stateDiagram-v2
    [*] --> step_a
    step_a --> step_b : [guard] complete in reply
    step_b --> [*]
```

---

### 13-3: `event()` transition

Guard fires forward (draft → review on `"submitted"`); a named event `reject` sends the workflow back.

```mermaid
stateDiagram-v2
    [*] --> draft
    draft --> review : [guard] submitted in reply
    review --> draft : [event] reject
    review --> [*]
```

---

### 13-4: Both guard and event transitions on the same state

`analyse` can advance by guard (to `approve`) or by event `escalate` (to `escalate`).

```mermaid
stateDiagram-v2
    [*] --> analyse
    analyse --> approve  : [guard] approved in reply
    analyse --> escalate : [event] escalate
    approve  --> [*]
    escalate --> [*]
```

---

### 13-5: Runtime modification — add state and transition

**Before** — only `draft` is registered at startup:

```mermaid
stateDiagram-v2
    [*] --> draft
    draft --> [*]
```

**After** — `polish` and its guard transition are added at runtime via `add_state()` + `add_transition()`:

```mermaid
stateDiagram-v2
    [*] --> draft
    draft --> polish : [guard] polish in reply
    polish --> [*]
```

---

### 13-6: `set_state()` — force-jump

The workflow is force-jumped directly to `approve`, skipping `review`.

```mermaid
stateDiagram-v2
    [*] --> draft
    draft --> review   : normal path
    review --> approve : normal path
    draft --> approve  : set_state() force-jump
    approve --> [*]
```

---

### 13-7: `current_state()` and `last_output()` inspection

A single-state terminal workflow. After `run()`, both inspection methods reflect the completed execution.

```mermaid
stateDiagram-v2
    [*] --> start
    start --> [*]

    note right of start
        current_state returns start
        last_output returns reply text
    end note
```

---

## Example 14 — Parallel Actors and Async Execution

### 14-1: Parallel actors in a single state

Both actors receive the same instruction simultaneously via a fork; their replies are combined at the join.

```mermaid
stateDiagram-v2
    [*] --> fork_state
    state fork_state <<fork>>
    fork_state --> researcher
    fork_state --> critic
    state join_state <<join>>
    researcher --> join_state
    critic --> join_state
    join_state --> [*]

    note right of join_state
        combined output
        researcher then critic
    end note
```

---

### 14-2: Guard transition fires on combined parallel output

The guard matches a keyword from one of the parallel actors; both are wrapped in a composite `analyse` state.

```mermaid
stateDiagram-v2
    [*] --> analyse

    state analyse {
        state fork_a <<fork>>
        [*] --> fork_a
        fork_a --> researcher
        fork_a --> critic
        state join_a <<join>>
        researcher --> join_a
        critic --> join_a
        join_a --> [*]
    }

    analyse --> summarise : [guard] approved in combined reply
    summarise --> [*]
```

---

### 14-3: Mixed states — single actor then parallel actors

A single-actor `summarise` state transitions unconditionally into a parallel `analyse` state.

```mermaid
stateDiagram-v2
    [*] --> summarise

    summarise --> analyse : [guard] always true

    state analyse {
        state fork_b <<fork>>
        [*] --> fork_b
        fork_b --> researcher
        fork_b --> critic
        state join_b <<join>>
        researcher --> join_b
        critic --> join_b
        join_b --> [*]
    }

    analyse --> [*]
```

---

### 14-4: `run_detached()` with `on_complete` callback

The workflow runs in a background thread. The calling thread is free immediately and receives the result via callback.

```mermaid
sequenceDiagram
    participant Caller
    participant Workflow
    participant BgThread as Background Thread
    participant LLMActor as Actor

    Caller->>Workflow: run_detached(instruction, on_complete=cb)
    Workflow->>BgThread: start()
    Workflow-->>Caller: returns immediately

    Note over Caller: free to do other work

    BgThread->>Workflow: prepare_step(instruction)
    Workflow-->>BgThread: (chorus, formatted)
    BgThread->>LLMActor: instruct(formatted)
    LLMActor-->>BgThread: output
    BgThread->>Workflow: commit_step(output)
    Workflow-->>BgThread: False (terminal)
    BgThread->>Caller: on_complete(output)
```

---

### 14-5: `run_detached()` with `on_error` callback

A missing state triggers a `KeyError` in the background thread and is delivered via `on_error`.

```mermaid
sequenceDiagram
    participant Caller
    participant Workflow
    participant BgThread as Background Thread

    Caller->>Workflow: run_detached(instruction, on_error=cb)
    Workflow->>BgThread: start()
    Workflow-->>Caller: returns immediately

    BgThread->>Workflow: prepare_step(instruction)
    Note over Workflow: state not found
    Workflow-->>BgThread: raises KeyError
    BgThread->>Caller: on_error(KeyError)
```

---

### 14-6: `event()` fired while `run_detached()` is running

The workflow actor's mailbox stays free during detached execution, so `event()` can be dispatched concurrently.

State machine:

```mermaid
stateDiagram-v2
    [*] --> waiting
    waiting --> advanced : [event] advance
    advanced --> [*]
```

Execution sequence:

```mermaid
sequenceDiagram
    participant Caller
    participant Workflow
    participant BgThread as Background Thread
    participant WaitingActor

    Caller->>Workflow: run_detached(instruction, on_complete=cb)
    Workflow->>BgThread: start()
    Workflow-->>Caller: returns immediately

    BgThread->>Workflow: prepare_step(instruction)
    Workflow-->>BgThread: (waiting_chorus, formatted)
    BgThread->>WaitingActor: instruct(formatted)

    Note over Caller: actor mailbox is free
    Caller->>Workflow: event(advance)
    Workflow-->>Caller: True

    WaitingActor-->>BgThread: reply
    BgThread->>Workflow: commit_step(output)
    Note over Workflow: no guard matches, terminal
    BgThread->>Caller: on_complete(last_output)
```

---

### 14-7: `prepare_step()` / `commit_step()` — custom orchestration

Three-phase manual execution: prepare reads state atomically; execute runs outside the workflow thread; commit writes output and advances.

```mermaid
sequenceDiagram
    participant CustomCode
    participant Workflow
    participant LLMActor as Actor

    CustomCode->>Workflow: prepare_step(instruction)
    Note over Workflow: reads current state, formats instruction
    Workflow-->>CustomCode: (chorus, formatted_instruction)

    Note over CustomCode: execute outside workflow thread
    CustomCode->>LLMActor: Workflow._execute(chorus, formatted_instruction)
    LLMActor-->>CustomCode: output

    CustomCode->>Workflow: commit_step(output)
    Note over Workflow: stores last_output, applies guard
    Workflow-->>CustomCode: True (state advanced)
```

State machine:

```mermaid
stateDiagram-v2
    [*] --> work
    work --> gate : [guard] always true
    gate --> [*]
```

---

## Workflow Execution Modes — Summary

```mermaid
flowchart TD
    A[Workflow.start] --> B{Choose execution mode}

    B --> C["run()\nBlocking loop\nguard transitions\nuntil terminal"]
    B --> D["step()\nSingle execution\napply guard once\nreturn output"]
    B --> E["run_detached()\nBackground thread\nactor stays free\ncallbacks for result"]

    C --> F[Final output]
    D --> G["Step output\nnew current_state"]
    E --> H["on_complete\nor on_error\ncallback"]

    I["event(name)"] --> J{Matching\ntransition?}
    J -- Yes --> K["Advance state\nreturn True"]
    J -- No  --> L["No change\nreturn False"]
```
