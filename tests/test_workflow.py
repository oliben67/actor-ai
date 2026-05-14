"""Tests for Workflow: state machine orchestration of Chorus actors."""

# Future imports (must occur at the beginning of the file):
from __future__ import annotations

# Third party imports:
import pytest

# Local imports:
from actor_ai import AIActor, Chorus, Workflow, WorkflowState, WorkflowTransition
from tests.conftest import FakeProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_chorus(replies: list[str], name: str = "Agent") -> tuple:
    """Return (chorus_ref, provider) backed by a single FakeProvider agent.

    Chorus.instruct(text) returns "agent: <reply>" because the single-arg
    form formats each member's reply as "<name>: <reply>".
    """
    prov = FakeProvider(replies)
    cls = type(name, (AIActor,), {"provider": prov})
    agent_ref = cls.start()
    chorus_ref = Chorus.start(agents={"agent": agent_ref})
    return chorus_ref, prov


def chorus_reply(reply: str) -> str:
    """Format a reply as it comes out of a single-member Chorus."""
    return f"agent: {reply}"


def make_workflow(
    states: dict,
    transitions: list | None = None,
    initial_state: str | None = None,
):
    return Workflow.start(
        states=states,
        transitions=transitions or [],
        initial_state=initial_state,
    )


# ---------------------------------------------------------------------------
# State machine management
# ---------------------------------------------------------------------------


class TestStateManagement:
    def test_empty_workflow_has_no_states(self, actor_factory):
        wf = actor_factory(Workflow)
        assert wf.proxy().states().get() == []

    def test_initial_states_registered(self, actor_factory):
        chorus_ref, _ = make_chorus(["r"])
        wf = actor_factory(Workflow, states={"a": WorkflowState(chorus_ref)})
        assert wf.proxy().states().get() == ["a"]
        chorus_ref.stop()

    def test_add_state_at_runtime(self, actor_factory):
        wf = actor_factory(Workflow)
        chorus_ref, _ = make_chorus(["r"])
        wf.proxy().add_state("new", WorkflowState(chorus_ref)).get()
        assert "new" in wf.proxy().states().get()
        chorus_ref.stop()

    def test_add_state_replaces_existing(self, actor_factory):
        chorus_a, _ = make_chorus(["a"], "A")
        chorus_b, _ = make_chorus(["b"], "B")
        wf = actor_factory(Workflow, states={"s": WorkflowState(chorus_a)})
        wf.proxy().add_state("s", WorkflowState(chorus_b)).get()
        assert wf.proxy().states().get() == ["s"]
        chorus_a.stop()
        chorus_b.stop()

    def test_remove_state(self, actor_factory):
        chorus_ref, _ = make_chorus(["r"])
        wf = actor_factory(Workflow, states={"s": WorkflowState(chorus_ref)})
        wf.proxy().remove_state("s").get()
        assert "s" not in wf.proxy().states().get()
        chorus_ref.stop()

    def test_remove_nonexistent_state_is_no_op(self, actor_factory):
        wf = actor_factory(Workflow)
        wf.proxy().remove_state("ghost").get()  # must not raise

    def test_set_state(self, actor_factory):
        chorus_ref, _ = make_chorus(["r"])
        wf = actor_factory(Workflow, states={"s": WorkflowState(chorus_ref)})
        wf.proxy().set_state("s").get()
        assert wf.proxy().current_state().get() == "s"
        chorus_ref.stop()

    def test_set_unknown_state_raises(self, actor_factory):
        wf = actor_factory(Workflow)
        with pytest.raises(KeyError):
            wf.proxy().set_state("ghost").get()

    def test_initial_state_set_at_construction(self, actor_factory):
        chorus_ref, _ = make_chorus(["r"])
        wf = actor_factory(
            Workflow,
            states={"s": WorkflowState(chorus_ref)},
            initial_state="s",
        )
        assert wf.proxy().current_state().get() == "s"
        chorus_ref.stop()

    def test_current_state_none_when_unset(self, actor_factory):
        wf = actor_factory(Workflow)
        assert wf.proxy().current_state().get() is None

    def test_add_transition_at_runtime(self, actor_factory):
        wf = actor_factory(Workflow)
        wf.proxy().add_transition(WorkflowTransition("a", "b", on_event="go")).get()

    def test_remove_transitions_all_from_source(self, actor_factory):
        wf = actor_factory(Workflow)
        wf.proxy().add_transition(WorkflowTransition("a", "b", on_event="x")).get()
        wf.proxy().add_transition(WorkflowTransition("a", "c", on_event="y")).get()
        wf.proxy().remove_transitions("a").get()
        chorus_ref, _ = make_chorus(["r"])
        wf.proxy().add_state("a", WorkflowState(chorus_ref)).get()
        wf.proxy().set_state("a").get()
        fired = wf.proxy().event("x").get()
        assert fired is False
        chorus_ref.stop()

    def test_remove_transitions_specific_target(self, actor_factory):
        chorus_a, _ = make_chorus(["r"], "RA")
        chorus_b, _ = make_chorus(["r"], "RB")
        wf = actor_factory(
            Workflow,
            states={"a": WorkflowState(chorus_a), "b": WorkflowState(chorus_b)},
            transitions=[
                WorkflowTransition("a", "b", on_event="go_b"),
                WorkflowTransition("a", "c", on_event="go_c"),
            ],
            initial_state="a",
        )
        wf.proxy().remove_transitions("a", "b").get()
        assert wf.proxy().event("go_b").get() is False
        chorus_a.stop()
        chorus_b.stop()


# ---------------------------------------------------------------------------
# step()
# ---------------------------------------------------------------------------


class TestStep:
    def test_step_executes_current_state(self, actor_factory):
        chorus_ref, _ = make_chorus(["step reply"])
        wf = actor_factory(
            Workflow,
            states={"s": WorkflowState(chorus_ref, instruction="{input}")},
            initial_state="s",
        )
        result = wf.proxy().step("ping").get()
        assert result == chorus_reply("step reply")
        chorus_ref.stop()

    def test_step_no_current_state_raises(self, actor_factory):
        wf = actor_factory(Workflow)
        with pytest.raises(ValueError):
            wf.proxy().step("hi").get()

    def test_step_unknown_current_state_raises(self, actor_factory):
        wf = actor_factory(Workflow, initial_state="ghost")
        with pytest.raises(KeyError):
            wf.proxy().step("hi").get()

    def test_step_updates_last_output(self, actor_factory):
        chorus_ref, _ = make_chorus(["the reply"])
        wf = actor_factory(
            Workflow,
            states={"s": WorkflowState(chorus_ref, instruction="{input}")},
            initial_state="s",
        )
        wf.proxy().step("go").get()
        assert wf.proxy().last_output().get() == chorus_reply("the reply")
        chorus_ref.stop()

    def test_step_applies_guard_transition(self, actor_factory):
        chorus_a, _ = make_chorus(["trigger"], "StepA")
        chorus_b, _ = make_chorus(["done"], "StepB")
        wf = actor_factory(
            Workflow,
            states={
                "a": WorkflowState(chorus_a, instruction="{input}"),
                "b": WorkflowState(chorus_b, instruction="{input}"),
            },
            transitions=[WorkflowTransition("a", "b", guard=lambda r: "trigger" in r)],
            initial_state="a",
        )
        wf.proxy().step("go").get()
        assert wf.proxy().current_state().get() == "b"
        chorus_a.stop()
        chorus_b.stop()

    def test_step_no_matching_guard_stays_in_state(self, actor_factory):
        chorus_ref, _ = make_chorus(["no match"])
        wf = actor_factory(
            Workflow,
            states={"s": WorkflowState(chorus_ref, instruction="{input}")},
            transitions=[WorkflowTransition("s", "other", guard=lambda r: "trigger" in r)],
            initial_state="s",
        )
        wf.proxy().step("go").get()
        assert wf.proxy().current_state().get() == "s"
        chorus_ref.stop()

    def test_step_instruction_template_input(self, actor_factory):
        prov = FakeProvider(["got it"])
        cls = type("TI", (AIActor,), {"provider": prov})
        agent_ref = cls.start()
        chorus_ref = Chorus.start(agents={"a": agent_ref})
        wf = actor_factory(
            Workflow,
            states={"s": WorkflowState(chorus_ref, instruction="PREFIX:{input}")},
            initial_state="s",
        )
        wf.proxy().step("original").get()
        assert prov.calls[0]["messages"][-1]["content"] == "PREFIX:original"
        chorus_ref.stop()

    def test_step_instruction_template_output(self, actor_factory):
        prov = FakeProvider(["first", "second"])
        cls = type("TO", (AIActor,), {"provider": prov})
        agent_ref = cls.start()
        chorus_ref = Chorus.start(agents={"a": agent_ref})
        wf = actor_factory(
            Workflow,
            states={"s": WorkflowState(chorus_ref, instruction="PREV:{output}")},
            initial_state="s",
        )
        wf.proxy().step("x").get()  # last_output becomes "a: first"
        wf.proxy().step("x").get()  # instruction = "PREV:a: first"
        assert "PREV:" in prov.calls[1]["messages"][-1]["content"]
        assert "first" in prov.calls[1]["messages"][-1]["content"]
        chorus_ref.stop()

    def test_step_default_instruction_is_none_uses_last_output(self, actor_factory):
        prov = FakeProvider(["first", "second"])
        cls = type("TN", (AIActor,), {"provider": prov})
        agent_ref = cls.start()
        chorus_ref = Chorus.start(agents={"a": agent_ref})
        wf = actor_factory(
            Workflow,
            states={"s": WorkflowState(chorus_ref, instruction="{input}")},
            initial_state="s",
        )
        wf.proxy().step("explicit").get()  # last_output = "a: first"
        wf.proxy().step().get()  # instruction=None → uses "a: first"
        assert prov.calls[1]["messages"][-1]["content"] == "a: first"
        chorus_ref.stop()


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


class TestRun:
    def test_run_single_terminal_state(self, actor_factory):
        chorus_ref, _ = make_chorus(["terminal"])
        wf = actor_factory(
            Workflow,
            states={"only": WorkflowState(chorus_ref, instruction="{input}")},
            initial_state="only",
        )
        result = wf.proxy().run("start").get()
        assert result == chorus_reply("terminal")
        chorus_ref.stop()

    def test_run_follows_guard_transitions(self, actor_factory):
        chorus_a, _ = make_chorus(["go"], "RunA")
        chorus_b, _ = make_chorus(["done"], "RunB")
        wf = actor_factory(
            Workflow,
            states={
                "a": WorkflowState(chorus_a, instruction="{input}"),
                "b": WorkflowState(chorus_b, instruction="{input}"),
            },
            transitions=[WorkflowTransition("a", "b", guard=lambda r: "go" in r)],
            initial_state="a",
        )
        result = wf.proxy().run("begin").get()
        assert result == chorus_reply("done")
        assert wf.proxy().current_state().get() == "b"
        chorus_a.stop()
        chorus_b.stop()

    def test_run_stops_at_terminal_state(self, actor_factory):
        chorus_a, _ = make_chorus(["trigger"], "RA2")
        chorus_b, _ = make_chorus(["final"], "RB2")
        wf = actor_factory(
            Workflow,
            states={
                "a": WorkflowState(chorus_a, instruction="{input}"),
                "b": WorkflowState(chorus_b, instruction="{input}"),
            },
            transitions=[WorkflowTransition("a", "b", guard=lambda r: "trigger" in r)],
            initial_state="a",
        )
        result = wf.proxy().run("go").get()
        assert result == chorus_reply("final")
        chorus_a.stop()
        chorus_b.stop()

    def test_run_no_current_state_raises(self, actor_factory):
        wf = actor_factory(Workflow)
        with pytest.raises(ValueError):
            wf.proxy().run("hi").get()

    def test_run_passes_input_to_all_states(self, actor_factory):
        prov_a = FakeProvider(["mid"])
        prov_b = FakeProvider(["end"])
        cls_a = type("RIA", (AIActor,), {"provider": prov_a})
        cls_b = type("RIB", (AIActor,), {"provider": prov_b})
        chorus_a = Chorus.start(agents={"a": cls_a.start()})
        chorus_b = Chorus.start(agents={"b": cls_b.start()})
        wf = actor_factory(
            Workflow,
            states={
                "a": WorkflowState(chorus_a, instruction="A:{input}"),
                "b": WorkflowState(chorus_b, instruction="B:{input}"),
            },
            transitions=[WorkflowTransition("a", "b", guard=lambda r: True)],
            initial_state="a",
        )
        wf.proxy().run("original").get()
        assert prov_a.calls[0]["messages"][-1]["content"] == "A:original"
        assert prov_b.calls[0]["messages"][-1]["content"] == "B:original"
        chorus_a.stop()
        chorus_b.stop()

    def test_run_output_propagates_between_states(self, actor_factory):
        prov_a = FakeProvider(["step1"])
        prov_b = FakeProvider(["step2"])
        cls_a = type("RPa", (AIActor,), {"provider": prov_a})
        cls_b = type("RPb", (AIActor,), {"provider": prov_b})
        chorus_a = Chorus.start(agents={"a": cls_a.start()})
        chorus_b = Chorus.start(agents={"b": cls_b.start()})
        wf = actor_factory(
            Workflow,
            states={
                "a": WorkflowState(chorus_a, instruction="{input}"),
                "b": WorkflowState(chorus_b, instruction="{output}"),
            },
            transitions=[WorkflowTransition("a", "b", guard=lambda r: True)],
            initial_state="a",
        )
        wf.proxy().run("start").get()
        # State "b" receives the reply from state "a" as its instruction ("a: step1")
        assert prov_b.calls[0]["messages"][-1]["content"] == "a: step1"
        chorus_a.stop()
        chorus_b.stop()

    def test_run_without_instruction_uses_last_output(self, actor_factory):
        chorus_ref, _ = make_chorus(["initial"])
        wf = actor_factory(
            Workflow,
            states={"s": WorkflowState(chorus_ref, instruction="{input}")},
            initial_state="s",
        )
        wf.proxy().run("first call").get()  # last_output = "agent: initial"
        chorus_ref2, prov2 = make_chorus(["second"])
        wf.proxy().add_state("s2", WorkflowState(chorus_ref2, instruction="{input}")).get()
        wf.proxy().set_state("s2").get()
        wf.proxy().run().get()
        assert prov2.calls[0]["messages"][-1]["content"] == chorus_reply("initial")
        chorus_ref.stop()
        chorus_ref2.stop()


# ---------------------------------------------------------------------------
# event()
# ---------------------------------------------------------------------------


class TestEvent:
    def test_event_fires_matching_transition(self, actor_factory):
        chorus_a, _ = make_chorus(["r"], "EA")
        chorus_b, _ = make_chorus(["r"], "EB")
        wf = actor_factory(
            Workflow,
            states={
                "a": WorkflowState(chorus_a),
                "b": WorkflowState(chorus_b),
            },
            transitions=[WorkflowTransition("a", "b", on_event="go")],
            initial_state="a",
        )
        fired = wf.proxy().event("go").get()
        assert fired is True
        assert wf.proxy().current_state().get() == "b"
        chorus_a.stop()
        chorus_b.stop()

    def test_event_returns_false_when_no_match(self, actor_factory):
        chorus_ref, _ = make_chorus(["r"])
        wf = actor_factory(
            Workflow,
            states={"s": WorkflowState(chorus_ref)},
            initial_state="s",
        )
        assert wf.proxy().event("unknown").get() is False
        chorus_ref.stop()

    def test_event_only_matches_current_state(self, actor_factory):
        chorus_a, _ = make_chorus(["r"], "Eva")
        chorus_b, _ = make_chorus(["r"], "Evb")
        wf = actor_factory(
            Workflow,
            states={
                "a": WorkflowState(chorus_a),
                "b": WorkflowState(chorus_b),
            },
            transitions=[WorkflowTransition("b", "a", on_event="back")],
            initial_state="a",
        )
        assert wf.proxy().event("back").get() is False
        chorus_a.stop()
        chorus_b.stop()

    def test_event_transition_then_run(self, actor_factory):
        chorus_a, _ = make_chorus(["draft"], "DraftC")
        chorus_b, _ = make_chorus(["reviewed"], "ReviewC")
        wf = actor_factory(
            Workflow,
            states={
                "draft": WorkflowState(chorus_a, instruction="{input}"),
                "review": WorkflowState(chorus_b, instruction="{input}"),
            },
            transitions=[
                WorkflowTransition("draft", "review", on_event="submit"),
            ],
            initial_state="draft",
        )
        wf.proxy().event("submit").get()
        assert wf.proxy().current_state().get() == "review"
        result = wf.proxy().run("check this").get()
        assert result == chorus_reply("reviewed")
        chorus_a.stop()
        chorus_b.stop()

    def test_both_guard_and_event_transitions_defined(self, actor_factory):
        chorus_a, _ = make_chorus(["ok"], "BGA")
        chorus_b, _ = make_chorus(["via_guard"], "BGB")
        chorus_c, _ = make_chorus(["via_event"], "BGC")
        wf = actor_factory(
            Workflow,
            states={
                "a": WorkflowState(chorus_a, instruction="{input}"),
                "b": WorkflowState(chorus_b, instruction="{input}"),
                "c": WorkflowState(chorus_c, instruction="{input}"),
            },
            transitions=[
                WorkflowTransition("a", "b", guard=lambda r: "ok" in r),
                WorkflowTransition("a", "c", on_event="skip"),
            ],
            initial_state="a",
        )
        # Guard fires → goes to "b"
        wf.proxy().run("go").get()
        assert wf.proxy().current_state().get() == "b"
        # Reset to "a" and fire event instead
        wf.proxy().set_state("a").get()
        wf.proxy().event("skip").get()
        assert wf.proxy().current_state().get() == "c"
        chorus_a.stop()
        chorus_b.stop()
        chorus_c.stop()


# ---------------------------------------------------------------------------
# Runtime modification
# ---------------------------------------------------------------------------


class TestRuntimeModification:
    def test_add_state_while_running_and_transition_to_it(self, actor_factory):
        chorus_a, _ = make_chorus(["trigger"], "RTA")
        chorus_new, _ = make_chorus(["new state reply"], "RTNew")
        wf = actor_factory(
            Workflow,
            states={"a": WorkflowState(chorus_a, instruction="{input}")},
            initial_state="a",
        )
        wf.proxy().add_state("b", WorkflowState(chorus_new, instruction="{input}")).get()
        wf.proxy().add_transition(
            WorkflowTransition("a", "b", guard=lambda r: "trigger" in r)
        ).get()
        result = wf.proxy().run("go").get()
        assert result == chorus_reply("new state reply")
        assert wf.proxy().current_state().get() == "b"
        chorus_a.stop()
        chorus_new.stop()

    def test_remove_guard_transition_stops_auto_advance(self, actor_factory):
        chorus_a, _ = make_chorus(["trigger"], "RemA")
        chorus_b, _ = make_chorus(["b reply"], "RemB")
        wf = actor_factory(
            Workflow,
            states={
                "a": WorkflowState(chorus_a, instruction="{input}"),
                "b": WorkflowState(chorus_b, instruction="{input}"),
            },
            transitions=[WorkflowTransition("a", "b", guard=lambda r: True)],
            initial_state="a",
        )
        wf.proxy().remove_transitions("a", "b").get()
        wf.proxy().run("go").get()
        assert wf.proxy().current_state().get() == "a"
        chorus_a.stop()
        chorus_b.stop()

    def test_replace_state_chorus_at_runtime(self, actor_factory):
        chorus_old, _ = make_chorus(["old"], "Old")
        chorus_new, _ = make_chorus(["new"], "New")
        wf = actor_factory(
            Workflow,
            states={"s": WorkflowState(chorus_old, instruction="{input}")},
            initial_state="s",
        )
        wf.proxy().add_state("s", WorkflowState(chorus_new, instruction="{input}")).get()
        result = wf.proxy().run("go").get()
        assert result == chorus_reply("new")
        chorus_old.stop()
        chorus_new.stop()


# ---------------------------------------------------------------------------
# Parallel actors within a state (dict[str, ActorRef])
# ---------------------------------------------------------------------------


class TestParallelActors:
    def _make_actor(self, replies: list[str], name: str) -> tuple:
        prov = FakeProvider(replies)
        cls = type(name, (AIActor,), {"provider": prov})
        return cls.start(), prov

    def test_parallel_state_fires_all_actors(self, actor_factory):
        ref_a, prov_a = self._make_actor(["reply_a"], "PA1")
        ref_b, prov_b = self._make_actor(["reply_b"], "PB1")
        wf = actor_factory(
            Workflow,
            states={
                "s": WorkflowState(
                    chorus={"a": ref_a, "b": ref_b},
                    instruction="{input}",
                )
            },
            initial_state="s",
        )
        result = wf.proxy().run("ping").get()
        assert "a: reply_a" in result
        assert "b: reply_b" in result
        ref_a.stop()
        ref_b.stop()

    def test_parallel_state_all_receive_same_instruction(self, actor_factory):
        ref_a, prov_a = self._make_actor(["ra"], "PA2")
        ref_b, prov_b = self._make_actor(["rb"], "PB2")
        wf = actor_factory(
            Workflow,
            states={"s": WorkflowState(chorus={"a": ref_a, "b": ref_b}, instruction="Q:{input}")},
            initial_state="s",
        )
        wf.proxy().run("hello").get()
        assert prov_a.calls[0]["messages"][-1]["content"] == "Q:hello"
        assert prov_b.calls[0]["messages"][-1]["content"] == "Q:hello"
        ref_a.stop()
        ref_b.stop()

    def test_parallel_state_guard_receives_combined_output(self, actor_factory):
        ref_a, _ = self._make_actor(["approved"], "PA3")
        ref_b, _ = self._make_actor(["done"], "PB3")
        ref_next, _ = self._make_actor(["next"], "PN3")
        wf = actor_factory(
            Workflow,
            states={
                "parallel": WorkflowState(chorus={"a": ref_a, "b": ref_b}, instruction="{input}"),
                "next": WorkflowState(chorus=ref_next, instruction="{input}"),
            },
            transitions=[WorkflowTransition("parallel", "next", guard=lambda r: "approved" in r)],
            initial_state="parallel",
        )
        result = wf.proxy().run("go").get()
        assert result == "next"  # plain AIActor ref, not wrapped in a Chorus
        assert wf.proxy().current_state().get() == "next"
        ref_a.stop()
        ref_b.stop()
        ref_next.stop()

    def test_parallel_state_single_actor_dict_works(self, actor_factory):
        ref_a, _ = self._make_actor(["only"], "PA4")
        wf = actor_factory(
            Workflow,
            states={"s": WorkflowState(chorus={"solo": ref_a}, instruction="{input}")},
            initial_state="s",
        )
        result = wf.proxy().run("hi").get()
        assert result == "solo: only"
        ref_a.stop()

    def test_mixed_single_and_parallel_states(self, actor_factory):
        single_ref, _ = make_chorus(["step1"], "Single")
        ref_a, _ = self._make_actor(["pa"], "MPA")
        ref_b, _ = self._make_actor(["pb"], "MPB")
        wf = actor_factory(
            Workflow,
            states={
                "single": WorkflowState(chorus=single_ref, instruction="{input}"),
                "parallel": WorkflowState(chorus={"a": ref_a, "b": ref_b}, instruction="{input}"),
            },
            transitions=[WorkflowTransition("single", "parallel", guard=lambda r: True)],
            initial_state="single",
        )
        result = wf.proxy().run("start").get()
        assert "a: pa" in result
        assert "b: pb" in result
        single_ref.stop()
        ref_a.stop()
        ref_b.stop()


# ---------------------------------------------------------------------------
# run_detached() — non-blocking background execution
# ---------------------------------------------------------------------------


class TestRunDetached:
    def test_run_detached_completes_and_calls_on_complete(self, actor_factory):
        # Standard library imports:
        import threading

        chorus_ref, _ = make_chorus(["done"])
        wf = actor_factory(
            Workflow,
            states={"s": WorkflowState(chorus_ref, instruction="{input}")},
            initial_state="s",
        )
        results = []
        done = threading.Event()

        def _cb(output):
            results.append(output)
            done.set()

        wf.proxy().run_detached("go", on_complete=_cb).get()
        assert done.wait(timeout=5)
        assert results == [chorus_reply("done")]
        chorus_ref.stop()

    def test_run_detached_returns_immediately(self, actor_factory):
        # Standard library imports:
        import time

        chorus_ref, _ = make_chorus(["slow"])
        wf = actor_factory(
            Workflow,
            states={"s": WorkflowState(chorus_ref, instruction="{input}")},
            initial_state="s",
        )
        t0 = time.monotonic()
        wf.proxy().run_detached("go").get()  # should return very fast
        elapsed = time.monotonic() - t0
        assert elapsed < 2.0
        chorus_ref.stop()

    def test_run_detached_event_can_fire_during_run(self, actor_factory):
        # Standard library imports:
        import threading

        chorus_a, _ = make_chorus(["a_reply"], "DetA")
        chorus_b, _ = make_chorus(["b_reply"], "DetB")
        wf = actor_factory(
            Workflow,
            states={
                "a": WorkflowState(chorus_a, instruction="{input}"),
                "b": WorkflowState(chorus_b, instruction="{input}"),
            },
            transitions=[WorkflowTransition("a", "b", on_event="go")],
            initial_state="a",
        )
        done = threading.Event()
        wf.proxy().run_detached("ping", on_complete=lambda _: done.set()).get()
        # Fire event while run_detached is executing (actor mailbox is free)
        wf.proxy().event("go").get()
        done.wait(timeout=5)
        chorus_a.stop()
        chorus_b.stop()

    def test_run_detached_on_error_called_for_missing_state(self, actor_factory):
        # Standard library imports:
        import threading

        wf = actor_factory(Workflow, initial_state="ghost")
        errors = []
        done = threading.Event()

        def _err(exc):
            errors.append(exc)
            done.set()

        wf.proxy().run_detached("hi", on_error=_err).get()
        assert done.wait(timeout=5)
        assert errors
        assert isinstance(errors[0], KeyError)

    def test_run_detached_follows_guard_transition(self, actor_factory):
        # Standard library imports:
        import threading

        chorus_a, _ = make_chorus(["mid"], "DGA")
        chorus_b, _ = make_chorus(["final"], "DGB")
        wf = actor_factory(
            Workflow,
            states={
                "a": WorkflowState(chorus_a, instruction="{input}"),
                "b": WorkflowState(chorus_b, instruction="{input}"),
            },
            transitions=[WorkflowTransition("a", "b", guard=lambda r: "mid" in r)],
            initial_state="a",
        )
        results = []
        done = threading.Event()

        def _cb(output):
            results.append(output)
            done.set()

        wf.proxy().run_detached("go", on_complete=_cb).get()
        assert done.wait(timeout=5)
        assert results == [chorus_reply("final")]
        assert wf.proxy().current_state().get() == "b"
        chorus_a.stop()
        chorus_b.stop()

    def test_run_detached_no_on_complete_runs_silently(self, actor_factory):
        # Standard library imports:
        import time

        chorus_ref, _ = make_chorus(["ok"])
        wf = actor_factory(
            Workflow,
            states={"s": WorkflowState(chorus_ref, instruction="{input}")},
            initial_state="s",
        )
        wf.proxy().run_detached("go").get()
        time.sleep(0.3)  # give background thread time to finish
        assert wf.proxy().last_output().get() == chorus_reply("ok")
        chorus_ref.stop()

    def test_run_detached_no_current_state_exits_immediately(self, actor_factory):
        # Standard library imports:
        import threading

        # _current is None → prepare_step() returns None → background loop breaks
        # immediately and calls on_complete with the empty last_output.
        wf = actor_factory(Workflow)  # no initial_state
        done = threading.Event()
        results = []
        wf.proxy().run_detached(
            "any",
            on_complete=lambda out: (results.append(out), done.set()),
        ).get()
        assert done.wait(timeout=5)
        assert results == [""]

    def test_run_detached_exception_silently_swallowed_without_on_error(self, actor_factory):
        # Standard library imports:
        import time

        # Missing state raises KeyError inside the loop; with no on_error callback
        # the exception is caught and silently discarded.
        wf = actor_factory(Workflow, initial_state="ghost")
        wf.proxy().run_detached("trigger").get()  # no on_error
        time.sleep(0.3)  # give the background thread time to finish
        # no exception propagated — test passes if we reach this line

    def test_prepare_step_returns_none_when_no_current_state(self, actor_factory):
        wf = actor_factory(Workflow)
        result = wf.proxy().prepare_step("anything").get()
        assert result is None

    def test_commit_step_returns_false_when_no_current_state(self, actor_factory):
        wf = actor_factory(Workflow)
        result = wf.proxy().commit_step("output").get()
        assert result is False

    def test_commit_step_advances_state_on_guard_match(self, actor_factory):
        chorus_a, _ = make_chorus(["r"], "CSA")
        chorus_b, _ = make_chorus(["r"], "CSB")
        wf = actor_factory(
            Workflow,
            states={"a": WorkflowState(chorus_a), "b": WorkflowState(chorus_b)},
            transitions=[WorkflowTransition("a", "b", guard=lambda r: True)],
            initial_state="a",
        )
        advanced = wf.proxy().commit_step("anything").get()
        assert advanced is True
        assert wf.proxy().current_state().get() == "b"
        chorus_a.stop()
        chorus_b.stop()
