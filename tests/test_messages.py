"""Tests for message dataclasses (Instruct, Remember, Forget) and top-level exports."""

# Future imports (must occur at the beginning of the file):
from __future__ import annotations

# Third party imports:
import pytest

# Local imports:
from actor_ai.messages import Forget, Instruct, Remember


class TestInstruct:
    def test_instruction_is_required(self):
        with pytest.raises(TypeError):
            Instruct()  # type: ignore[call-arg]

    def test_default_history_is_empty_list(self):
        msg = Instruct("hello")
        assert msg.history == []

    def test_default_use_session_is_true(self):
        msg = Instruct("hello")
        assert msg.use_session is True

    def test_all_fields_set(self):
        history = [{"role": "user", "content": "prior"}]
        msg = Instruct("hi", history=history, use_session=False)
        assert msg.instruction == "hi"
        assert msg.history == history
        assert msg.use_session is False

    def test_history_default_is_independent_per_instance(self):
        a = Instruct("a")
        b = Instruct("b")
        a.history.append({"role": "user", "content": "x"})
        assert b.history == []

    def test_equality(self):
        assert Instruct("hello") == Instruct("hello")
        assert Instruct("hello") != Instruct("world")

    def test_use_session_false(self):
        msg = Instruct("hi", use_session=False)
        assert msg.use_session is False

    def test_instruction_stored_exactly(self):
        text = "   some instruction with spaces   "
        msg = Instruct(text)
        assert msg.instruction == text


class TestRemember:
    def test_requires_key_and_value(self):
        with pytest.raises(TypeError):
            Remember()  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            Remember("key")  # type: ignore[call-arg]

    def test_fields(self):
        msg = Remember("name", "Alice")
        assert msg.key == "name"
        assert msg.value == "Alice"

    def test_equality(self):
        assert Remember("k", "v") == Remember("k", "v")
        assert Remember("k", "v") != Remember("k", "other")
        assert Remember("k", "v") != Remember("other", "v")


class TestForget:
    def test_requires_key(self):
        with pytest.raises(TypeError):
            Forget()  # type: ignore[call-arg]

    def test_field(self):
        msg = Forget("name")
        assert msg.key == "name"

    def test_equality(self):
        assert Forget("k") == Forget("k")
        assert Forget("k") != Forget("other")


class TestPackageExports:
    def test_main_runs_without_error(self, capsys):
        # Local imports:
        from actor_ai import main

        main()
        out = capsys.readouterr().out
        assert "actor-ai" in out

    def test_all_public_symbols_importable(self):
        # Local imports:
        import actor_ai

        for name in actor_ai.__all__:
            assert hasattr(actor_ai, name), f"{name} missing from actor_ai"
