from __future__ import annotations

import sys
import types

import pytest

from ail.sdk import ImprovementAgent, improve, load_callable, trace


def test_improve_wraps_callable_without_mlflow() -> None:
    agent = improve("demo", lambda prompt: prompt.upper())
    result = agent.run("hello")
    assert result.output == "HELLO"
    assert result.agent_name == "demo"
    assert result.trace_id is None or result.trace_id.startswith("tr-")
    assert result.duration_seconds >= 0


def test_improve_validates_inputs() -> None:
    with pytest.raises(ValueError):
        ImprovementAgent(" ", lambda: None)
    with pytest.raises(TypeError):
        ImprovementAgent("demo", "not callable")  # type: ignore[arg-type]


def test_trace_decorator_preserves_callable_shape() -> None:
    @trace(agent="decorated")
    def answer(prompt: str) -> str:
        return prompt[::-1]

    assert answer("abc") == "cba"
    assert answer.__name__ == "answer"


def test_load_callable() -> None:
    module = types.ModuleType("test_sdk_module")
    module.answer = lambda prompt: prompt.upper()
    sys.modules[module.__name__] = module
    assert load_callable("test_sdk_module:answer")("ok") == "OK"

    with pytest.raises(ValueError):
        load_callable("missing-separator")
