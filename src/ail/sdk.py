"""Small public facade for instrumenting and running any Python agent.

The facade deliberately delegates storage and tracing to MLflow. It does not
implement a second evaluation system; the Databricks improvement plane consumes
the resulting MLflow traces through the existing normalized trace contracts.
"""

from __future__ import annotations

import importlib
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import wraps
from typing import Any, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])


@dataclass(frozen=True, slots=True)
class RunResult:
    """Result returned by :class:`ImprovementAgent.run`."""

    output: Any
    trace_id: str | None
    duration_seconds: float
    agent_name: str


class _MlflowTracer:
    def __init__(self, *, name: str, experiment_id: str | None, tracking_uri: str | None) -> None:
        self.name = name
        self.experiment_id = experiment_id
        self.tracking_uri = tracking_uri
        self.mlflow: Any | None = None
        try:
            self.mlflow = importlib.import_module("mlflow")
            if tracking_uri:
                self.mlflow.set_tracking_uri(tracking_uri)
            if experiment_id:
                self.mlflow.set_experiment(experiment_id=experiment_id)
        except ImportError:
            self.mlflow = None

    def call(
        self,
        fn: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> tuple[Any, str | None]:
        if self.mlflow is None:
            return fn(*args, **kwargs), None

        start_span = getattr(self.mlflow, "start_span", None)
        if start_span is None:
            traced = self.mlflow.trace(name=self.name)(fn)
            return traced(*args, **kwargs), None

        with start_span(name=self.name) as span:
            _set_span_value(span, "set_inputs", {"args": args, "kwargs": dict(kwargs)})
            try:
                output = fn(*args, **kwargs)
                _set_span_value(span, "set_outputs", output)
                return output, _trace_id(span)
            except Exception as exc:
                _set_span_value(span, "record_exception", exc)
                raise


def _set_span_value(span: Any, method_name: str, value: Any) -> None:
    method = getattr(span, method_name, None)
    if method is None:
        return
    try:
        method(value)
    except (TypeError, ValueError):
        method(str(value))


def _trace_id(span: Any) -> str | None:
    for attr in ("trace_id", "request_id"):
        value = getattr(span, attr, None)
        if value:
            return str(value)
    return None


class ImprovementAgent:
    """A tiny wrapper that makes any callable observable by AIL.

    ``run`` accepts either a positional task or keyword arguments, so it works
    with simple LLM functions as well as richer custom agents. Evaluation,
    optimization, and promotion remain in the Databricks control plane.
    """

    def __init__(
        self,
        name: str,
        run: Callable[..., Any],
        *,
        objective: str | None = None,
        experiment_id: str | None = None,
        tracking_uri: str | None = None,
    ) -> None:
        if not name.strip():
            raise ValueError("name must not be empty")
        if not callable(run):
            raise TypeError("run must be callable")
        self.name = name.strip()
        self.objective = objective
        self._run = run
        self._tracer = _MlflowTracer(
            name=self.name,
            experiment_id=experiment_id,
            tracking_uri=tracking_uri,
        )

    def run(self, *args: Any, **kwargs: Any) -> RunResult:
        started = time.perf_counter()
        output, trace_id = self._tracer.call(self._run, args, kwargs)
        return RunResult(
            output=output,
            trace_id=trace_id,
            duration_seconds=time.perf_counter() - started,
            agent_name=self.name,
        )

    __call__ = run


def improve(
    name: str,
    run: Callable[..., Any],
    *,
    objective: str | None = None,
    experiment_id: str | None = None,
    tracking_uri: str | None = None,
) -> ImprovementAgent:
    """Wrap a callable for MLflow tracing and continuous improvement."""

    return ImprovementAgent(
        name,
        run,
        objective=objective,
        experiment_id=experiment_id,
        tracking_uri=tracking_uri,
    )


def trace(
    *,
    agent: str,
    experiment_id: str | None = None,
    tracking_uri: str | None = None,
) -> Callable[[F], F]:
    """Decorate any callable with the same AIL tracing facade."""

    def decorator(fn: F) -> F:
        wrapped_agent = improve(
            agent,
            fn,
            experiment_id=experiment_id,
            tracking_uri=tracking_uri,
        )

        @wraps(fn)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return wrapped_agent(*args, **kwargs).output

        return cast(F, wrapped)

    return decorator


def load_callable(spec: str) -> Callable[..., Any]:
    """Load ``module:function`` for the ``ail run`` command."""

    module_name, separator, function_name = spec.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("callable must use module:function syntax")
    function = getattr(importlib.import_module(module_name), function_name, None)
    if not callable(function):
        raise TypeError(f"{spec} is not callable")
    return function


__all__ = ["ImprovementAgent", "RunResult", "improve", "load_callable", "trace"]
