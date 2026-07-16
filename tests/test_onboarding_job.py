"""Offline tests for the deployed onboarding wheel-task adapter."""

from __future__ import annotations

import types

import pytest

from ail.compare.monitoring import TRACING_WAREHOUSE_ENV
from ail.jobs import onboarding_job
from ail.jobs.onboarding_job import _parse_args


def _required_args() -> list[str]:
    return [
        "--warehouse-id=wh",
        "--catalog=cat",
        "--schema=sch",
        "--trace-schema=trace_sch",
        "--goal-llm-endpoint=endpoint",
    ]


def test_parse_args_accepts_lakeflow_underscore_job_parameters() -> None:
    args = _parse_args(
        [
            "--request_id=req-1",
            *_required_args(),
        ]
    )

    assert args.request_id == "req-1"


def test_parse_args_keeps_hyphenated_task_parameters() -> None:
    args = _parse_args(
        [
            "--request-id=req-2",
            *_required_args(),
        ]
    )

    assert args.request_id == "req-2"


def test_main_exports_trusted_trace_location_before_running_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload: dict[str, object] = {
        "action": "create_experiment",
        "trace_catalog": "${var.catalog}",
        "trace_schema": "${var.trace_schema}",
    }
    seen: dict[str, str] = {}
    redacted: list[str] = []
    monkeypatch.delenv(TRACING_WAREHOUSE_ENV, raising=False)
    monkeypatch.setattr(onboarding_job, "resolve_job_auth", lambda **kwargs: None)
    monkeypatch.setattr(onboarding_job, "_read_payload", lambda **kwargs: payload)

    def _run_action(action: dict[str, object]) -> object:
        seen["warehouse"] = onboarding_job.os.environ[TRACING_WAREHOUSE_ENV]
        seen["trace_catalog"] = str(action["trace_catalog"])
        seen["trace_schema"] = str(action["trace_schema"])
        return types.SimpleNamespace(outcome="requirements", model_dump_json=lambda: "{}")

    monkeypatch.setattr(onboarding_job, "run_action", _run_action)
    monkeypatch.setattr(onboarding_job, "_persist_result", lambda **kwargs: None)
    monkeypatch.setattr(
        onboarding_job,
        "_redact_request",
        lambda **kwargs: redacted.append(str(kwargs["request_id"])),
    )

    assert (
        onboarding_job.main(
            [
                "--request_id=req-3",
                *_required_args(),
            ]
        )
        == 0
    )
    assert seen["warehouse"] == "wh"
    assert seen["trace_catalog"] == "cat"
    assert seen["trace_schema"] == "trace_sch"
    assert redacted == ["req-3"]


def test_main_redacts_payload_even_when_engine_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    redacted: list[str] = []
    monkeypatch.setattr(onboarding_job, "resolve_job_auth", lambda **kwargs: None)
    monkeypatch.setattr(
        onboarding_job,
        "_read_payload",
        lambda **kwargs: {"action": "create_experiment"},
    )
    monkeypatch.setattr(
        onboarding_job,
        "run_action",
        lambda payload: (_ for _ in ()).throw(RuntimeError("engine failed")),
    )
    monkeypatch.setattr(
        onboarding_job,
        "_redact_request",
        lambda **kwargs: redacted.append(str(kwargs["request_id"])),
    )

    with pytest.raises(RuntimeError, match="engine failed"):
        onboarding_job.main(["--request_id=req-failed", *_required_args()])

    assert redacted == ["req-failed"]


def test_main_redacts_request_even_when_payload_read_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redacted: list[str] = []
    monkeypatch.setattr(onboarding_job, "resolve_job_auth", lambda **kwargs: None)
    monkeypatch.setattr(
        onboarding_job,
        "_read_payload",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("request expired")),
    )
    monkeypatch.setattr(
        onboarding_job,
        "_redact_request",
        lambda **kwargs: redacted.append(str(kwargs["request_id"])),
    )

    with pytest.raises(RuntimeError, match="request expired"):
        onboarding_job.main(["--request_id=req-expired", *_required_args()])

    assert redacted == ["req-expired"]
