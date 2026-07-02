"""Offline tests for the consolidated ``python -m ail.companion`` entrypoint."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import pytest

from ail.companion import __main__ as companion


def _auth_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(companion, "_auth", lambda args: None)


def test_plan_dispatch_reuses_companion_planner(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[argparse.Namespace] = []
    parsed = argparse.Namespace(mode="plan")
    _auth_ok(monkeypatch)
    monkeypatch.setattr(companion.companion_planner, "_parse_args", lambda argv: parsed)
    monkeypatch.setattr(companion.companion_planner, "run", lambda args: seen.append(args) or 0)

    assert companion.main(["plan", "--experiment", "exp", "--warehouse-id", "wh"]) == 0
    assert seen == [parsed]


def test_execute_dispatch_reuses_agent_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[argparse.Namespace] = []
    parsed = argparse.Namespace(mode="execute")
    _auth_ok(monkeypatch)
    monkeypatch.setattr(companion.agent_executor, "_parse_args", lambda argv: parsed)
    monkeypatch.setattr(companion.agent_executor, "run", lambda args: seen.append(args) or 0)

    assert companion.main(["execute", "--warehouse-id", "wh", "--volume-root", "/Volumes/x"]) == 0
    assert seen == [parsed]


class _Artifact:
    n_tasks = 1
    n_promote = 1
    n_block = 0
    n_errored = 0

    def model_dump_json(self, *, indent: int) -> str:
        return f'{{"indent": {indent}}}'


def test_prove_dispatch_reuses_phase2_prover(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, Any]] = []
    _auth_ok(monkeypatch)
    monkeypatch.setattr(companion, "load_task_suite", lambda version, root=None: "suite")
    monkeypatch.setattr(companion, "_build_adapter", lambda args: "adapter")

    def _prove(**kwargs: Any) -> _Artifact:
        calls.append(kwargs)
        return _Artifact()

    monkeypatch.setattr(companion, "run_phase2_comparison", _prove)
    out = tmp_path / "phase2.json"

    code = companion.main(
        [
            "prove",
            "--suite-version",
            "phase2-mini",
            "--output",
            str(out),
            "--host",
            "https://example.databricks.com",
        ]
    )

    assert code == 0
    assert calls and calls[0]["suite"] == "suite"
    assert calls[0]["adapter"] == "adapter"
    assert calls[0]["profile"] is None
    assert out.read_text(encoding="utf-8") == '{"indent": 2}'


def test_poll_bounded_loop_runs_executor_and_planner_cadence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _auth_ok(monkeypatch)
    monkeypatch.setattr(companion.time, "sleep", lambda seconds: None)
    executor_calls: list[argparse.Namespace] = []
    planner_calls: list[argparse.Namespace] = []
    monkeypatch.setattr(
        companion.agent_executor, "run", lambda args: executor_calls.append(args) or 0
    )
    monkeypatch.setattr(
        companion.companion_planner, "run", lambda args: planner_calls.append(args) or 0
    )

    code = companion.main(
        [
            "poll",
            "--warehouse-id",
            "wh",
            "--volume-root",
            "/Volumes/c/s/v",
            "--max-iterations",
            "3",
            "--interval-seconds",
            "0",
            "--plan-every",
            "2",
            "--experiment",
            "exp",
        ]
    )

    assert code == 0
    assert len(executor_calls) == 3
    assert len(planner_calls) == 2
    assert executor_calls[0].revert is None
    assert planner_calls[0].experiment == "exp"


def test_poll_no_pending_work_is_clean_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    _auth_ok(monkeypatch)
    calls = 0

    def _executor_noop(args: argparse.Namespace) -> int:
        nonlocal calls
        calls += 1
        return 0

    monkeypatch.setattr(companion.agent_executor, "run", _executor_noop)

    code = companion.main(
        [
            "poll",
            "--warehouse-id",
            "wh",
            "--volume-root",
            "/Volumes/c/s/v",
            "--max-iterations",
            "1",
        ]
    )

    assert code == 0
    assert calls == 1


def test_missing_static_token_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "oauth-profile")

    with pytest.raises(SystemExit, match="STATIC Databricks token"):
        companion.main(
            [
                "execute",
                "--warehouse-id",
                "wh",
                "--volume-root",
                "/Volumes/c/s/v",
                "--host",
                "https://example.databricks.com",
            ]
        )
    assert "DATABRICKS_CONFIG_PROFILE" not in os.environ


def test_profile_oauth_is_refused() -> None:
    with pytest.raises(SystemExit, match="refusing --profile"):
        companion.main(["plan", "--profile", "oauth"])
