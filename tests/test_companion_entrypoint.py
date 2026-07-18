"""Offline tests for the consolidated ``python -m ail.companion`` entrypoint."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import pytest

from ail.companion import cli as companion


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
    def __init__(self, *, n_block: int = 0, n_errored: int = 0) -> None:
        self.n_tasks = 1
        self.n_promote = 0 if n_block or n_errored else 1
        self.n_block = n_block
        self.n_errored = n_errored

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


def test_prove_ignores_ambient_warehouse_without_experiment_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, Any]] = []
    _auth_ok(monkeypatch)
    monkeypatch.setenv("AIL_WAREHOUSE_ID", "ambient-wh")
    monkeypatch.setattr(companion, "load_task_suite", lambda version, root=None: "suite")
    monkeypatch.setattr(companion, "_build_adapter", lambda args: "adapter")
    monkeypatch.setattr(
        companion,
        "run_phase2_comparison",
        lambda **kwargs: calls.append(kwargs) or _Artifact(),
    )

    code = companion.main(
        [
            "prove",
            "--suite-version",
            "phase2-mini",
            "--output",
            str(tmp_path / "phase2.json"),
            "--host",
            "https://example.databricks.com",
        ]
    )

    assert code == 0
    assert calls[0]["warehouse_id"] is None


def test_prove_returns_nonzero_for_blocked_or_errored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _auth_ok(monkeypatch)
    monkeypatch.setattr(companion, "load_task_suite", lambda version, root=None: "suite")
    monkeypatch.setattr(companion, "_build_adapter", lambda args: "adapter")
    monkeypatch.setattr(
        companion,
        "run_phase2_comparison",
        lambda **kwargs: _Artifact(n_block=1, n_errored=1),
    )

    code = companion.main(
        [
            "prove",
            "--suite-version",
            "phase2-mini",
            "--output",
            str(tmp_path / "phase2.json"),
            "--host",
            "https://example.databricks.com",
        ]
    )

    assert code == 2


def _poll_args(extra: list[str]) -> argparse.Namespace:
    planning = [] if "--plan-every" in extra else ["--plan-every", "0"]
    return companion._poll_parser().parse_args(
        ["--warehouse-id", "wh", "--volume-root", "/Volumes/c/s/v", *planning] + extra
    )


def test_poll_planning_is_enabled_by_default() -> None:
    args = companion._poll_parser().parse_args(["--warehouse-id", "wh"])
    assert args.plan_every == 0


def test_executor_argv_registry_is_optional() -> None:
    # Default: no --registry forwarded -> the executor resolves from the UC agent_registry.
    argv_default = companion._executor_argv(_poll_args([]))
    assert "--registry" not in argv_default
    # Explicit local-dev override is forwarded through.
    argv_set = companion._executor_argv(_poll_args(["--registry", "config/agents.yaml"]))
    assert argv_set[argv_set.index("--registry") + 1] == "config/agents.yaml"


def test_planner_argv_experiment_is_optional_and_threads_uc_connection() -> None:
    # Default: no --experiment forwarded -> the planner resolves it from the UC registry,
    # and the UC connection args ARE threaded so it can read the registry.
    argv_default = companion._planner_argv(_poll_args([]))
    assert "--experiment" not in argv_default
    assert argv_default[argv_default.index("--warehouse-id") + 1] == "wh"
    assert "--catalog" in argv_default and "--schema" in argv_default
    # Explicit experiment override is forwarded through.
    argv_set = companion._planner_argv(_poll_args(["--experiment", "exp-1"]))
    assert argv_set[argv_set.index("--experiment") + 1] == "exp-1"


def test_poll_plan_every_without_experiment_defers_to_planner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Previously --plan-every without --experiment raised at argv construction; now it
    # defers to the planner (registry-driven), which resolves or fails closed at runtime.
    _auth_ok(monkeypatch)
    monkeypatch.setattr(companion.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(companion.agent_executor, "run", lambda args: 0)
    planner_calls: list[argparse.Namespace] = []
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
            "1",
            "--interval-seconds",
            "0",
            "--plan-every",
            "1",
        ]
    )

    assert code == 0
    assert len(planner_calls) == 1
    assert planner_calls[0].experiment is None  # no explicit experiment -> registry-driven


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


def test_poll_verify_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tier-2 verify is opt-in: with no --verify-every, the handler never runs."""
    _auth_ok(monkeypatch)
    monkeypatch.setattr(companion.agent_executor, "run", lambda args: 0)
    verify_calls: list[Any] = []
    monkeypatch.setattr(companion, "_run_verify_once", lambda *a, **k: verify_calls.append(k))
    # A client build would signal the verify path engaged — it must NOT be built.
    monkeypatch.setattr(
        companion,
        "_build_workspace_client",
        lambda profile: (_ for _ in ()).throw(AssertionError("verify must be disabled")),
    )

    code = companion.main(
        [
            "poll",
            "--warehouse-id",
            "wh",
            "--volume-root",
            "/Volumes/c/s/v",
            "--max-iterations",
            "2",
            "--plan-every",
            "0",
        ]
    )

    assert code == 0
    assert verify_calls == []


def test_verify_tick_uses_live_registry_experiment(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    args = argparse.Namespace(
        agent="claude_code",
        registry=None,
        experiment=None,
        catalog="cat",
        schema="sch",
    )
    monkeypatch.setattr(
        companion.agent_executor,
        "_resolve_agent",
        lambda *a, **k: argparse.Namespace(experiment_id="exp-live"),
    )

    def _select(**kwargs: Any) -> list[dict[str, Any]]:
        calls.append(kwargs)
        return []

    monkeypatch.setattr(companion, "select_pending_verify_requests", _select)

    companion._run_verify_once(args, client="client", warehouse_id="wh")

    assert calls[0]["experiment_id"] == "exp-live"


def test_poll_runs_verify_tick_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    _auth_ok(monkeypatch)
    monkeypatch.setattr(companion.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(companion.agent_executor, "run", lambda args: 0)
    monkeypatch.setattr(companion, "_build_workspace_client", lambda profile: object())
    verify_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(companion, "_run_verify_once", lambda args, **k: verify_calls.append(k))

    code = companion.main(
        [
            "poll",
            "--warehouse-id",
            "wh",
            "--volume-root",
            "/Volumes/c/s/v",
            "--max-iterations",
            "2",
            "--interval-seconds",
            "0",
            "--verify-every",
            "1",
            "--plan-every",
            "0",
        ]
    )

    assert code == 0
    assert len(verify_calls) == 2
    assert verify_calls[0]["warehouse_id"] == "wh"


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
            "--plan-every",
            "0",
        ]
    )

    assert code == 0
    assert calls == 1


def test_poll_real_executor_arg_construction_reaches_real_run_access(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DATABRICKS_HOST", "https://example.databricks.com")
    monkeypatch.setenv("DATABRICKS_TOKEN", "token")
    registry = tmp_path / "agents.yaml"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry.write_text(
        "\n".join(
            [
                "agents:",
                "  - agent_name: claude_code",
                "    experiment_id: exp",
                f"    target_workspace: {workspace}",
            ]
        ),
        encoding="utf-8",
    )
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        companion.agent_executor, "_build_workspace_client", lambda profile: object()
    )
    monkeypatch.setattr(companion.agent_executor, "_build_volume_client", lambda profile: object())

    def _list(client: object, warehouse_id: str, **kwargs: Any) -> list[Any]:
        calls.append((warehouse_id, kwargs["status"].value))
        return []

    monkeypatch.setattr(companion.agent_executor, "list_agent_task_proposals", _list)

    code = companion.main(
        [
            "poll",
            "--registry",
            str(registry),
            "--warehouse-id",
            "wh",
            "--volume-root",
            "/Volumes/c/s/v",
            "--max-iterations",
            "1",
            "--plan-every",
            "0",
        ]
    )

    assert code == 0
    assert calls == [("wh", "pending"), ("wh", "approved")]


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
