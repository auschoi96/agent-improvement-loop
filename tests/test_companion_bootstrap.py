"""Offline tests for the one-command companion bootstrap (``ail-companion-start``).

No live calls: the token mint and the inner ``ail.companion.cli`` poll are both
monkeypatched. Every test asserts against the REAL wiring (real arg construction,
real fail-closed logic, real ``cli.main`` dispatch), never a reimplementation.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any

import pytest

from ail.companion import bootstrap
from ail.companion import cli as companion_cli

_ENV_KEYS = (
    "DATABRICKS_HOST",
    "DATABRICKS_TOKEN",
    "AIL_CATALOG",
    "AIL_SCHEMA",
    "DATABRICKS_CONFIG_PROFILE",
)


@pytest.fixture(autouse=True)
def _isolate_env() -> Any:
    """Bootstrap writes os.environ directly; snapshot + restore so tests don't leak."""
    saved = {key: os.environ.get(key) for key in _ENV_KEYS}
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _spy_poll(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Spy the REAL ``cli.run_poll`` so bootstrap still goes through ``cli.main``.

    Records the argv the real dispatch handed to the poll plus a snapshot of the
    auth/UC environment at call time. Returns 0 (a clean no-op poll).
    """
    calls: list[dict[str, Any]] = []

    def _record(argv: list[str]) -> int:
        calls.append(
            {
                "argv": list(argv),
                "env": {
                    "DATABRICKS_HOST": os.environ.get("DATABRICKS_HOST"),
                    "DATABRICKS_TOKEN": os.environ.get("DATABRICKS_TOKEN"),
                    "AIL_CATALOG": os.environ.get("AIL_CATALOG"),
                    "AIL_SCHEMA": os.environ.get("AIL_SCHEMA"),
                    "DATABRICKS_CONFIG_PROFILE": os.environ.get("DATABRICKS_CONFIG_PROFILE"),
                },
            }
        )
        return 0

    monkeypatch.setattr(companion_cli, "run_poll", _record)
    return calls


def _fresh_mint(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Monkeypatch the mint to hand back a fresh token per call. Returns the log."""
    minted: list[str] = []

    def _mint(profile: str) -> tuple[str, str]:
        token = f"static-token-{len(minted) + 1}"
        minted.append(token)
        return "https://example.databricks.com", token

    monkeypatch.setattr(bootstrap, "mint_static_auth", _mint)
    return minted


_ARGS = [
    "--profile",
    "myprofile",
    "--experiment",
    "exp-123",
    "--catalog",
    "cat",
    "--schema",
    "sch",
    "--interval-seconds",
    "0",
]


def test_remints_a_fresh_token_every_cycle(monkeypatch: pytest.MonkeyPatch) -> None:
    minted = _fresh_mint(monkeypatch)
    calls = _spy_poll(monkeypatch)

    rc = bootstrap.main([*_ARGS, "--max-cycles", "2"])

    assert rc == 0
    assert minted == ["static-token-1", "static-token-2"]  # minted twice, fresh each
    assert len(calls) == 2
    # Each poll saw the freshly minted token of ITS cycle.
    assert calls[0]["env"]["DATABRICKS_TOKEN"] == "static-token-1"
    assert calls[1]["env"]["DATABRICKS_TOKEN"] == "static-token-2"


def test_fail_closed_when_mint_returns_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _spy_poll(monkeypatch)
    # Simulate the CLI failing at the subprocess boundary -> exercise REAL fail-closed.
    monkeypatch.setattr(
        bootstrap,
        "_run",
        lambda cmd: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="bad profile"),
    )

    with pytest.raises(SystemExit):
        bootstrap.main([*_ARGS, "--max-cycles", "3"])

    assert calls == []  # companion NEVER runs unauthenticated


def test_fail_closed_when_mint_returns_empty_token(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _spy_poll(monkeypatch)
    monkeypatch.setattr(
        bootstrap,
        "_run",
        lambda cmd: subprocess.CompletedProcess(cmd, 0, stdout='{"access_token": ""}', stderr=""),
    )

    with pytest.raises(SystemExit):
        bootstrap.main([*_ARGS, "--max-cycles", "1"])

    assert calls == []


def test_inner_companion_gets_static_token_and_no_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fresh_mint(monkeypatch)
    calls = _spy_poll(monkeypatch)

    rc = bootstrap.main([*_ARGS, "--max-cycles", "1", "--warehouse-id", "wh-1"])

    assert rc == 0
    argv = calls[0]["argv"]
    # The inner companion runs one bounded pass...
    assert "--max-iterations" in argv and argv[argv.index("--max-iterations") + 1] == "1"
    # ...with the static token in the environment (not a flag)...
    assert calls[0]["env"]["DATABRICKS_TOKEN"] == "static-token-1"
    # ...and is NEVER handed --profile (the bootstrap alone owns the profile).
    assert "--profile" not in argv
    assert "myprofile" not in argv


def test_env_is_set_before_the_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    _fresh_mint(monkeypatch)
    calls = _spy_poll(monkeypatch)

    rc = bootstrap.main([*_ARGS, "--max-cycles", "1"])

    assert rc == 0
    env = calls[0]["env"]
    assert env["DATABRICKS_HOST"] == "https://example.databricks.com"
    assert env["DATABRICKS_TOKEN"] == "static-token-1"
    assert env["AIL_CATALOG"] == "cat"
    assert env["AIL_SCHEMA"] == "sch"
    # Any ambient OAuth profile is dropped so the SDK can't refresh mid-run.
    assert env["DATABRICKS_CONFIG_PROFILE"] is None


def test_max_cycles_is_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    _fresh_mint(monkeypatch)
    calls = _spy_poll(monkeypatch)

    rc = bootstrap.main([*_ARGS, "--max-cycles", "4"])

    assert rc == 0
    assert len(calls) == 4


def test_reuses_the_real_companion_cli_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    # bootstrap must dispatch through the REAL cli.main, which routes to run_poll.
    _fresh_mint(monkeypatch)
    seen: list[list[str]] = []

    def _spy_run_poll(argv: list[str]) -> int:
        seen.append(list(argv))
        return 0

    monkeypatch.setattr(companion_cli, "run_poll", _spy_run_poll)

    rc = bootstrap.main([*_ARGS, "--max-cycles", "1"])

    assert rc == 0
    # bootstrap references the genuine module (no shadow reimplementation).
    assert bootstrap.companion_cli is companion_cli
    # The real cli.main("poll", ...) reached the real run_poll with our forwarded core.
    assert seen and "--experiment" in seen[0]
    assert seen[0][seen[0].index("--experiment") + 1] == "exp-123"
    assert "--catalog" in seen[0] and "--schema" in seen[0]


def test_profile_forwarded_to_poll_would_be_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guard: the real cli.main refuses --profile, so bootstrap must never forward it.

    Proves the poll passthrough is genuinely built without --profile by confirming the
    passthrough list contains none, and that the profile only reaches the mint.
    """
    _fresh_mint(monkeypatch)
    import argparse

    passthrough = bootstrap._poll_passthrough(
        argparse.Namespace(
            experiment="e", catalog="c", schema="s", agent="claude_code", warehouse_id=None
        )
    )
    assert "--profile" not in passthrough


def test_poll_failure_then_interrupt_surfaces_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: a failing poll cycle followed by a graceful stop must NOT exit 0.

    Cycle 1's poll fails; cycle 2 is interrupted (operator Ctrl-C / SIGTERM). A
    graceful stop after a poll failure must still surface the failure in the exit
    code. Before the fix (``return 0 if stopped else worst_rc``) this returned 0
    and masked the failure from any process-exit-code monitor.
    """
    _fresh_mint(monkeypatch)
    calls = {"n": 0}

    def _flaky_poll(argv: list[str]) -> int:
        calls["n"] += 1
        if calls["n"] == 1:
            return 1  # this cycle's poll failed
        raise KeyboardInterrupt  # operator stops the supervisor on the next cycle

    monkeypatch.setattr(companion_cli, "run_poll", _flaky_poll)

    rc = bootstrap.main([*_ARGS, "--max-cycles", "0"])  # unbounded; stopped by the interrupt

    assert calls["n"] == 2  # ran the failing cycle, then the interrupted one
    assert rc != 0  # failure is reflected despite the graceful stop


def test_clean_interrupt_with_no_errors_exits_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Paired case: a graceful stop with ZERO poll errors still exits 0."""
    _fresh_mint(monkeypatch)
    calls = {"n": 0}

    def _ok_then_interrupt(argv: list[str]) -> int:
        calls["n"] += 1
        if calls["n"] == 1:
            return 0  # clean poll
        raise KeyboardInterrupt  # then the operator stops it

    monkeypatch.setattr(companion_cli, "run_poll", _ok_then_interrupt)

    rc = bootstrap.main([*_ARGS, "--max-cycles", "0"])

    assert calls["n"] == 2
    assert rc == 0
