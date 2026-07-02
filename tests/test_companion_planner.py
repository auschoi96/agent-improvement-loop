"""Companion-planner entrypoint tests — static auth + fail-closed run, no live calls.

The ``ail-companion-planner`` entrypoint (:mod:`ail.jobs.companion_planner`) wires the
evidence-first cycle to real feedback/gate/publish seams. These tests fake those seams
(no live MLflow / warehouse) and pin the entrypoint's own contracts:

* **Static auth, no refreshing OAuth**: it refuses to run without a static token and
  drops any ambient ``DATABRICKS_CONFIG_PROFILE`` (the hard-won lesson).
* **Fail-closed on unreadable evidence**: a feedback-read failure returns non-zero and
  publishes **nothing** (never clears the agent's slice on an unknown state).
* **Dry-run publishes nothing**; a real run publishes the cycle's PENDING proposals.
"""

from __future__ import annotations

import pytest

from ail.jobs import companion_planner as cp
from ail.jobs import optimization_cycle as oc
from ail.loop.decision_rules import FeedbackBundle, RedundantReadSignal
from ail.readiness.contract import (
    EvalHealth,
    Gate,
    GateName,
    ReadinessStatus,
    ReadinessTier,
)


def _args(argv_extra: list[str] | None = None):  # type: ignore[no-untyped-def]
    base = [
        "--agent",
        "claude_code",
        "--experiment",
        "660599403165942",
        "--warehouse-id",
        "wh1",
        "--host",
        "https://example.databricks.com",
        "--goal-confirmed",
        "true",
    ]
    return cp._parse_args(base + (argv_extra or []))


def _ready() -> ReadinessStatus:
    return ReadinessStatus(
        cohort_name="claude_code",
        objective_metric="total_tokens",
        trace_count=80,
        tier=ReadinessTier.READY_TO_PROVE,
        gates=[Gate(name=GateName.TRACE_PROVE, passed=True, reason="enough traces")],
        reasons=[],
        eval_health=EvalHealth(cohort_name="claude_code", scored_coverage=0.9),
    )


def _redundant_bundle() -> FeedbackBundle:
    return FeedbackBundle(
        objective_metric_value=900.0,
        objective_baseline_value=1000.0,
        redundant_reads=(
            RedundantReadSignal(
                tool="Read",
                repeated_target="/x",
                occurrences=5,
                dominant=True,
                estimated_wasted_tokens=4200,
                trace_ids=("t1", "t2"),
            ),
        ),
    )


# -- static auth -----------------------------------------------------------


def test_resolve_static_auth_refuses_without_static_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "some-oauth-profile")
    args = _args()  # carries --host but no token / no secret scope
    with pytest.raises(SystemExit, match="STATIC Databricks token"):
        cp.resolve_static_auth(args)
    # even when it refuses, the profile is dropped (never fall back to refreshing OAuth)
    import os

    assert "DATABRICKS_CONFIG_PROFILE" not in os.environ


def test_resolve_static_auth_uses_env_token_and_drops_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABRICKS_TOKEN", "static-pat")
    monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "some-oauth-profile")
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    args = _args()
    path = cp.resolve_static_auth(args)
    import os

    assert path == "env"  # static token already present -> no minting, no refresh
    assert os.environ["DATABRICKS_HOST"] == "https://example.databricks.com"
    assert "DATABRICKS_CONFIG_PROFILE" not in os.environ


# -- fail-closed run: unreadable evidence ----------------------------------


def test_run_unreadable_feedback_returns_nonzero_and_never_publishes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _broken_source(agent, args):  # type: ignore[no-untyped-def]
        def _src() -> FeedbackBundle:
            raise RuntimeError("trace store unreachable")

        return _src

    published: list[object] = []

    def _no_publish(proposals, *, agent, args):  # type: ignore[no-untyped-def]
        published.append(proposals)
        return len(proposals)

    monkeypatch.setattr(oc, "_default_feedback_source", _broken_source)
    monkeypatch.setattr(cp, "_publish", _no_publish)
    # gate must never be reached
    monkeypatch.setattr(
        oc, "_default_gate", lambda args: (_ for _ in ()).throw(AssertionError("gate reached"))
    )

    code = cp.run(_args())
    assert code == 2
    assert published == []  # nothing published: no slice cleared on an unknown state
    out = capsys.readouterr().out
    assert "could not read the agent's feedback" in out


# -- dry-run vs real publish -----------------------------------------------


def _wire_ok(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Wire a canned redundant-read feedback + ready gate; return the publish sink."""
    monkeypatch.setattr(oc, "_default_feedback_source", lambda agent, args: _redundant_bundle)
    monkeypatch.setattr(oc, "_default_gate", lambda args: lambda *, goal, agent: _ready())
    published: list[object] = []

    def _sink(proposals, *, agent, args):  # type: ignore[no-untyped-def]
        published.append(list(proposals))
        return len(proposals)

    monkeypatch.setattr(cp, "_publish", _sink)
    return published


def test_run_dry_run_surfaces_but_publishes_nothing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    published = _wire_ok(monkeypatch)
    code = cp.run(_args(["--dry-run"]))
    assert code == 0
    assert published == []  # dry-run: never writes
    out = capsys.readouterr().out
    assert "EVIDENCE READ" in out
    assert "PLAN (Lane A + Lane B)" in out
    assert "DRY-RUN" in out
    # the evidence-first proposal (token-efficiency skill) is surfaced as proof=NONE
    assert "proof=NONE(evidence-first)" in out


def test_run_real_publishes_the_pending_proposals(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    published = _wire_ok(monkeypatch)
    code = cp.run(_args())
    assert code == 0
    assert len(published) == 1
    proposals = published[0]
    assert len(proposals) == 1  # the redundant-read token-efficiency skill proposal
    p = proposals[0]
    assert p.proof is None  # evidence-first: published with no frozen-suite proof
    out = capsys.readouterr().out
    assert "PUBLISHED 1 row(s)" in out
