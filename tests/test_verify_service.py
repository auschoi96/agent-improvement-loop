"""Offline tests for the opt-in Tier-2 "verify on my suite" engine (L9).

Covers both halves — the request write-path (:func:`run_verify_request` / :func:`main`)
and the companion poll handler (:func:`run_verify_tick`) — against injected fakes, so
the load-bearing FAIL-CLOSED / EVIDENCE-ONLY invariants are exercised with no live
workspace:

* a failed / errored / no-suite prove writes an HONEST state, NEVER a fabricated proof;
* the proof is evidence only — no write here ever touches a proposal's ``status``;
* a non-provable action kind and an anonymous requester are refused.
"""

from __future__ import annotations

import json
import types
from typing import Any

import pytest

from ail.loop import verify_service as vs
from ail.loop.proposals import ActionKind

# ---------------------------------------------------------------------------
# is_provable
# ---------------------------------------------------------------------------


def test_is_provable_only_suite_runnable_kinds() -> None:
    assert vs.is_provable(ActionKind.SKILL_UPDATE)
    assert vs.is_provable(ActionKind.INSTRUCTION_UPDATE)
    assert vs.is_provable(ActionKind.GEPA_PROMPT)
    assert not vs.is_provable(ActionKind.METRIC_VIEW)
    assert not vs.is_provable(ActionKind.REVERT)
    assert not vs.is_provable(ActionKind.AGENT_TASK)


# ---------------------------------------------------------------------------
# run_verify_request — fail-closed
# ---------------------------------------------------------------------------


def _patch_request_seams(
    monkeypatch: pytest.MonkeyPatch,
    *,
    proposal: Any,
    mark_calls: list[dict[str, Any]],
) -> None:
    monkeypatch.setattr(vs, "resolve_catalog_schema", lambda c, s: ("cat", "sch"))
    monkeypatch.setattr(vs, "_build_workspace_client", lambda profile: object())
    monkeypatch.setattr(
        vs,
        "_resolve_agent",
        lambda *args, **kwargs: types.SimpleNamespace(experiment_id="exp-subject"),
    )
    monkeypatch.setattr(vs, "load_pending_proposal", lambda **kwargs: proposal)
    monkeypatch.setattr(vs, "mark_verify_requested", lambda **kwargs: mark_calls.append(kwargs))


def test_request_refuses_anonymous_requester(monkeypatch: pytest.MonkeyPatch) -> None:
    result = vs.run_verify_request(
        proposal_id="p1", agent_name="a", requested_by="  ", requested_at="t", warehouse_id="wh"
    )
    assert result.outcome is vs.VerifyRequestOutcome.REFUSED
    assert "anonymous" in (result.refused_reason or "")


def test_request_errors_without_warehouse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABRICKS_WAREHOUSE_ID", raising=False)
    result = vs.run_verify_request(
        proposal_id="p1", agent_name="a", requested_by="r@x.com", requested_at="t"
    )
    assert result.outcome is vs.VerifyRequestOutcome.ERROR
    assert "warehouse" in (result.error or "")


def test_request_refuses_missing_proposal(monkeypatch: pytest.MonkeyPatch) -> None:
    mark_calls: list[dict[str, Any]] = []
    _patch_request_seams(monkeypatch, proposal=None, mark_calls=mark_calls)
    result = vs.run_verify_request(
        proposal_id="p1",
        agent_name="a",
        requested_by="r@x.com",
        requested_at="t",
        warehouse_id="wh",
    )
    assert result.outcome is vs.VerifyRequestOutcome.REFUSED
    assert "no pending proposal" in (result.refused_reason or "")
    assert mark_calls == []  # nothing was flagged


def test_request_refuses_non_provable_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    mark_calls: list[dict[str, Any]] = []
    proposal = types.SimpleNamespace(action_kind=ActionKind.METRIC_VIEW)
    _patch_request_seams(monkeypatch, proposal=proposal, mark_calls=mark_calls)
    result = vs.run_verify_request(
        proposal_id="p1",
        agent_name="a",
        requested_by="r@x.com",
        requested_at="t",
        warehouse_id="wh",
    )
    assert result.outcome is vs.VerifyRequestOutcome.REFUSED
    assert "cannot be proven" in (result.refused_reason or "")
    assert result.action_kind == "metric_view"
    assert mark_calls == []  # defence in depth — never flagged despite a client request


def test_request_flags_a_provable_pending_proposal(monkeypatch: pytest.MonkeyPatch) -> None:
    mark_calls: list[dict[str, Any]] = []
    load_calls: list[dict[str, Any]] = []
    proposal = types.SimpleNamespace(action_kind=ActionKind.SKILL_UPDATE)
    _patch_request_seams(monkeypatch, proposal=proposal, mark_calls=mark_calls)
    monkeypatch.setattr(
        vs,
        "load_pending_proposal",
        lambda **kwargs: load_calls.append(kwargs) or proposal,
    )
    result = vs.run_verify_request(
        proposal_id="p1",
        agent_name="claude_code",
        requested_by="r@x.com",
        requested_at="2026-07-03T00:00:00+00:00",
        warehouse_id="wh",
    )
    assert result.outcome is vs.VerifyRequestOutcome.REQUESTED
    assert result.verify_status == vs.VerifyStatus.REQUESTED.value
    assert load_calls[0]["experiment_id"] == "exp-subject"
    assert len(mark_calls) == 1
    assert mark_calls[0]["proposal_id"] == "p1"
    assert mark_calls[0]["experiment_id"] == "exp-subject"
    assert mark_calls[0]["requested_by"] == "r@x.com"


def test_request_infra_failure_is_honest_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vs, "resolve_catalog_schema", lambda c, s: ("cat", "sch"))

    def _boom(profile: Any) -> Any:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(vs, "_build_workspace_client", _boom)
    result = vs.run_verify_request(
        proposal_id="p1",
        agent_name="a",
        requested_by="r@x.com",
        requested_at="t",
        warehouse_id="wh",
    )
    assert result.outcome is vs.VerifyRequestOutcome.ERROR
    assert "kaboom" in (result.error or "")


# ---------------------------------------------------------------------------
# write_verify_result — evidence only (never touches status), fail-closed proof
# ---------------------------------------------------------------------------


def _capture_execute(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    statements: list[str] = []
    monkeypatch.setattr(vs, "_execute", lambda client, wh, sql: statements.append(sql))
    return statements


def test_select_requests_is_scoped_to_experiment(monkeypatch: pytest.MonkeyPatch) -> None:
    statements: list[str] = []

    def _query(client: Any, warehouse_id: str, statement: str) -> list[dict[str, Any]]:
        statements.append(statement)
        return []

    monkeypatch.setattr(vs, "_query_rows", _query)
    assert (
        vs.select_pending_verify_requests(
            client=object(),
            warehouse_id="wh",
            agent_name="a",
            experiment_id="exp-subject",
        )
        == []
    )
    assert "experiment_id = 'exp-subject'" in statements[0]


def test_write_result_none_proof_writes_all_null_and_never_sets_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statements = _capture_execute(monkeypatch)
    vs.write_verify_result(
        client=object(),
        warehouse_id="wh",
        agent_name="a",
        experiment_id="exp-subject",
        proposal_id="p1",
        proof=None,
        verify_status=vs.VerifyStatus.NO_SUITE,
        verify_error="no frozen suite configured",
        completed_at="2026-07-03T00:00:00+00:00",
        catalog="cat",
        schema="sch",
    )
    assert len(statements) == 1
    sql = statements[0]
    # A failed / no-suite prove leaves NO fabricated proof.
    assert "proof_proved_improvement = NULL" in sql
    assert "proof_realized_savings_absolute = NULL" in sql
    assert "verify_status = 'no_suite'" in sql
    # Evidence only: the write is scoped to the pending row and never flips approval.
    assert "status = 'pending'" in sql
    assert "experiment_id = 'exp-subject'" in sql
    set_clause = sql.split("SET", 1)[1].split("WHERE", 1)[0]
    assert "verify_status" in set_clause
    assert " status = " not in set_clause  # approval status is untouched


def test_write_result_real_proof_populates_proof_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    from ail.loop.proposals import ProofSummary

    statements = _capture_execute(monkeypatch)
    proof = ProofSummary(
        objective_metric="total_tokens",
        proved_improvement=True,
        correctness_held=True,
        realized_savings_absolute=1200.0,
        realized_savings_pct=30.0,
        n_promote=3,
        n_block=0,
        n_errored=0,
        suite_content_hash="abc",
        suite_version="v1-seed",
    )
    vs.write_verify_result(
        client=object(),
        warehouse_id="wh",
        agent_name="a",
        experiment_id="exp-subject",
        proposal_id="p1",
        proof=proof,
        verify_status=vs.VerifyStatus.VERIFIED,
        verify_error=None,
        completed_at="t",
    )
    sql = statements[0]
    assert "proof_proved_improvement = TRUE" in sql
    assert "proof_n_promote = 3" in sql
    assert "verify_status = 'verified'" in sql
    assert "verify_error = NULL" in sql


# ---------------------------------------------------------------------------
# run_verify_tick — the fail-closed / evidence-only matrix
# ---------------------------------------------------------------------------


def _fake_artifact(*, n_promote: int, n_block: int = 0, n_errored: int = 0) -> Any:
    return types.SimpleNamespace(
        outcomes=[],
        objective_metric="total_tokens",
        n_promote=n_promote,
        n_block=n_block,
        n_errored=n_errored,
        realized_token_savings_absolute=1000.0 if n_promote else 0.0,
        realized_token_savings_pct=25.0 if n_promote else None,
        suite_content_hash="hash",
        suite_version="v1-seed",
    )


def _run_tick(
    *,
    requested: list[dict[str, Any]],
    load_suite: Any,
    run_prover: Any,
) -> tuple[vs.VerifyTickSummary, list[dict[str, Any]]]:
    writes: list[dict[str, Any]] = []

    def _write(**kwargs: Any) -> None:
        writes.append(kwargs)

    summary = vs.run_verify_tick(
        agent_name="claude_code",
        select_requested=lambda: requested,
        load_suite=load_suite,
        run_prover=run_prover,
        write_result=_write,
        now=lambda: "2026-07-03T00:00:00+00:00",
    )
    return summary, writes


def test_tick_no_requests_is_clean_noop() -> None:
    summary, writes = _run_tick(
        requested=[],
        load_suite=lambda: (_ for _ in ()).throw(AssertionError("suite must not load")),
        run_prover=lambda suite: (_ for _ in ()).throw(AssertionError("prover must not run")),
    )
    assert summary.n_requested == 0
    assert writes == []


def test_tick_missing_suite_writes_honest_no_suite_never_a_proof() -> None:
    def _no_suite() -> Any:
        raise FileNotFoundError("no Task Suite artifact at /x")

    summary, writes = _run_tick(
        requested=[{"proposal_id": "p1"}, {"proposal_id": "p2"}],
        load_suite=_no_suite,
        run_prover=lambda suite: (_ for _ in ()).throw(AssertionError("prover must not run")),
    )
    assert summary.n_no_suite == 2
    assert {w["proposal_id"] for w in writes} == {"p1", "p2"}
    for w in writes:
        assert w["verify_status"] is vs.VerifyStatus.NO_SUITE
        assert w["proof"] is None  # never a fabricated proof
        assert "no frozen suite" in w["verify_error"]


def test_tick_prover_raises_writes_honest_errored_never_verified() -> None:
    def _boom(suite: Any) -> Any:
        raise RuntimeError("boom")

    summary, writes = _run_tick(
        requested=[{"proposal_id": "p1"}],
        load_suite=lambda: object(),
        run_prover=_boom,
    )
    assert summary.n_errored == 1
    assert writes[0]["verify_status"] is vs.VerifyStatus.ERRORED
    assert writes[0]["proof"] is None
    assert "boom" in writes[0]["verify_error"]


def test_tick_promote_writes_verified_with_the_real_proof() -> None:
    summary, writes = _run_tick(
        requested=[{"proposal_id": "p1"}],
        load_suite=lambda: object(),
        run_prover=lambda suite: _fake_artifact(n_promote=3),
    )
    assert summary.n_verified == 1
    assert writes[0]["verify_status"] is vs.VerifyStatus.VERIFIED
    assert writes[0]["proof"] is not None
    assert writes[0]["proof"].proved_improvement is True
    assert writes[0]["proof"].correctness_held is True


def test_tick_no_promote_is_honestly_blocked_not_verified() -> None:
    summary, writes = _run_tick(
        requested=[{"proposal_id": "p1"}],
        load_suite=lambda: object(),
        run_prover=lambda suite: _fake_artifact(n_promote=0, n_block=2),
    )
    assert summary.n_blocked == 1
    assert writes[0]["verify_status"] is vs.VerifyStatus.BLOCKED
    # The real proof numbers still land as evidence — a block is shown as a block.
    assert writes[0]["proof"] is not None
    assert writes[0]["proof"].proved_improvement is False


# ---------------------------------------------------------------------------
# main — the stdin/stdout CLI bridge
# ---------------------------------------------------------------------------


def test_main_prints_verify_result_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "sys.stdin",
        types.SimpleNamespace(
            read=lambda: json.dumps(
                {
                    "proposal_id": "p1",
                    "agent_name": "a",
                    "requested_by": "r@x.com",
                    "requested_at": "t",
                }
            )
        ),
    )
    monkeypatch.setattr(
        vs,
        "run_verify_request",
        lambda **kwargs: vs.VerifyRequestResult(
            outcome=vs.VerifyRequestOutcome.REQUESTED,
            proposal_id="p1",
            agent_name="a",
            requested_by="r@x.com",
            requested_at="t",
            verify_status="requested",
        ),
    )
    assert vs.main() == 0
    printed = json.loads(capsys.readouterr().out.strip())
    assert printed["outcome"] == "requested"
    assert printed["verify_status"] == "requested"


def test_main_unparseable_stdin_returns_nonzero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdin", types.SimpleNamespace(read=lambda: "not json {"))
    assert vs.main() == 2
    assert json.loads(capsys.readouterr().out.strip())["outcome"] == "error"
