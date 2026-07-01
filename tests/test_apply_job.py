"""Unit tests for the Job-transport entrypoint (:mod:`ail.jobs.apply_job`).

Offline: auth resolution, the workspace client, the engine call, and the SQL
executor are all faked, so no live Databricks/warehouse access. Covers the thin
param-adapter contract:

* named parameters map straight onto :func:`ail.loop.apply_service.run_decision`
  (empty ``--reason`` collapses to ``None``);
* a malformed decision produces a fail-closed ``REFUSED`` result recorded to the
  handoff table (an honest outcome, never a fake apply), and ``main`` still exits 0;
* a missing ``--warehouse-id`` fails fast;
* the engine already REFUSES a non-pending proposal, so a duplicated/retried trigger
  on an already-applied proposal never re-applies; and
* the result-table writer emits the expected DDL + INSERT with the verbatim result
  JSON the app bridge reads back.
"""

from __future__ import annotations

from typing import Any

import pytest

from ail.jobs import apply_job
from ail.jobs.apply_job import APPLY_RESULTS_TABLE, main, write_apply_result
from ail.loop import apply_service
from ail.loop.apply import DecisionKind
from ail.loop.apply_service import ApplyServiceOutcome, ApplyServiceResult, run_decision

DECIDED_AT = "2026-06-30T12:00:00+00:00"


def _result(
    outcome: ApplyServiceOutcome = ApplyServiceOutcome.APPLIED,
    *,
    proposal_id: str = "prop-1",
    agent_name: str = "claude_code",
    decision: DecisionKind = DecisionKind.APPROVE,
    approver: str = "reviewer@databricks.com",
) -> ApplyServiceResult:
    return ApplyServiceResult(
        outcome=outcome,
        proposal_id=proposal_id,
        agent_name=agent_name,
        decision=decision,
        approver=approver,
        decided_at=DECIDED_AT,
        created_view="cat.sch.mv" if outcome is ApplyServiceOutcome.APPLIED else None,
    )


# -- main: named params -> run_decision + result write ----------------------


def test_main_wires_named_params_into_run_decision(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    written: dict[str, Any] = {}

    monkeypatch.setattr(apply_job, "resolve_job_auth", lambda **_: "minted")
    monkeypatch.setattr(apply_job, "_build_workspace_client", lambda _p: object())

    def fake_run_decision(**kwargs: Any) -> ApplyServiceResult:
        captured.update(kwargs)
        return _result()

    def fake_write(result: ApplyServiceResult, **kwargs: Any) -> None:
        written["result"] = result
        written["kwargs"] = kwargs

    monkeypatch.setattr(apply_job, "run_decision", fake_run_decision)
    monkeypatch.setattr(apply_job, "write_apply_result", fake_write)

    rc = main(
        [
            "--proposal-id=prop-1",
            "--agent-name=claude_code",
            "--decision=approve",
            "--approver=reviewer@databricks.com",
            "--reason=",  # empty -> None
            f"--decided-at={DECIDED_AT}",
            "--warehouse-id=wh-1",
            "--catalog=cat",
            "--schema=sch",
        ]
    )

    assert rc == 0
    assert captured["proposal_id"] == "prop-1"
    assert captured["agent_name"] == "claude_code"
    assert captured["decision"] == "approve"
    assert captured["approver"] == "reviewer@databricks.com"
    assert captured["reason"] is None  # empty string collapsed to None
    assert captured["decided_at"] == DECIDED_AT
    assert captured["warehouse_id"] == "wh-1"
    assert captured["catalog"] == "cat"
    assert captured["schema"] == "sch"
    # The real engine result is handed to the writer verbatim.
    assert written["result"].outcome is ApplyServiceOutcome.APPLIED
    assert written["kwargs"]["warehouse_id"] == "wh-1"
    assert written["kwargs"]["catalog"] == "cat"
    assert written["kwargs"]["schema"] == "sch"


def test_main_defaults_decided_at_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(apply_job, "resolve_job_auth", lambda **_: "minted")
    monkeypatch.setattr(apply_job, "_build_workspace_client", lambda _p: object())
    monkeypatch.setattr(apply_job, "write_apply_result", lambda *a, **k: None)

    def fake_run_decision(**kwargs: Any) -> ApplyServiceResult:
        captured.update(kwargs)
        return _result()

    monkeypatch.setattr(apply_job, "run_decision", fake_run_decision)

    rc = main(
        [
            "--proposal-id=prop-1",
            "--agent-name=claude_code",
            "--decision=approve",
            "--approver=reviewer@databricks.com",
            "--warehouse-id=wh-1",
        ]
    )
    assert rc == 0
    # A missing decided-at is stamped server-side (ISO 8601), never left blank.
    assert captured["decided_at"] and captured["decided_at"][:4].isdigit()


def test_main_requires_warehouse_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AIL_WAREHOUSE_ID", raising=False)
    monkeypatch.delenv("DATABRICKS_WAREHOUSE_ID", raising=False)
    with pytest.raises(SystemExit):
        main(
            [
                "--proposal-id=prop-1",
                "--agent-name=claude_code",
                "--decision=approve",
                "--approver=reviewer@databricks.com",
            ]
        )


def test_main_records_refused_result_for_malformed_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed decision is a fail-closed REFUSED result, recorded — never faked.

    Uses the REAL ``run_decision``: an unknown decision refuses at its early guard
    (before any client is built), so only the result WRITE is faked here.
    """
    written: dict[str, Any] = {}
    monkeypatch.setattr(apply_job, "resolve_job_auth", lambda **_: "minted")
    monkeypatch.setattr(apply_job, "_build_workspace_client", lambda _p: object())
    monkeypatch.setattr(
        apply_job,
        "write_apply_result",
        lambda result, **_: written.update(result=result),
    )

    rc = main(
        [
            "--proposal-id=prop-1",
            "--agent-name=claude_code",
            "--decision=delete",  # not approve/reject
            "--approver=reviewer@databricks.com",
            "--warehouse-id=wh-1",
        ]
    )

    assert rc == 0  # a decision-level outcome is not a job failure
    assert written["result"].outcome is ApplyServiceOutcome.REFUSED
    assert "unknown decision" in (written["result"].refused_reason or "")


# -- idempotency: the engine refuses a non-pending proposal -----------------


def test_run_decision_refuses_non_pending_proposal(monkeypatch: pytest.MonkeyPatch) -> None:
    """A retried/duplicated trigger on an already-applied proposal is REFUSED.

    ``load_pending_proposal`` returns ``None`` for any non-``pending`` row (already
    applied / rejected / superseded / unknown), so the engine refuses fail-closed
    rather than re-applying — the no-double-apply guarantee the Job transport relies
    on.
    """
    monkeypatch.setattr(apply_service, "_build_workspace_client", lambda _p: object())
    monkeypatch.setattr(apply_service, "load_pending_proposal", lambda **_: None)

    result = run_decision(
        proposal_id="prop-1",
        agent_name="claude_code",
        decision="approve",
        approver="reviewer@databricks.com",
        reason=None,
        decided_at=DECIDED_AT,
        warehouse_id="wh-1",
    )

    assert result.outcome is ApplyServiceOutcome.REFUSED
    assert "no pending proposal" in (result.refused_reason or "")


# -- write_apply_result: DDL + INSERT with the verbatim result JSON ---------


def test_write_apply_result_emits_ddl_and_insert(monkeypatch: pytest.MonkeyPatch) -> None:
    executed: list[str] = []
    monkeypatch.setattr(apply_job, "_execute", lambda _c, _w, sql: executed.append(sql))

    result = _result()
    write_apply_result(result, client=object(), warehouse_id="wh-1", catalog="cat", schema="sch")

    # Two DDL (schema + table) then one INSERT.
    assert len(executed) == 3
    assert "CREATE SCHEMA IF NOT EXISTS `cat`.`sch`" in executed[0]
    assert f"CREATE TABLE IF NOT EXISTS `cat`.`sch`.{APPLY_RESULTS_TABLE}" in executed[1]
    insert = executed[2]
    assert insert.startswith(f"INSERT INTO `cat`.`sch`.{APPLY_RESULTS_TABLE}")
    # The verbatim engine result JSON is stored (round-trips to the app bridge).
    assert result.model_dump_json() in insert
    assert "'prop-1'" in insert  # proposal_id key
    assert f"'{DECIDED_AT}'" in insert  # decided_at key
