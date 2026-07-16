"""Companion-executor runner tests (:mod:`ail.jobs.agent_executor`) — offline, seams faked.

The ``ail-agent-executor`` entrypoint wires the L7b-2 executor (preview + commit) to
the app's ``agent_proposed_actions`` table. These tests fake every live seam (the
workspace/volume/runner builders and the persistence read/write helpers), so no live
Claude Agent SDK / MLflow / Databricks call is made (no ``live`` marker). They pin the
entrypoint's own contract:

* **static auth** — refuses to run without a static token (reuses the companion's
  resolver), dropping any ambient ``DATABRICKS_CONFIG_PROFILE``;
* **fail-closed on an unreadable table** — a read failure returns non-zero and does
  nothing (never previews/commits on an unknown state);
* **dry-run** — surfaces what it WOULD do and writes/commits nothing;
* **a real run** — previews a not-yet-previewed pending proposal (recording the diff),
  SKIPS a pending one that already has a preview, and commits an approved one to the
  live workspace (advancing its status), driving the REAL preview/commit functions
  through fakes end-to-end; and
* the persistence writers emit the expected SQL.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from databricks.sdk.service.sql import StatementState

from ail.executor import CommittedChangeRecord, produce_preview
from ail.ingest.base import AgentRunResult, AgentTask, NormalizedTrace
from ail.jobs import agent_executor as ax
from ail.loop.proposals import (
    ActionKind,
    ChangeKind,
    GateStatus,
    ProposalStatus,
    ProposedAction,
    ProposedChange,
    TriggerKind,
    TriggerSignal,
    default_risk_class,
    derive_proposal_id,
)
from ail.publish_versions import REGISTRY_COLUMNS, _registry_row
from ail.registry import Agent

VOLUME_ROOT = "/Volumes/cat/sch/vol/ail_snapshots"


# ---------------------------------------------------------------------------
# A stub UC client that serves the registry read (same serialization the publish tier
# writes), so the executor's registry-driven resolution is exercised through the REAL
# read path with no live warehouse.
# ---------------------------------------------------------------------------


class _Col:
    def __init__(self, name: str) -> None:
        self.name = name


class _Schema:
    def __init__(self, cols: list[str]) -> None:
        self.columns = [_Col(c) for c in cols]


class _Manifest:
    def __init__(self, cols: list[str]) -> None:
        self.schema = _Schema(cols)


class _Result:
    def __init__(self, rows: list[list]) -> None:  # type: ignore[type-arg]
        self.data_array = rows


class _Status:
    def __init__(self) -> None:
        self.state = StatementState.SUCCEEDED
        self.error = None


class _Resp:
    def __init__(self, cols: list[str], rows: list[list]) -> None:  # type: ignore[type-arg]
        self.statement_id = "stmt"
        self.status = _Status()
        self.manifest = _Manifest(cols)
        self.result = _Result(rows)


class _StmtExec:
    def __init__(self, resp: _Resp) -> None:
        self._resp = resp

    def execute_statement(self, *, warehouse_id, statement, wait_timeout=None):  # type: ignore[no-untyped-def]
        return self._resp

    def get_statement(self, statement_id):  # type: ignore[no-untyped-def]
        return self._resp


class _RaisingStmtExec:
    def execute_statement(self, *, warehouse_id, statement, wait_timeout=None):  # type: ignore[no-untyped-def]
        raise AssertionError("UC registry was read on the local-YAML override path")

    def get_statement(self, statement_id):  # type: ignore[no-untyped-def]
        raise AssertionError("UC registry was read on the local-YAML override path")


class _RegistryStubClient:
    """Serves the given agents as an ``agent_registry`` SELECT * (real read path)."""

    def __init__(self, *agents: Agent) -> None:
        rows = [_registry_row(a, generated_at="t") for a in agents]
        self.statement_execution = _StmtExec(_Resp(list(REGISTRY_COLUMNS), rows))


class _NeverReadClient:
    """A client whose registry read would raise — proves the YAML path never touches UC."""

    def __init__(self) -> None:
        self.statement_execution = _RaisingStmtExec()


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class FakeVolumeClient:
    store: dict[str, bytes] = field(default_factory=dict)

    def upload(self, volume_path: str, contents: bytes) -> None:
        self.store[volume_path] = bytes(contents)

    def download(self, volume_path: str) -> bytes:
        if volume_path not in self.store:
            raise FileNotFoundError(volume_path)
        return self.store[volume_path]


class SpyRunner:
    def __init__(self, edits: dict[str, bytes]) -> None:
        self.edits = dict(edits)
        self.calls = 0

    def run(self, task: AgentTask) -> AgentRunResult:
        self.calls += 1
        for rel, data in self.edits.items():
            fp = Path(task.cwd or ".") / rel
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_bytes(data)
        return AgentRunResult(trace=NormalizedTrace(trace_id="t"), success=True)


def _args(extra: list[str] | None = None):  # type: ignore[no-untyped-def]
    base = [
        "--agent",
        "claude_code",
        "--warehouse-id",
        "wh1",
        "--volume-root",
        VOLUME_ROOT,
        "--host",
        "https://example.databricks.com",
    ]
    return ax._parse_args(base + (extra or []))


def _proposal(
    *, plan: str, status: ProposalStatus, produced_change_ref: str | None = None
) -> ProposedAction:
    change = ProposedChange(
        kind=ChangeKind.AGENT_TASK_PLAN,
        summary="s",
        plan=plan,
        produced_change_ref=produced_change_ref,
    )
    pid = derive_proposal_id(
        agent_name="claude_code", action_kind=ActionKind.AGENT_TASK, change=change
    )
    return ProposedAction(
        proposal_id=pid,
        agent_name="claude_code",
        experiment_id="660599403165942",
        action_kind=ActionKind.AGENT_TASK,
        risk_class=default_risk_class(ActionKind.AGENT_TASK),
        status=status,
        objective_metric="total_tokens",
        goal_cohort="claude_code",
        trigger=TriggerSignal(kind=TriggerKind.AGENT_PLANNER, summary="why"),
        change=change,
        gate_status=GateStatus(readiness_tier="ready"),
    )


# ---------------------------------------------------------------------------
# static auth
# ---------------------------------------------------------------------------


def test_refuses_without_static_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "some-oauth-profile")
    with pytest.raises(SystemExit, match="STATIC Databricks token"):
        ax.resolve_static_auth(_args())
    assert "DATABRICKS_CONFIG_PROFILE" not in os.environ  # never falls back to OAuth


# ---------------------------------------------------------------------------
# registry-driven resolution: UC by default, YAML as an explicit local-dev override
# ---------------------------------------------------------------------------


def test_resolve_agent_defaults_to_uc_registry_via_stub_client() -> None:
    # A UI-onboarded agent present ONLY in UC (no --registry, no YAML) is resolvable by
    # the executor: its target_workspace + experiment_id come from the UC agent_registry.
    ui_agent = Agent(
        agent_name="ui_onboarded",
        experiment_id="exp-uc",
        target_workspace="/repos/ui_onboarded",
    )
    client = _RegistryStubClient(Agent(agent_name="other", experiment_id="e0"), ui_agent)

    got = ax._resolve_agent(
        "ui_onboarded", None, warehouse_id="wh1", catalog="cat", schema="sch", client=client
    )
    assert got.experiment_id == "exp-uc"  # from UC, not a YAML / guessed value
    assert got.target_workspace == "/repos/ui_onboarded"  # from UC — the executor's edit target


def test_resolve_agent_yaml_override_does_not_read_uc(tmp_path: Path) -> None:
    # An EXISTING --registry YAML is the local-dev override: resolve from the file, and
    # never touch UC (the passed client would raise if read).
    ws = tmp_path / "ws"
    ws.mkdir()
    registry = tmp_path / "agents.yaml"
    registry.write_text(
        "\n".join(
            [
                "agents:",
                "  - agent_name: local_dev",
                "    experiment_id: exp-yaml",
                f"    target_workspace: {ws}",
            ]
        ),
        encoding="utf-8",
    )
    got = ax._resolve_agent(
        "local_dev",
        str(registry),
        warehouse_id="wh1",
        catalog="cat",
        schema="sch",
        client=_NeverReadClient(),
    )
    assert got.experiment_id == "exp-yaml"
    assert got.target_workspace == str(ws)


def test_run_uc_agent_without_target_workspace_fails_closed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The unchanged fail-closed guarantee, now sourced from UC: a UI-onboarded agent with
    # no target_workspace configured cannot be previewed — the executor SKIPS it loud,
    # never runs the agent against a guessed tree.
    unconfigured = Agent(agent_name="ui_onboarded", experiment_id="exp-uc")  # target_workspace=None
    monkeypatch.setattr(
        ax, "_build_workspace_client", lambda *a, **k: _RegistryStubClient(unconfigured)
    )

    pending = _proposal(plan="preview me", status=ProposalStatus.PENDING)

    def _list(client, wh, *, status, catalog, schema):  # type: ignore[no-untyped-def]
        return [pending] if status is ProposalStatus.PENDING else []

    monkeypatch.setattr(ax, "list_agent_task_proposals", _list)
    monkeypatch.setattr(ax, "_build_volume_client", lambda *a, **k: FakeVolumeClient())
    monkeypatch.setattr(ax, "_build_agent_runner", lambda *a, **k: SpyRunner({}))

    # --agent ui_onboarded, NO --registry -> resolves from the stub UC registry.
    code = ax.run(
        ax._parse_args(
            [
                "--agent",
                "ui_onboarded",
                "--warehouse-id",
                "wh1",
                "--volume-root",
                VOLUME_ROOT,
                "--host",
                "https://example.databricks.com",
            ]
        )
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "resolved_from=uc-registry" in out  # UC was the source, not a YAML
    assert "SKIPPED" in out and "no target_workspace" in out  # fail-closed, never a guessed tree


# ---------------------------------------------------------------------------
# fail-closed on an unreadable proposals table
# ---------------------------------------------------------------------------


def test_run_unreadable_table_returns_nonzero_and_does_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    agent = Agent(agent_name="claude_code", experiment_id="1", target_workspace=str(tmp_path))
    monkeypatch.setattr(ax, "_resolve_agent", lambda *a, **k: agent)
    monkeypatch.setattr(ax, "_build_workspace_client", lambda *a, **k: object())

    def _boom(*a, **k):  # type: ignore[no-untyped-def]
        raise RuntimeError("warehouse unreachable")

    monkeypatch.setattr(ax, "list_agent_task_proposals", _boom)
    # these must never be reached (fail-closed before building live volume/runner)
    monkeypatch.setattr(
        ax,
        "_build_volume_client",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("built volume client")),
    )

    code = ax.run(_args())
    assert code == 2
    assert "could not read AGENT_TASK proposals" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# dry-run: surfaces, does nothing
# ---------------------------------------------------------------------------


def test_run_dry_run_previews_and_commits_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    agent = Agent(agent_name="claude_code", experiment_id="1", target_workspace=str(tmp_path))
    monkeypatch.setattr(ax, "_resolve_agent", lambda *a, **k: agent)
    monkeypatch.setattr(ax, "_build_workspace_client", lambda *a, **k: object())

    pending = _proposal(plan="do a", status=ProposalStatus.PENDING)
    approved = _proposal(
        plan="do b", status=ProposalStatus.APPROVED, produced_change_ref=f"{VOLUME_ROOT}/ref"
    )

    def _list(client, wh, *, status, catalog, schema):  # type: ignore[no-untyped-def]
        return [pending] if status is ProposalStatus.PENDING else [approved]

    monkeypatch.setattr(ax, "list_agent_task_proposals", _list)

    def _forbidden(*_a, **_k):  # type: ignore[no-untyped-def]
        raise AssertionError("dry-run touched a live seam / wrote a row")

    # dry-run must not build the live volume/runner nor write/commit anything
    for attr in (
        "_build_volume_client",
        "_build_agent_runner",
        "write_preview",
        "mark_committed",
        "record_commit",
    ):
        monkeypatch.setattr(ax, attr, _forbidden)

    code = ax.run(_args(["--dry-run"]))
    out = capsys.readouterr().out
    assert code == 0
    assert "WOULD PREVIEW" in out
    assert "WOULD COMMIT" in out
    assert "DRY-RUN" in out


# ---------------------------------------------------------------------------
# a real run: preview the un-previewed, skip the previewed, commit the approved
# ---------------------------------------------------------------------------


def test_run_previews_pending_skips_previewed_and_commits_approved(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    ws = tmp_path / "workspace"
    (ws / "skills").mkdir(parents=True)
    (ws / "skills" / "token.md").write_text("OLD\n")
    agent = Agent(agent_name="claude_code", experiment_id="1", target_workspace=str(ws))
    vol = FakeVolumeClient()

    # Pre-create the approved proposal's produced snapshot in the SAME volume store, so
    # the run's commit phase applies exactly that stored change.
    approved_base = _proposal(plan="commit me", status=ProposalStatus.PENDING)
    pre = produce_preview(
        approved_base,
        agent,
        volume_client=vol,
        volume_root=VOLUME_ROOT,
        preview_writer=lambda **k: None,
        agent_runner=SpyRunner({"skills/token.md": b"COMMITTED BODY\n"}),
    )
    approved = pre.proposal.model_copy(update={"status": ProposalStatus.APPROVED})

    pending_new = _proposal(plan="preview me", status=ProposalStatus.PENDING)
    pending_done = _proposal(
        plan="already previewed",
        status=ProposalStatus.PENDING,
        produced_change_ref=f"{VOLUME_ROOT}/already",
    )

    def _list(client, wh, *, status, catalog, schema):  # type: ignore[no-untyped-def]
        if status is ProposalStatus.PENDING:
            return [pending_new, pending_done]
        return [approved]

    previews_written: list[str] = []
    marked: list[str] = []
    recorded: list[object] = []

    monkeypatch.setattr(ax, "_resolve_agent", lambda *a, **k: agent)
    monkeypatch.setattr(ax, "_build_workspace_client", lambda *a, **k: object())
    monkeypatch.setattr(ax, "_build_volume_client", lambda *a, **k: vol)
    monkeypatch.setattr(
        ax, "_build_agent_runner", lambda *a, **k: SpyRunner({"skills/preview_new.md": b"NEW\n"})
    )
    monkeypatch.setattr(ax, "list_agent_task_proposals", _list)

    def _write_preview(client, wh, *, agent_name, proposal_id, **_k):  # type: ignore[no-untyped-def]
        previews_written.append(proposal_id)

    def _mark_committed(client, wh, *, agent_name, proposal_id, **_k):  # type: ignore[no-untyped-def]
        marked.append(proposal_id)

    def _record_commit(record, **_k):  # type: ignore[no-untyped-def]
        recorded.append(record)

    monkeypatch.setattr(ax, "write_preview", _write_preview)
    monkeypatch.setattr(ax, "mark_committed", _mark_committed)
    monkeypatch.setattr(ax, "record_commit", _record_commit)
    monkeypatch.setattr(ax, "latest_approver", lambda *a, **k: "human@databricks.com")

    code = ax.run(_args())
    assert code == 0

    # the un-previewed pending proposal was previewed (its row recorded); the already-
    # previewed one was skipped (never re-run under the human's feet).
    assert previews_written == [pending_new.proposal_id]
    assert pending_done.proposal_id not in previews_written

    # the approved proposal was committed: the STORED change landed in the live workspace,
    # its status advanced, and the commit was recorded with the resolved human approver.
    assert (ws / "skills" / "token.md").read_bytes() == b"COMMITTED BODY\n"
    assert marked == [approved.proposal_id]
    assert recorded and recorded[0].approver == "human@databricks.com"


# ---------------------------------------------------------------------------
# persistence writers emit the expected SQL
# ---------------------------------------------------------------------------


def _guarded_query_spy(monkeypatch: pytest.MonkeyPatch, *, affected: str) -> list[str]:
    """Route the guarded-update path through a fake ``_query_rows`` returning ``affected`` rows."""
    executed: list[str] = []

    def _fake(client, warehouse_id, statement):  # type: ignore[no-untyped-def]
        executed.append(statement)
        return [{"num_affected_rows": affected}]

    monkeypatch.setattr(ax, "_query_rows", _fake)
    return executed


def test_write_preview_is_idempotent_guarded(monkeypatch: pytest.MonkeyPatch) -> None:
    executed = _guarded_query_spy(monkeypatch, affected="1")
    n = ax.write_preview(
        object(),
        "w",
        agent_name="claude_code",
        proposal_id="p1",
        preview_diff="--- a\n+++ b\n",
        produced_change_ref=f"{VOLUME_ROOT}/p1",
    )
    sql = executed[0]
    assert n == 1  # confirmed the intended row was updated
    assert sql.startswith("UPDATE")
    assert "change_produced_change_ref IS NULL" in sql  # never overwrites an existing preview
    assert "status = 'pending'" in sql


def test_mark_committed_only_advances_approved(monkeypatch: pytest.MonkeyPatch) -> None:
    executed = _guarded_query_spy(monkeypatch, affected="1")
    n = ax.mark_committed(object(), "w", agent_name="claude_code", proposal_id="p1")
    sql = executed[0]
    assert n == 1
    assert "SET status = 'applied'" in sql
    assert "status = 'approved'" in sql  # only advances a still-approved row


def test_write_preview_zero_rows_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """B5: a zero-row guard match is a FAIL, never a silent success."""
    _guarded_query_spy(monkeypatch, affected="0")
    with pytest.raises(ax.GuardedUpdateError, match="matched 0 rows"):
        ax.write_preview(
            object(),
            "w",
            agent_name="claude_code",
            proposal_id="p1",
            preview_diff="d",
            produced_change_ref=f"{VOLUME_ROOT}/p1",
        )


def test_mark_committed_zero_rows_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """B5: a zero-row status-mark is a FAIL, never a silent success."""
    _guarded_query_spy(monkeypatch, affected="0")
    with pytest.raises(ax.GuardedUpdateError, match="matched 0 rows"):
        ax.mark_committed(object(), "w", agent_name="claude_code", proposal_id="p1")


def test_guarded_update_unconfirmed_count_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """B5: an unreadable affected-row count is fail-closed too (never assumed a success)."""
    monkeypatch.setattr(ax, "_query_rows", lambda c, w, s: [])  # no count returned
    with pytest.raises(ax.GuardedUpdateError, match="could not confirm"):
        ax.mark_committed(object(), "w", agent_name="claude_code", proposal_id="p1")


# ---------------------------------------------------------------------------
# B5 end-to-end: a zero-row status-mark after a live commit → committed-but-unrecorded
# ---------------------------------------------------------------------------


def test_commit_zero_row_status_mark_is_committed_but_unrecorded(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    ws = tmp_path / "workspace"
    (ws / "skills").mkdir(parents=True)
    (ws / "skills" / "token.md").write_text("OLD\n")
    agent = Agent(agent_name="claude_code", experiment_id="1", target_workspace=str(ws))
    vol = FakeVolumeClient()

    base = _proposal(plan="commit me", status=ProposalStatus.PENDING)
    pre = produce_preview(
        base,
        agent,
        volume_client=vol,
        volume_root=VOLUME_ROOT,
        preview_writer=lambda **k: None,
        agent_runner=SpyRunner({"skills/token.md": b"COMMITTED\n"}),
    )
    approved = pre.proposal.model_copy(update={"status": ProposalStatus.APPROVED})

    def _list(client, wh, *, status, catalog, schema):  # type: ignore[no-untyped-def]
        return [approved] if status is ProposalStatus.APPROVED else []

    monkeypatch.setattr(ax, "_resolve_agent", lambda *a, **k: agent)
    monkeypatch.setattr(ax, "_build_workspace_client", lambda *a, **k: object())
    monkeypatch.setattr(ax, "_build_volume_client", lambda *a, **k: vol)
    monkeypatch.setattr(ax, "_build_agent_runner", lambda *a, **k: SpyRunner({}))
    monkeypatch.setattr(ax, "list_agent_task_proposals", _list)
    monkeypatch.setattr(ax, "record_commit", lambda record, **k: None)
    monkeypatch.setattr(ax, "latest_approver", lambda *a, **k: "human@databricks.com")

    def _zero_row_mark(*a, **k):  # type: ignore[no-untyped-def]
        raise ax.GuardedUpdateError("status-mark: matched 0 rows (fail-closed)")

    monkeypatch.setattr(ax, "mark_committed", _zero_row_mark)

    code = ax.run(_args())
    out = capsys.readouterr().out
    assert code == 0
    # the change WAS applied live...
    assert (ws / "skills" / "token.md").read_bytes() == b"COMMITTED\n"
    # ...but a zero-row status-mark is surfaced LOUD as committed-but-unrecorded, never clean
    assert "COMMITTED-BUT-UNRECORDED" in out
    assert f"COMMITTED {approved.proposal_id}:" not in out


# ---------------------------------------------------------------------------
# B3 end-to-end: --revert removes exactly the recorded added files
# ---------------------------------------------------------------------------


def test_revert_mode_removes_recorded_added_files(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    ws = tmp_path / "workspace"
    (ws / "skills").mkdir(parents=True)
    added = ws / "skills" / "brand_new.md"
    added.write_text("# brand new\n")  # the file a prior pure-add commit created
    keep = ws / "keep.txt"
    keep.write_text("keep\n")
    agent = Agent(agent_name="claude_code", experiment_id="1", target_workspace=str(ws))

    record = CommittedChangeRecord(
        proposal_id="pid-1",
        agent_name="claude_code",
        target_workspace=str(ws),
        produced_change_ref=f"{VOLUME_ROOT}/pid-1",
        pre_change_ref=None,  # pure add: revert is purely deletes
        n_files=1,
        changed_paths=[str(added)],
        added_paths=[str(added)],
        summary="s",
        approver="a",
        committed_at="t",
    )

    monkeypatch.setattr(ax, "_resolve_agent", lambda *a, **k: agent)
    monkeypatch.setattr(ax, "_build_workspace_client", lambda *a, **k: object())
    monkeypatch.setattr(ax, "_build_volume_client", lambda *a, **k: FakeVolumeClient())
    monkeypatch.setattr(ax, "load_commit_record", lambda *a, **k: record)

    code = ax.run(_args(["--revert", "pid-1"]))
    out = capsys.readouterr().out
    assert code == 0
    assert not added.exists()  # the added file was deleted (revert of an add)
    assert keep.read_bytes() == b"keep\n"  # nothing else touched
    assert "REVERTED pid-1" in out


def test_revert_mode_no_record_fails_closed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    agent = Agent(agent_name="claude_code", experiment_id="1", target_workspace=str(tmp_path))
    monkeypatch.setattr(ax, "_resolve_agent", lambda *a, **k: agent)
    monkeypatch.setattr(ax, "_build_workspace_client", lambda *a, **k: object())
    monkeypatch.setattr(ax, "load_commit_record", lambda *a, **k: None)
    monkeypatch.setattr(
        ax,
        "_build_volume_client",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("built volume client")),
    )
    code = ax.run(_args(["--revert", "unknown"]))
    assert code == 2
    assert "no recorded commit" in capsys.readouterr().out


def test_record_commit_emits_ddl_and_insert(monkeypatch: pytest.MonkeyPatch) -> None:
    executed: list[str] = []
    monkeypatch.setattr(ax, "_execute", lambda c, w, s: executed.append(s))
    record = CommittedChangeRecord(
        proposal_id="p1",
        agent_name="claude_code",
        target_workspace="/ws",
        produced_change_ref=f"{VOLUME_ROOT}/p1",
        pre_change_ref=f"{VOLUME_ROOT}/p1-pre",
        n_files=2,
        changed_paths=["a", "b"],
        added_paths=["b"],
        summary="committed 2 files",
        approver="human@databricks.com",
        committed_at="2026-07-02T00:00:00Z",
    )
    ax.record_commit(record, client=object(), warehouse_id="w")
    assert any(
        "CREATE TABLE IF NOT EXISTS" in s and "agent_executor_commits" in s for s in executed
    )
    ddl = [s for s in executed if "CREATE TABLE IF NOT EXISTS" in s][0]
    assert "added_paths STRING" in ddl  # the revert channel column is persisted
    insert = [s for s in executed if s.startswith("INSERT INTO")][0]
    assert "'human@databricks.com'" in insert
    assert f"'{VOLUME_ROOT}/p1-pre'" in insert
    assert '["b"]' in insert  # added_paths recorded as a JSON array
