"""Tests for the L7b-2 open-ended executor core (:mod:`ail.executor.executor`).

Every test is **offline**: the agent runner, the UC-Volume client, and the
preview/commit persistence are all fakes/spies, so no live Claude Agent SDK /
MLflow / Databricks call is ever made (no ``live`` marker). The suite pins each
load-bearing safety invariant the executor promises (``docs/EXECUTOR.md``):

* ``test_preview_leaves_live_workspace_untouched`` (a) — the live tree is
  byte-for-byte unchanged even though the agent ran (it edits the sandbox copy).
* ``test_commit_applies_stored_change_never_reruns_agent`` (b) — commit has no
  runner, never constructs one, and applies the STORED produced bytes (not a re-run
  that would differ).
* ``test_committed_equals_previewed`` (c) — the committed live bytes equal the
  previewed produced bytes (and match the ``preview_diff``).
* ``test_commit_snapshot_first_crash_safety`` (d) — a failed apply leaves the live
  tree untouched (L6 verify-before-write), with the pre-change revert point taken.
* the ``test_preview_refuses_*`` / ``test_commit_refuses_*`` group (e) — every
  fail-closed refusal.
* ``test_no_live_seams_touched`` (f) — only injected seams are exercised.
"""

from __future__ import annotations

import hashlib
import inspect
import os
from dataclasses import dataclass, field
from pathlib import Path

import pytest

import ail.executor.executor as ex
from ail.executor import (
    CommitRecordError,
    CommitRefused,
    CommittedChangeRecord,
    PreviewError,
    RevertError,
    commit_approved,
    produce_preview,
    revert_committed_change,
)
from ail.ingest.base import AgentRunResult, AgentTask, NormalizedTrace
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
from ail.registry import Agent
from ail.versioning import MANIFEST_FILENAME, FileSnapshot, SnapshotError, SnapshotRef

VOLUME_ROOT = "/Volumes/cat/sch/vol/ail_snapshots"


# ---------------------------------------------------------------------------
# Fakes / spies (in-memory; record calls — no live I/O)
# ---------------------------------------------------------------------------


@dataclass
class FakeVolumeClient:
    """In-memory UC Volume: a ``{volume_path: bytes}`` store (mirrors the L6 tests)."""

    store: dict[str, bytes] = field(default_factory=dict)
    upload_calls: list[str] = field(default_factory=list)
    download_calls: list[str] = field(default_factory=list)

    def upload(self, volume_path: str, contents: bytes) -> None:
        self.upload_calls.append(volume_path)
        self.store[volume_path] = bytes(contents)

    def download(self, volume_path: str) -> bytes:
        self.download_calls.append(volume_path)
        if volume_path not in self.store:
            raise FileNotFoundError(f"NOT_FOUND: {volume_path}")
        return self.store[volume_path]


class SpyRunner:
    """A fake :class:`~ail.executor.AgentRunner`: applies canned edits to the sandbox cwd.

    Records every call (and cwd) so a test can assert the agent ran exactly once — and,
    critically, that commit never re-ran it. ``edits``/``deletes`` are mutable so a test
    can change what a *hypothetical* re-run would produce and prove commit ignores it.
    """

    def __init__(
        self,
        edits: dict[str, bytes] | None = None,
        *,
        deletes: list[str] | None = None,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        self.edits = dict(edits or {})
        self.deletes = list(deletes or [])
        self.success = success
        self.error = error
        self.calls = 0
        self.cwds: list[str | None] = []

    def run(self, task: AgentTask) -> AgentRunResult:
        self.calls += 1
        self.cwds.append(task.cwd)
        if self.success:
            for rel, data in self.edits.items():
                fp = Path(task.cwd or ".") / rel
                fp.parent.mkdir(parents=True, exist_ok=True)
                fp.write_bytes(data)
            for rel in self.deletes:
                fp = Path(task.cwd or ".") / rel
                if fp.exists():
                    fp.unlink()
        return AgentRunResult(
            trace=NormalizedTrace(trace_id="t"), success=self.success, error=self.error
        )


@dataclass
class PreviewWriterSpy:
    calls: list[dict[str, str]] = field(default_factory=list)

    def __call__(
        self, *, agent_name: str, proposal_id: str, preview_diff: str, produced_change_ref: str
    ) -> None:
        self.calls.append(
            {
                "agent_name": agent_name,
                "proposal_id": proposal_id,
                "preview_diff": preview_diff,
                "produced_change_ref": produced_change_ref,
            }
        )


@dataclass
class CommitRecorderSpy:
    records: list[ex.CommittedChangeRecord] = field(default_factory=list)

    def __call__(self, record: ex.CommittedChangeRecord) -> None:
        self.records.append(record)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    (ws / "skills").mkdir(parents=True)
    (ws / "skills" / "token.md").write_text("# token skill\n\nOLD BODY\n")
    (ws / "keep.txt").write_text("unchanged\n")
    (ws / ".git").mkdir()  # VCS metadata — must be ignored by the copy/diff
    (ws / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    return ws


def _agent(ws: Path | None) -> Agent:
    return Agent(
        agent_name="claude_code",
        experiment_id="660599403165942",
        target_workspace=str(ws) if ws is not None else None,
    )


def _agent_task_proposal(
    *,
    plan: str = "Improve the token skill",
    produced_change_ref: str | None = None,
    preview_diff: str | None = None,
    status: ProposalStatus = ProposalStatus.PENDING,
) -> ProposedAction:
    change = ProposedChange(
        kind=ChangeKind.AGENT_TASK_PLAN,
        summary="open-ended change",
        plan=plan,
        produced_change_ref=produced_change_ref,
        preview_diff=preview_diff,
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
        trigger=TriggerSignal(kind=TriggerKind.AGENT_PLANNER, summary="planner said so"),
        change=change,
        gate_status=GateStatus(readiness_tier="ready_to_prove"),
    )


def _metric_view_proposal() -> ProposedAction:
    change = ProposedChange(
        kind=ChangeKind.METRIC_VIEW_SQL,
        summary="a view",
        sql="CREATE OR REPLACE VIEW `c`.`s`.`v` AS SELECT 1",
    )
    return ProposedAction(
        proposal_id="mv1",
        agent_name="claude_code",
        experiment_id="660599403165942",
        action_kind=ActionKind.METRIC_VIEW,
        risk_class=default_risk_class(ActionKind.METRIC_VIEW),
        objective_metric="total_tokens",
        goal_cohort="claude_code",
        trigger=TriggerSignal(kind=TriggerKind.RLM_RECOMMENDED_ASSET, summary="rlm"),
        change=change,
        gate_status=GateStatus(readiness_tier="ready_to_prove"),
    )


def _tree_hash(root: Path) -> dict[str, str]:
    """A path→sha256 map of every file under ``root`` (to assert a tree is untouched)."""
    out: dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def _preview(
    tmp_path: Path,
    *,
    runner: SpyRunner,
    proposal: ProposedAction | None = None,
    agent: Agent | None = None,
    volume: FakeVolumeClient | None = None,
    writer: PreviewWriterSpy | None = None,
) -> tuple[ex.PreviewResult, Agent, FakeVolumeClient]:
    ws = _workspace(tmp_path)
    ag = agent if agent is not None else _agent(ws)
    vol = volume if volume is not None else FakeVolumeClient()
    wr = writer if writer is not None else PreviewWriterSpy()
    prop = proposal if proposal is not None else _agent_task_proposal()
    result = produce_preview(
        prop,
        ag,
        volume_client=vol,
        volume_root=VOLUME_ROOT,
        preview_writer=wr,
        agent_runner=runner,
    )
    return result, ag, vol


# ---------------------------------------------------------------------------
# (a) produce_preview leaves the live workspace untouched
# ---------------------------------------------------------------------------


def test_preview_leaves_live_workspace_untouched(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    before = _tree_hash(ws)
    runner = SpyRunner({"skills/token.md": b"# token skill\n\nNEW BODY\n", "new/tool.py": b"x=1\n"})
    writer = PreviewWriterSpy()

    result = produce_preview(
        _agent_task_proposal(),
        _agent(ws),
        volume_client=FakeVolumeClient(),
        volume_root=VOLUME_ROOT,
        preview_writer=writer,
        agent_runner=runner,
    )

    assert runner.calls == 1  # the agent DID run
    assert _tree_hash(ws) == before  # ...but the live tree is byte-for-byte unchanged
    assert result.n_files == 2
    # the agent ran in a sandbox, never the live workspace
    assert runner.cwds[0] is not None and runner.cwds[0] != str(ws)
    assert writer.calls and writer.calls[0]["produced_change_ref"] == result.produced_change_ref


# ---------------------------------------------------------------------------
# (b) commit applies the STORED change and never re-runs the agent
# ---------------------------------------------------------------------------


def test_commit_has_no_agent_runner_parameter() -> None:
    # The strongest form of "commit never re-runs the agent": there is no seam to.
    assert "agent_runner" not in inspect.signature(commit_approved).parameters


def test_commit_applies_stored_change_never_reruns_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = SpyRunner({"skills/token.md": b"# token skill\n\nSTORED PREVIEW\n"})
    result, agent, vol = _preview(tmp_path, runner=runner)
    ws = Path(agent.target_workspace or "")

    # A commit must NEVER build the default SDK runner. Make that fail loudly if attempted.
    monkeypatch.setattr(
        ex,
        "_default_agent_runner",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("commit built an agent runner")),
    )
    # And if commit somehow re-ran the (injected-nowhere) spy, it would now write DIFFERENT
    # bytes — proving by the applied content that the STORED change was used, not a re-run.
    runner.edits = {"skills/token.md": b"# token skill\n\nRE-RUN DIFFERENT\n"}

    approved = result.proposal.model_copy(update={"status": ProposalStatus.APPROVED})
    recorder = CommitRecorderSpy()
    commit_result = commit_approved(
        approved,
        agent,
        volume_client=vol,
        volume_root=VOLUME_ROOT,
        commit_recorder=recorder,
        approver="alice@example.com",
        committed_at="2026-07-02T12:00:00Z",
    )

    assert runner.calls == 1  # still only the ONE preview run; commit did not re-run it
    assert (ws / "skills" / "token.md").read_bytes() == b"# token skill\n\nSTORED PREVIEW\n"
    assert commit_result.lineage_recorded is True
    assert commit_result.approver == "alice@example.com"
    assert recorder.records and recorder.records[0].approver == "alice@example.com"


# ---------------------------------------------------------------------------
# (c) committed change == previewed diff
# ---------------------------------------------------------------------------


def test_committed_equals_previewed(tmp_path: Path) -> None:
    produced = {
        "skills/token.md": b"# token skill\n\nNEW BODY\n",  # modified
        "skills/new_helper.md": b"# helper\n",  # added
    }
    runner = SpyRunner(produced)
    result, agent, vol = _preview(tmp_path, runner=runner)
    ws = Path(agent.target_workspace or "")

    # the preview_diff reflects exactly the produced change the human will review
    assert "NEW BODY" in result.preview_diff
    assert "modified: skills/token.md" in result.preview_diff
    assert "added: skills/new_helper.md" in result.preview_diff

    approved = result.proposal.model_copy(update={"status": ProposalStatus.APPROVED})
    commit_approved(
        approved,
        agent,
        volume_client=vol,
        volume_root=VOLUME_ROOT,
        commit_recorder=CommitRecorderSpy(),
        approver="a",
        committed_at="t",
    )

    # every committed file's live bytes equal the produced bytes the human previewed
    for rel, data in produced.items():
        assert (ws / rel).read_bytes() == data
    assert (ws / "keep.txt").read_bytes() == b"unchanged\n"  # untouched files stay


# ---------------------------------------------------------------------------
# (d) commit snapshots live FIRST → a failed apply leaves the tree restorable/untouched
# ---------------------------------------------------------------------------


def test_commit_snapshot_first_crash_safety(tmp_path: Path) -> None:
    runner = SpyRunner({"skills/token.md": b"# token skill\n\nWOULD-BE NEW\n"})
    result, agent, vol = _preview(tmp_path, runner=runner)
    ws = Path(agent.target_workspace or "")
    live_before = _tree_hash(ws)

    # Corrupt the stored produced change-set so the apply's verify-before-write fails.
    produced = ex.load_snapshot_ref(result.produced_change_ref, client=vol)
    del vol.store[produced.files[0].volume_path]

    approved = result.proposal.model_copy(update={"status": ProposalStatus.APPROVED})
    recorder = CommitRecorderSpy()
    with pytest.raises(SnapshotError):
        commit_approved(
            approved,
            agent,
            volume_client=vol,
            volume_root=VOLUME_ROOT,
            commit_recorder=recorder,
            approver="a",
            committed_at="t",
        )

    assert _tree_hash(ws) == live_before  # the live tree is byte-for-byte untouched
    assert recorder.records == []  # nothing recorded (the apply never succeeded)
    # the pre-change revert point WAS taken first (snapshot-live-first ordering)
    pre_dir = f"{VOLUME_ROOT}/{approved.proposal_id}-pre"
    pre_ref = ex.load_snapshot_ref(pre_dir, client=vol)
    assert any(str(ws / "skills" / "token.md") in f.original_path for f in pre_ref.files)


def test_commit_pure_add_has_no_prechange_ref(tmp_path: Path) -> None:
    runner = SpyRunner({"skills/brand_new.md": b"# brand new\n"})  # pure addition
    result, agent, vol = _preview(tmp_path, runner=runner)
    ws = Path(agent.target_workspace or "")

    approved = result.proposal.model_copy(update={"status": ProposalStatus.APPROVED})
    commit_result = commit_approved(
        approved,
        agent,
        volume_client=vol,
        volume_root=VOLUME_ROOT,
        commit_recorder=CommitRecorderSpy(),
        approver="a",
        committed_at="t",
    )

    assert commit_result.pre_change_ref is None  # nothing pre-existing to snapshot
    assert (ws / "skills" / "brand_new.md").read_bytes() == b"# brand new\n"


# ---------------------------------------------------------------------------
# (e) fail-closed refusals — produce_preview
# ---------------------------------------------------------------------------


def test_preview_refuses_non_agent_task(tmp_path: Path) -> None:
    with pytest.raises(PreviewError, match="only handles AGENT_TASK"):
        produce_preview(
            _metric_view_proposal(),
            _agent(_workspace(tmp_path)),
            volume_client=FakeVolumeClient(),
            volume_root=VOLUME_ROOT,
            preview_writer=PreviewWriterSpy(),
            agent_runner=SpyRunner(),
        )


def test_preview_refuses_missing_target_workspace(tmp_path: Path) -> None:
    runner = SpyRunner({"x": b"y"})
    with pytest.raises(PreviewError, match="no target_workspace"):
        produce_preview(
            _agent_task_proposal(),
            _agent(None),
            volume_client=FakeVolumeClient(),
            volume_root=VOLUME_ROOT,
            preview_writer=PreviewWriterSpy(),
            agent_runner=runner,
        )
    assert runner.calls == 0  # refused before running the agent


def test_preview_refuses_unreadable_workspace(tmp_path: Path) -> None:
    missing = _agent(tmp_path / "does_not_exist")
    with pytest.raises(PreviewError, match="not a readable directory"):
        produce_preview(
            _agent_task_proposal(),
            missing,
            volume_client=FakeVolumeClient(),
            volume_root=VOLUME_ROOT,
            preview_writer=PreviewWriterSpy(),
            agent_runner=SpyRunner(),
        )


def test_preview_refuses_agent_error_and_writes_nothing(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    before = _tree_hash(ws)
    vol = FakeVolumeClient()
    writer = PreviewWriterSpy()
    with pytest.raises(PreviewError, match="executor agent failed"):
        produce_preview(
            _agent_task_proposal(),
            _agent(ws),
            volume_client=vol,
            volume_root=VOLUME_ROOT,
            preview_writer=writer,
            agent_runner=SpyRunner(success=False, error="boom"),
        )
    assert vol.store == {}  # no snapshot written
    assert writer.calls == []  # no preview recorded
    assert _tree_hash(ws) == before  # live untouched


def test_preview_refuses_when_agent_produced_no_change(tmp_path: Path) -> None:
    vol = FakeVolumeClient()
    writer = PreviewWriterSpy()
    with pytest.raises(PreviewError, match="produced no change"):
        produce_preview(
            _agent_task_proposal(),
            _agent(_workspace(tmp_path)),
            volume_client=vol,
            volume_root=VOLUME_ROOT,
            preview_writer=writer,
            agent_runner=SpyRunner({}),  # edits nothing
        )
    assert vol.store == {}
    assert writer.calls == []


def test_preview_refuses_deletions(tmp_path: Path) -> None:
    # The L6 restore substrate cannot apply a deletion, so a deleting change is refused.
    with pytest.raises(PreviewError, match="deleted"):
        produce_preview(
            _agent_task_proposal(),
            _agent(_workspace(tmp_path)),
            volume_client=FakeVolumeClient(),
            volume_root=VOLUME_ROOT,
            preview_writer=PreviewWriterSpy(),
            agent_runner=SpyRunner(deletes=["keep.txt"]),
        )


def test_preview_record_failure_is_fail_closed(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    before = _tree_hash(ws)

    def _boom(**_kwargs: str) -> None:
        raise RuntimeError("UC UPDATE failed")

    with pytest.raises(PreviewError, match="recording it onto the proposal failed"):
        produce_preview(
            _agent_task_proposal(),
            _agent(ws),
            volume_client=FakeVolumeClient(),
            volume_root=VOLUME_ROOT,
            preview_writer=_boom,
            agent_runner=SpyRunner({"skills/token.md": b"# token skill\n\nNEW\n"}),
        )
    assert _tree_hash(ws) == before  # a failed record still leaves the live tree untouched


def test_preview_ignores_vcs_and_cache_dirs(tmp_path: Path) -> None:
    # An edit only under .git must not register as a change (produced nothing).
    with pytest.raises(PreviewError, match="produced no change"):
        produce_preview(
            _agent_task_proposal(),
            _agent(_workspace(tmp_path)),
            volume_client=FakeVolumeClient(),
            volume_root=VOLUME_ROOT,
            preview_writer=PreviewWriterSpy(),
            agent_runner=SpyRunner({".git/config": b"[core]\n"}),
        )


# ---------------------------------------------------------------------------
# (e) fail-closed refusals — commit_approved
# ---------------------------------------------------------------------------


def test_commit_refuses_non_approved(tmp_path: Path) -> None:
    runner = SpyRunner({"skills/token.md": b"# token skill\n\nNEW\n"})
    result, agent, vol = _preview(tmp_path, runner=runner)
    # still PENDING (never approved)
    with pytest.raises(CommitRefused, match="not approved"):
        commit_approved(
            result.proposal,
            agent,
            volume_client=vol,
            volume_root=VOLUME_ROOT,
            commit_recorder=CommitRecorderSpy(),
            approver="a",
            committed_at="t",
        )


def test_commit_refuses_missing_produced_ref(tmp_path: Path) -> None:
    # An approved AGENT_TASK with no produced_change_ref = the human approved no concrete diff.
    prop = _agent_task_proposal(status=ProposalStatus.APPROVED, produced_change_ref=None)
    with pytest.raises(CommitRefused, match="no produced_change_ref"):
        commit_approved(
            prop,
            _agent(_workspace(tmp_path)),
            volume_client=FakeVolumeClient(),
            volume_root=VOLUME_ROOT,
            commit_recorder=CommitRecorderSpy(),
            approver="a",
            committed_at="t",
        )


def test_commit_refuses_non_agent_task(tmp_path: Path) -> None:
    prop = _metric_view_proposal().model_copy(update={"status": ProposalStatus.APPROVED})
    with pytest.raises(CommitRefused, match="only handles AGENT_TASK"):
        commit_approved(
            prop,
            _agent(_workspace(tmp_path)),
            volume_client=FakeVolumeClient(),
            volume_root=VOLUME_ROOT,
            commit_recorder=CommitRecorderSpy(),
            approver="a",
            committed_at="t",
        )


def test_commit_refuses_unreadable_produced_ref(tmp_path: Path) -> None:
    prop = _agent_task_proposal(
        status=ProposalStatus.APPROVED, produced_change_ref=f"{VOLUME_ROOT}/nonexistent"
    )
    with pytest.raises(CommitRefused, match="missing or unreadable"):
        commit_approved(
            prop,
            _agent(_workspace(tmp_path)),
            volume_client=FakeVolumeClient(),  # empty store → no manifest there
            volume_root=VOLUME_ROOT,
            commit_recorder=CommitRecorderSpy(),
            approver="a",
            committed_at="t",
        )


def test_commit_refuses_path_outside_workspace(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    vol = FakeVolumeClient()
    # Craft a tampered produced manifest that targets a path OUTSIDE the workspace.
    snapshot_dir = f"{VOLUME_ROOT}/tampered"
    evil = str(tmp_path / "outside" / "evil.txt")
    ref = SnapshotRef(
        change_id="tampered",
        volume_root=VOLUME_ROOT,
        snapshot_dir=snapshot_dir,
        manifest_path=f"{snapshot_dir}/{MANIFEST_FILENAME}",
        files=[
            FileSnapshot(
                original_path=evil,
                volume_path=f"{snapshot_dir}/blobs/deadbeef",
                sha256="deadbeef",
                size=3,
            )
        ],
        created_at="t",
    )
    vol.upload(ref.manifest_path, ref.model_dump_json().encode("utf-8"))
    prop = _agent_task_proposal(status=ProposalStatus.APPROVED, produced_change_ref=snapshot_dir)

    with pytest.raises(CommitRefused, match="outside the target_workspace"):
        commit_approved(
            prop,
            _agent(ws),
            volume_client=vol,
            volume_root=VOLUME_ROOT,
            commit_recorder=CommitRecorderSpy(),
            approver="a",
            committed_at="t",
        )
    assert not (tmp_path / "outside" / "evil.txt").exists()  # nothing written outside


def test_commit_record_failure_surfaces_committed_but_unrecorded(tmp_path: Path) -> None:
    runner = SpyRunner({"skills/token.md": b"# token skill\n\nNEW\n"})
    result, agent, vol = _preview(tmp_path, runner=runner)
    ws = Path(agent.target_workspace or "")
    approved = result.proposal.model_copy(update={"status": ProposalStatus.APPROVED})

    def _boom(_record: ex.CommittedChangeRecord) -> None:
        raise RuntimeError("audit write failed")

    with pytest.raises(CommitRecordError) as ei:
        commit_approved(
            approved,
            agent,
            volume_client=vol,
            volume_root=VOLUME_ROOT,
            commit_recorder=_boom,
            approver="a",
            committed_at="t",
        )
    # The change IS live (applied before the record was attempted) — never rolled back.
    assert (ws / "skills" / "token.md").read_bytes() == b"# token skill\n\nNEW\n"
    assert ei.value.result.lineage_recorded is False
    assert ei.value.result.n_files == 1


# ---------------------------------------------------------------------------
# (f) only injected seams are exercised — no live SDK/DB path
# ---------------------------------------------------------------------------


def test_no_live_seams_touched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # If produce_preview reached its default (live) runner, this would raise.
    monkeypatch.setattr(
        ex,
        "_default_agent_runner",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("built the live SDK runner")),
    )
    runner = SpyRunner({"skills/token.md": b"# token skill\n\nNEW\n"})
    vol = FakeVolumeClient()
    result = produce_preview(
        _agent_task_proposal(),
        _agent(_workspace(tmp_path)),
        volume_client=vol,
        volume_root=VOLUME_ROOT,
        preview_writer=PreviewWriterSpy(),
        agent_runner=runner,
    )
    # all Volume I/O went through the injected fake
    assert vol.upload_calls and result.produced_change_ref.startswith(VOLUME_ROOT)


# ---------------------------------------------------------------------------
# Cross-review fixes — BLOCKER 1 + 2: symlink-escape safety
# ---------------------------------------------------------------------------


def test_preview_symlink_escape_does_not_write_outside(tmp_path: Path) -> None:
    """B1: an agent write THROUGH a pre-existing escaping symlink stays in the sandbox."""
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_bytes(b"SECRET")

    ws = tmp_path / "workspace"
    (ws / "skills").mkdir(parents=True)
    (ws / "skills" / "token.md").write_text("OLD\n")
    # a pre-existing (absolute) symlink in the workspace pointing OUTSIDE it
    os.symlink(str(secret), str(ws / "escape"))

    # the agent tries to write THROUGH the escaping symlink
    runner = SpyRunner({"escape": b"PWNED"})
    produce_preview(
        _agent_task_proposal(),
        _agent(ws),
        volume_client=FakeVolumeClient(),
        volume_root=VOLUME_ROOT,
        preview_writer=PreviewWriterSpy(),
        agent_runner=runner,
    )

    # the outside target is byte-for-byte untouched — the write landed inside the sandbox
    assert secret.read_bytes() == b"SECRET"


def test_preview_sdk_sandbox_blocks_absolute_write_outside(tmp_path: Path) -> None:
    """B1: the preview runner receives required SDK write scoping and fails closed."""
    outside = tmp_path / "outside.txt"
    outside.write_text("SECRET\n")

    ws = tmp_path / "workspace"
    (ws / "skills").mkdir(parents=True)
    (ws / "skills" / "token.md").write_text("OLD\n")

    class SandboxAwareRunner:
        def run(self, task: AgentTask) -> AgentRunResult:
            raw = task.params.get("claude_code_filesystem_sandbox")
            assert isinstance(raw, dict)
            assert raw["required"] is True
            sandbox_dir = os.path.realpath(str(raw["sandbox_dir"]))
            assert os.path.realpath(task.cwd or "") == sandbox_dir
            assert task.allowed_tools is not None
            assert f"Write({sandbox_dir}/**)" in task.allowed_tools

            attempted = os.path.realpath(outside)
            if attempted != sandbox_dir and not attempted.startswith(sandbox_dir + os.sep):
                return AgentRunResult(
                    trace=NormalizedTrace(trace_id="t"),
                    success=False,
                    error="blocked by Claude SDK filesystem sandbox",
                )
            outside.write_text("PWNED\n")
            return AgentRunResult(trace=NormalizedTrace(trace_id="t"), success=True)

    with pytest.raises(PreviewError, match="blocked by Claude SDK filesystem sandbox"):
        produce_preview(
            _agent_task_proposal(),
            _agent(ws),
            volume_client=FakeVolumeClient(),
            volume_root=VOLUME_ROOT,
            preview_writer=PreviewWriterSpy(),
            agent_runner=SandboxAwareRunner(),
        )

    assert outside.read_text() == "SECRET\n"


def test_preview_refuses_agent_created_escaping_symlink(tmp_path: Path) -> None:
    """B1 defense: a produced file whose real path escapes the sandbox is refused."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "target.txt").write_bytes(b"OUT")
    ws = tmp_path / "workspace"
    (ws / "skills").mkdir(parents=True)
    (ws / "skills" / "token.md").write_text("OLD\n")

    class EscapingRunner:
        calls = 0

        def run(self, task: AgentTask) -> AgentRunResult:
            self.calls += 1
            # the agent creates a NEW symlink inside the sandbox pointing outside it
            os.symlink(str(outside / "target.txt"), str(Path(task.cwd or ".") / "sneaky"))
            return AgentRunResult(trace=NormalizedTrace(trace_id="t"), success=True)

    with pytest.raises(PreviewError, match="escapes the sandbox"):
        produce_preview(
            _agent_task_proposal(),
            _agent(ws),
            volume_client=FakeVolumeClient(),
            volume_root=VOLUME_ROOT,
            preview_writer=PreviewWriterSpy(),
            agent_runner=EscapingRunner(),
        )


def test_commit_refuses_symlinked_parent_escape(tmp_path: Path) -> None:
    """B2: a stored change-set whose target resolves outside (symlinked parent) is refused."""
    outside = tmp_path / "outside"
    outside.mkdir()
    ws = tmp_path / "workspace"
    (ws / "skills").mkdir(parents=True)
    # a live symlink DIR inside the workspace that escapes to `outside`
    os.symlink(str(outside), str(ws / "linkdir"), target_is_directory=True)

    vol = FakeVolumeClient()
    snapshot_dir = f"{VOLUME_ROOT}/escape"
    # produced change targets <ws>/linkdir/evil.txt → realpath resolves to <outside>/evil.txt
    evil_target = str(ws / "linkdir" / "evil.txt")
    ref = SnapshotRef(
        change_id="escape",
        volume_root=VOLUME_ROOT,
        snapshot_dir=snapshot_dir,
        manifest_path=f"{snapshot_dir}/{MANIFEST_FILENAME}",
        files=[
            FileSnapshot(
                original_path=evil_target,
                volume_path=f"{snapshot_dir}/blobs/deadbeef",
                sha256="deadbeef",
                size=3,
            )
        ],
        created_at="t",
    )
    vol.upload(ref.manifest_path, ref.model_dump_json().encode("utf-8"))
    prop = _agent_task_proposal(status=ProposalStatus.APPROVED, produced_change_ref=snapshot_dir)

    with pytest.raises(CommitRefused, match="outside the target_workspace"):
        commit_approved(
            prop,
            _agent(ws),
            volume_client=vol,
            volume_root=VOLUME_ROOT,
            commit_recorder=CommitRecorderSpy(),
            approver="a",
            committed_at="t",
        )
    assert not (outside / "evil.txt").exists()  # nothing written through the symlink


# ---------------------------------------------------------------------------
# Cross-review fix — BLOCKER 3: a committed pure-add is revertible
# ---------------------------------------------------------------------------


def test_pure_add_commit_is_revertible(tmp_path: Path) -> None:
    """B3: a pure-add commit records added_paths; revert deletes exactly them."""
    runner = SpyRunner({"skills/brand_new.md": b"# brand new\n"})
    result, agent, vol = _preview(tmp_path, runner=runner)
    ws = Path(agent.target_workspace or "")
    approved = result.proposal.model_copy(update={"status": ProposalStatus.APPROVED})
    recorder = CommitRecorderSpy()

    commit_result = commit_approved(
        approved,
        agent,
        volume_client=vol,
        volume_root=VOLUME_ROOT,
        commit_recorder=recorder,
        approver="a",
        committed_at="t",
    )
    added = str(ws / "skills" / "brand_new.md")
    assert commit_result.pre_change_ref is None  # nothing overwritten
    assert commit_result.added_paths == [added]  # ...but the addition IS recorded
    assert (ws / "skills" / "brand_new.md").exists()

    record = recorder.records[0]
    revert = revert_committed_change(record, volume_client=vol)

    assert not (ws / "skills" / "brand_new.md").exists()  # the added file is gone
    assert (ws / "skills" / "token.md").read_bytes() == b"# token skill\n\nOLD BODY\n"  # untouched
    assert (ws / "keep.txt").read_bytes() == b"unchanged\n"  # untouched
    assert revert.removed_added_paths == [added] and revert.n_removed == 1


def test_revert_restores_overwritten_and_deletes_added(tmp_path: Path) -> None:
    """B3: a mixed change reverts to restore modified files AND delete added ones."""
    runner = SpyRunner(
        {"skills/token.md": b"# token skill\n\nNEW BODY\n", "skills/added.md": b"# added\n"}
    )
    result, agent, vol = _preview(tmp_path, runner=runner)
    ws = Path(agent.target_workspace or "")
    approved = result.proposal.model_copy(update={"status": ProposalStatus.APPROVED})
    recorder = CommitRecorderSpy()
    commit_approved(
        approved,
        agent,
        volume_client=vol,
        volume_root=VOLUME_ROOT,
        commit_recorder=recorder,
        approver="a",
        committed_at="t",
    )
    assert (ws / "skills" / "token.md").read_bytes() == b"# token skill\n\nNEW BODY\n"
    assert (ws / "skills" / "added.md").exists()

    revert = revert_committed_change(recorder.records[0], volume_client=vol)

    assert (ws / "skills" / "token.md").read_bytes() == b"# token skill\n\nOLD BODY\n"  # restored
    assert not (ws / "skills" / "added.md").exists()  # deleted
    assert revert.n_restored == 1 and revert.n_removed == 1


def test_revert_refuses_added_path_outside_workspace(tmp_path: Path) -> None:
    """B3: revert never deletes outside the recorded workspace (fail-closed)."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / "victim.txt"
    victim.write_bytes(b"KEEP ME")
    record = CommittedChangeRecord(
        proposal_id="p1",
        agent_name="claude_code",
        target_workspace=str(ws),
        produced_change_ref=f"{VOLUME_ROOT}/p1",
        pre_change_ref=None,
        n_files=1,
        changed_paths=[str(victim)],
        added_paths=[str(victim)],
        summary="s",
        approver="a",
        committed_at="t",
    )
    with pytest.raises(RevertError, match="outside the recorded workspace"):
        revert_committed_change(record, volume_client=FakeVolumeClient())
    assert victim.read_bytes() == b"KEEP ME"  # never deleted outside the workspace


# ---------------------------------------------------------------------------
# Cross-review fix — BLOCKER 4: preview_diff reflects the STORED snapshot bytes
# ---------------------------------------------------------------------------


def test_preview_diff_from_snapshot_ignores_post_snapshot_sandbox_mutation(tmp_path: Path) -> None:
    """B4: the diff renders from the snapshot bytes, not a (mutable) sandbox re-read."""
    ws = tmp_path / "workspace"
    (ws / "skills").mkdir(parents=True)
    (ws / "skills" / "token.md").write_text("OLD\n")

    sandbox = tmp_path / "sandbox"
    (sandbox / "skills").mkdir(parents=True)
    (sandbox / "skills" / "token.md").write_bytes(b"STORED PREVIEW BYTES\n")

    vol = FakeVolumeClient()
    changes = [ex.FileChange(path="skills/token.md", change_type="modified")]
    ref = ex._snapshot_produced(
        changes,
        sandbox_root=sandbox,
        live_root=ws.resolve(),
        volume_root=VOLUME_ROOT,
        change_id="chg",
        client=vol,
    )
    # a background process the agent left mutates the sandbox AFTER the snapshot
    (sandbox / "skills" / "token.md").write_bytes(b"MUTATED AFTER SNAPSHOT\n")

    diff, rendered = ex._render_preview_from_snapshot(ref, live_root=ws.resolve(), client=vol)

    assert "STORED PREVIEW BYTES" in diff  # reflects the stored snapshot
    assert "MUTATED AFTER SNAPSHOT" not in diff  # NOT the post-snapshot sandbox
    # ...and it is exactly what commit would apply (the stored blob bytes)
    assert vol.download(ref.files[0].volume_path) == b"STORED PREVIEW BYTES\n"
    assert [c.path for c in rendered] == ["skills/token.md"]
