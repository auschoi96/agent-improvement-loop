from __future__ import annotations

import difflib
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from ail.compare.contract import Recommendation
from ail.executor.gepa_apply import (
    GepaApplyConflict,
    GepaValidationFailed,
    apply_approved_gepa,
)
from ail.loop.proposals import (
    ActionKind,
    ChangeKind,
    GateStatus,
    LocalApplySpec,
    LocalApplyTargetKind,
    ProofSummary,
    ProposalStatus,
    ProposedAction,
    ProposedChange,
    RiskClass,
    TriggerKind,
    TriggerSignal,
)
from ail.optimize.gepa_runner import GepaOptimizationResult
from ail.optimize.phase2 import L1Outcome, Phase2Artifact, TaskOutcome
from ail.registry import Agent


@dataclass
class FakeVolumeClient:
    store: dict[str, bytes] = field(default_factory=dict)

    def upload(self, volume_path: str, contents: bytes) -> None:
        self.store[volume_path] = bytes(contents)

    def download(self, volume_path: str) -> bytes:
        if volume_path not in self.store:
            raise FileNotFoundError(volume_path)
        return self.store[volume_path]


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _artifact(pct: float) -> Phase2Artifact:
    return Phase2Artifact(
        suite_version="suite-v1",
        suite_content_hash="abc",
        objective_metric="total_tokens",
        n_tasks=1,
        n_promote=1,
        realized_token_savings_absolute=pct,
        realized_token_savings_pct=pct,
        outcomes=[
            TaskOutcome(
                task_id="heldout-1",
                recommendation=Recommendation.PROMOTE,
                l1_outcome=L1Outcome.PASSED,
            )
        ],
    )


def _candidate(seed: str = "# Seed\n", evolved: str = "# Better\n") -> GepaOptimizationResult:
    return GepaOptimizationResult(
        generated_at="2026-07-16T00:00:00+00:00",
        changed=True,
        seed_skill_body=seed,
        evolved_skill_body=evolved,
        suite_version="suite-v1",
        suite_content_hash="abc",
        holdout_task_ids=["heldout-1"],
        holdout_evolved=_artifact(40.0),
        holdout_seed_baseline=_artifact(20.0),
    )


def _proposal(candidate: GepaOptimizationResult) -> ProposedAction:
    path = ".claude/skills/token/SKILL.md"
    diff = (
        "\n".join(
            difflib.unified_diff(
                candidate.seed_skill_body.splitlines(),
                candidate.evolved_skill_body.splitlines(),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
                lineterm="",
            )
        )
        + "\n"
    )
    spec = LocalApplySpec(
        target_kind=LocalApplyTargetKind.CLAUDE_SKILL,
        target_path=path,
        artifact_uri="runs:/run-1/gepa/gepa_candidate.json",
        artifact_path="gepa/gepa_candidate.json",
        baseline_sha256=_sha(candidate.seed_skill_body),
        candidate_sha256=_sha(candidate.evolved_skill_body),
        validation_command=["python", "-m", "pytest", "-q"],
        mlflow_run_id="run-1",
        reviewer_experiment_id="reviewer-1",
        holdout_savings_delta_pct=20.0,
    )
    return ProposedAction(
        proposal_id="gepa-1",
        agent_name="claude_code",
        experiment_id="subject-1",
        action_kind=ActionKind.GEPA_PROMPT,
        risk_class=RiskClass.AGENT_CHANGE,
        status=ProposalStatus.APPROVED,
        objective_metric="total_tokens",
        goal_cohort="claude_code",
        trigger=TriggerSignal(kind=TriggerKind.AGENT_PLANNER, summary="explicit GEPA"),
        change=ProposedChange(
            kind=ChangeKind.EVOLVED_BODY_REF,
            summary="rewrite local skill",
            diff=diff,
            evolved_body_ref=spec.artifact_uri,
            local_apply_spec=spec,
        ),
        proof=ProofSummary(
            objective_metric="total_tokens",
            proved_improvement=True,
            correctness_held=True,
            n_promote=1,
        ),
        gate_status=GateStatus(readiness_tier="heldout", gated=True),
    )


def _workspace(tmp_path: Path, body: str) -> tuple[Agent, Path]:
    target = tmp_path / ".claude" / "skills" / "token" / "SKILL.md"
    target.parent.mkdir(parents=True)
    target.write_text(f"---\nname: token\ndescription: test\n---\n\n{body}", encoding="utf-8")
    agent = Agent(
        agent_name="claude_code",
        experiment_id="subject-1",
        target_workspace=str(tmp_path),
    )
    return agent, target


def test_local_gepa_apply_snapshots_rewrites_validates_and_records(tmp_path: Path) -> None:
    candidate = _candidate()
    proposal = _proposal(candidate)
    agent, target = _workspace(tmp_path, candidate.seed_skill_body)
    volume = FakeVolumeClient()
    records = []

    result = apply_approved_gepa(
        proposal,
        agent,
        volume_client=volume,
        volume_root="/Volumes/cat/sch/snapshots",
        commit_recorder=records.append,
        approver="human@databricks.com",
        committed_at="2026-07-16T01:00:00+00:00",
        candidate_loader=lambda _spec: candidate,
        validator=lambda argv, cwd, timeout: (True, "2 passed"),
    )

    text = target.read_text(encoding="utf-8")
    assert text.startswith("---\nname: token\ndescription: test\n---\n\n")
    assert text.endswith(candidate.evolved_skill_body)
    assert result.lineage_recorded is True
    assert result.validation_output == "2 passed"
    assert len(records) == 1
    assert records[0].produced_change_ref == proposal.change.evolved_body_ref
    assert result.pre_change_ref.endswith("gepa-1-gepa-pre")


def test_local_gepa_apply_baseline_conflict_changes_nothing(tmp_path: Path) -> None:
    candidate = _candidate()
    proposal = _proposal(candidate)
    agent, target = _workspace(tmp_path, "# User changed this\n")
    before = target.read_bytes()
    volume = FakeVolumeClient()

    with pytest.raises(GepaApplyConflict, match="baseline hash conflict"):
        apply_approved_gepa(
            proposal,
            agent,
            volume_client=volume,
            volume_root="/Volumes/cat/sch/snapshots",
            commit_recorder=lambda _record: None,
            approver="human@databricks.com",
            committed_at="t",
            candidate_loader=lambda _spec: candidate,
            validator=lambda argv, cwd, timeout: (True, "ok"),
        )
    assert target.read_bytes() == before
    assert volume.store == {}


def test_local_gepa_apply_validation_failure_restores_snapshot(tmp_path: Path) -> None:
    candidate = _candidate()
    proposal = _proposal(candidate)
    agent, target = _workspace(tmp_path, candidate.seed_skill_body)
    before = target.read_bytes()
    records = []

    with pytest.raises(GepaValidationFailed, match="original file restored") as excinfo:
        apply_approved_gepa(
            proposal,
            agent,
            volume_client=FakeVolumeClient(),
            volume_root="/Volumes/cat/sch/snapshots",
            commit_recorder=records.append,
            approver="human@databricks.com",
            committed_at="t",
            candidate_loader=lambda _spec: candidate,
            validator=lambda argv, cwd, timeout: (False, "1 failed"),
        )
    assert excinfo.value.output == "1 failed"
    assert target.read_bytes() == before
    assert records == []
