"""Fail-closed local application of an approved GEPA prompt/skill candidate.

Hosted Databricks compute owns optimization, evidence, and approval state, but it
cannot write a user's laptop. This module is the local half of that boundary: it
downloads the exact approved MLflow artifact, verifies both artifact and current
target hashes, snapshots the target, rewrites it atomically, validates it, and
rolls it back on validation failure.
"""

from __future__ import annotations

import difflib
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from ail.executor.executor import CommittedChangeRecord
from ail.loop.proposals import (
    ActionKind,
    ChangeKind,
    LocalApplySpec,
    LocalApplyTargetKind,
    ProposalStatus,
    ProposedAction,
)
from ail.optimize.gepa_runner import GepaOptimizationResult
from ail.optimize.prompt_registry import candidate_improvement
from ail.registry import Agent
from ail.versioning import SnapshotError, VolumeClient, restore_snapshot, snapshot_paths

__all__ = [
    "GepaLocalApplyError",
    "GepaApplyConflict",
    "GepaValidationFailed",
    "GepaApplyRecordError",
    "GepaLocalApplyResult",
    "apply_approved_gepa",
]


class GepaLocalApplyError(RuntimeError):
    """Base for a GEPA local-apply failure."""


class GepaApplyConflict(GepaLocalApplyError):
    """The approved artifact/target no longer matches its reviewed baseline."""


class GepaValidationFailed(GepaLocalApplyError):
    """The candidate failed validation and the original target was restored."""

    def __init__(self, message: str, *, output: str, pre_change_ref: str) -> None:
        self.output = output
        self.pre_change_ref = pre_change_ref
        super().__init__(message)


class GepaApplyRecordError(GepaLocalApplyError):
    """The validated candidate is live, but its commit audit could not be recorded."""

    def __init__(
        self,
        *,
        result: GepaLocalApplyResult,
        record: CommittedChangeRecord,
        cause: BaseException,
    ) -> None:
        self.result = result
        self.record = record
        self.cause = cause
        super().__init__(
            f"GEPA proposal {result.proposal_id!r} is live and validated, but commit "
            f"lineage failed ({type(cause).__name__}: {cause})"
        )


class CandidateLoader(Protocol):
    def __call__(self, spec: LocalApplySpec) -> GepaOptimizationResult: ...


class CommitRecorder(Protocol):
    def __call__(self, record: CommittedChangeRecord) -> None: ...


class Validator(Protocol):
    def __call__(self, argv: list[str], cwd: Path, timeout_seconds: int) -> tuple[bool, str]: ...


class GepaLocalApplyResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal_id: str
    agent_name: str
    target_workspace: str
    target_path: str
    artifact_uri: str
    pre_change_ref: str
    applied_paths: list[str] = Field(default_factory=list)
    validation_command: list[str] = Field(default_factory=list)
    validation_output: str = ""
    approver: str
    committed_at: str
    lineage_recorded: bool = False


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _default_candidate_loader(spec: LocalApplySpec) -> GepaOptimizationResult:
    import mlflow

    mlflow.set_tracking_uri("databricks")
    downloaded = Path(mlflow.artifacts.download_artifacts(artifact_uri=spec.artifact_uri))
    if downloaded.is_dir():
        downloaded = downloaded / Path(spec.artifact_path).name
    if not downloaded.is_file():
        raise GepaApplyConflict(
            f"approved MLflow artifact {spec.artifact_uri!r} did not resolve to a file"
        )
    return GepaOptimizationResult.model_validate_json(downloaded.read_text(encoding="utf-8"))


def _default_validator(argv: list[str], cwd: Path, timeout_seconds: int) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    output = "\n".join(part for part in (proc.stdout, proc.stderr) if part).strip()
    return proc.returncode == 0, output[-12000:]


def _target(root: Path, relative: str) -> Path:
    if not root.is_dir():
        raise GepaApplyConflict(f"target workspace {str(root)!r} is not a readable directory")
    target = root / relative
    try:
        resolved = target.resolve(strict=True)
    except OSError as exc:
        raise GepaApplyConflict(f"approved target {relative!r} is missing: {exc}") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise GepaApplyConflict(
            f"approved target {relative!r} resolves outside target_workspace (fail-closed)"
        ) from exc
    if not resolved.is_file():
        raise GepaApplyConflict(f"approved target {relative!r} is not a regular file")
    return resolved


def _claude_skill_parts(raw: str) -> tuple[str, str]:
    lines = raw.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        raise GepaApplyConflict("Claude skill target has no leading YAML front matter")
    closing = next((i for i, line in enumerate(lines[1:], start=1) if line.strip() == "---"), None)
    if closing is None:
        raise GepaApplyConflict("Claude skill target has unterminated YAML front matter")
    prefix = "".join(lines[: closing + 1])
    body = "".join(lines[closing + 1 :]).lstrip("\r\n")
    return prefix, body


def _render_target(kind: LocalApplyTargetKind, raw: str, candidate: str) -> tuple[str, str]:
    if kind is LocalApplyTargetKind.CLAUDE_SKILL:
        prefix, current_body = _claude_skill_parts(raw)
        return current_body, f"{prefix}\n{candidate}"
    if kind in (LocalApplyTargetKind.PROMPT_FILE, LocalApplyTargetKind.AGENTS_MD):
        return raw, candidate
    raise GepaApplyConflict(f"unsupported local target kind {kind!r}")


def _review_diff(path: str, seed: str, candidate: str) -> str:
    return "\n".join(
        difflib.unified_diff(
            seed.splitlines(),
            candidate.splitlines(),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="",
        )
    ) + "\n"


def _atomic_write(path: Path, text: str) -> None:
    mode = path.stat().st_mode
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.ail-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(text.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_name, mode)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def apply_approved_gepa(
    proposal: ProposedAction,
    agent: Agent,
    *,
    volume_client: VolumeClient,
    volume_root: str,
    commit_recorder: CommitRecorder,
    approver: str,
    committed_at: str,
    candidate_loader: CandidateLoader | None = None,
    validator: Validator | None = None,
) -> GepaLocalApplyResult:
    """Apply the exact approved GEPA artifact locally, rolling back on validation failure."""
    if proposal.status is not ProposalStatus.APPROVED:
        raise GepaApplyConflict(
            f"proposal {proposal.proposal_id!r} is not approved (fail-closed)"
        )
    if proposal.action_kind is not ActionKind.GEPA_PROMPT:
        raise GepaApplyConflict("local GEPA apply only accepts gepa_prompt proposals")
    if proposal.change.kind is not ChangeKind.EVOLVED_BODY_REF:
        raise GepaApplyConflict("GEPA proposal carries the wrong change kind")
    spec = proposal.change.local_apply_spec
    if spec is None:
        raise GepaApplyConflict("approved GEPA proposal has no immutable local_apply_spec")
    if not agent.target_workspace:
        raise GepaApplyConflict(f"agent {agent.agent_name!r} has no target_workspace")

    workspace = Path(agent.target_workspace).resolve()
    target = _target(workspace, spec.target_path)
    load = candidate_loader or _default_candidate_loader
    result = load(spec)
    improving, improvement_reason = candidate_improvement(result)
    if not improving:
        raise GepaApplyConflict(
            f"downloaded candidate no longer passes the held-out gate: {improvement_reason}"
        )
    candidate = getattr(result, spec.artifact_field, None)
    if not isinstance(candidate, str) or not candidate:
        raise GepaApplyConflict(
            f"approved artifact has no non-empty {spec.artifact_field!r} field"
        )
    if _sha256(result.seed_skill_body) != spec.baseline_sha256:
        raise GepaApplyConflict("downloaded artifact seed hash differs from the approved baseline")
    if _sha256(candidate) != spec.candidate_sha256:
        raise GepaApplyConflict("downloaded artifact candidate hash differs from the approval")
    if _review_diff(spec.target_path, result.seed_skill_body, candidate) != proposal.change.diff:
        raise GepaApplyConflict("downloaded artifact does not reproduce the exact reviewed diff")

    raw = target.read_text(encoding="utf-8")
    current_component, rewritten = _render_target(spec.target_kind, raw, candidate)
    current_hash = _sha256(current_component)
    if current_hash != spec.baseline_sha256:
        raise GepaApplyConflict(
            f"baseline hash conflict for {spec.target_path}: expected {spec.baseline_sha256}, "
            f"found {current_hash}; refresh/re-run GEPA before approving a rewrite"
        )

    pre = snapshot_paths(
        [str(target)],
        volume_root=volume_root,
        change_id=f"{proposal.proposal_id}-gepa-pre",
        client=volume_client,
    )
    _atomic_write(target, rewritten)
    validate = validator or _default_validator
    ok, output = validate(
        list(spec.validation_command), workspace, spec.validation_timeout_seconds
    )
    if not ok:
        try:
            restore_snapshot(pre, client=volume_client)
        except SnapshotError as exc:
            raise GepaLocalApplyError(
                f"validation failed and rollback failed for {spec.target_path}: {exc}; "
                f"manual recovery required from {pre.snapshot_dir}"
            ) from exc
        raise GepaValidationFailed(
            f"validation failed for {spec.target_path}; original file restored",
            output=output,
            pre_change_ref=pre.snapshot_dir,
        )

    applied_path = str(target)
    local_result = GepaLocalApplyResult(
        proposal_id=proposal.proposal_id,
        agent_name=proposal.agent_name,
        target_workspace=str(workspace),
        target_path=spec.target_path,
        artifact_uri=spec.artifact_uri,
        pre_change_ref=pre.snapshot_dir,
        applied_paths=[applied_path],
        validation_command=list(spec.validation_command),
        validation_output=output,
        approver=approver,
        committed_at=committed_at,
        lineage_recorded=False,
    )
    record = CommittedChangeRecord(
        proposal_id=proposal.proposal_id,
        agent_name=proposal.agent_name,
        target_workspace=str(workspace),
        produced_change_ref=spec.artifact_uri,
        pre_change_ref=pre.snapshot_dir,
        n_files=1,
        changed_paths=[applied_path],
        added_paths=[],
        summary=(
            f"applied approved GEPA candidate from {spec.artifact_uri} to "
            f"{spec.target_path}; validation passed: {' '.join(spec.validation_command)}"
        ),
        approver=approver,
        committed_at=committed_at,
    )
    try:
        commit_recorder(record)
    except Exception as exc:  # noqa: BLE001 - live + validated, audit needs reconciliation
        raise GepaApplyRecordError(result=local_result, record=record, cause=exc) from exc
    return local_result.model_copy(update={"lineage_recorded": True})
