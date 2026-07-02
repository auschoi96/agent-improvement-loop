"""Lane L7b-2 — the open-ended Claude Agent SDK executor (the safety wrapper).

The **executor** carries out an approved open-ended ``AGENT_TASK`` proposal (L7b-1,
:mod:`ail.loop.proposals`) by running a local Claude Agent SDK agent to make
*arbitrary* changes to the target agent's own source — new/edited skills, tools,
tables, metric views, cached examples, multi-file refactors, whatever the agent
decides (``docs/PRODUCT_ARCHITECTURE.md`` §7). It is **Databricks-native, no git**:
the produced change-set is versioned/revertible via the L6 UC-Volume snapshot
substrate (:mod:`ail.versioning.snapshot`), never a git tree.

The safety lives in the *wrapper*, split into two clearly-separated, independently
testable halves so a reviewer can scrutinize each (``docs/EXECUTOR.md``):

* :func:`produce_preview` — **pre-approval, no live effect.** Copies the target
  workspace into an isolated sandbox, runs the agent **in the copy only**, captures
  the concrete produced change as a ``preview_diff`` + an L6 snapshot
  (``produced_change_ref``), and records both onto the proposal so the app shows the
  human the *real* diff. The live workspace is **never touched** (byte-for-byte
  untouched even while the agent runs).
* :func:`commit_approved` — **post-approval, live.** Applies the **stored** produced
  change-set (the exact one the human saw) — it **never re-runs the agent** (the SDK
  is non-deterministic; a re-run could ship a *different* change than was approved).
  It snapshots the live workspace **first** (the revert point), then applies the
  stored change via L6's all-or-nothing restore, then records the commit.

Everything is **fail-closed** and driven through **injectable seams** — the agent
runner, the UC-Volume client, and the preview/commit persistence — so the whole
module imports offline and every test runs with fakes (no live SDK / MLflow /
Databricks call).

The Databricks-native L6 substrate versions file **writes** (add/modify) — its
restore recreates files but cannot *delete* them — so a produced change that
**deletes** files is refused fail-closed here (git-backed executor is future work;
``docs/PRODUCT_ARCHITECTURE.md`` §8's "coarser than git" tradeoff).
"""

from __future__ import annotations

import difflib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from ail.ingest.adapters.claude_code import ClaudeCodeAdapter
from ail.ingest.base import AgentRunResult, AgentTask
from ail.loop.proposals import ActionKind, ChangeKind, ProposalStatus, ProposedAction
from ail.registry import Agent
from ail.versioning import (
    FileSnapshot,
    SnapshotError,
    SnapshotRef,
    VolumeClient,
    load_snapshot_ref,
    restore_snapshot,
    snapshot_paths,
)

__all__ = [
    "EXECUTOR_SYSTEM_PROMPT",
    "DEFAULT_TIMEOUT_SECONDS",
    "ExecutorError",
    "PreviewError",
    "CommitRefused",
    "CommitRecordError",
    "AgentRunner",
    "PreviewWriter",
    "CommitRecorder",
    "FileChange",
    "CommittedChangeRecord",
    "PreviewResult",
    "CommitResult",
    "produce_preview",
    "commit_approved",
]

#: The system prompt framing the sandboxed agent as the executor: it must carry out
#: the approved plan by editing files in its working directory, nothing more.
EXECUTOR_SYSTEM_PROMPT = (
    "You are the change executor for an agent-improvement loop. Carry out the "
    "approved change described in the user message by editing files IN YOUR CURRENT "
    "WORKING DIRECTORY only. Make exactly the change the plan describes — do not add "
    "unrelated edits. Your working directory is an isolated sandbox copy; make the "
    "edits directly to the files there."
)

#: Generous default per-run ceiling for the sandboxed agent (an open-ended change can
#: span many files). The adapter also enforces its own hard timeout on top.
DEFAULT_TIMEOUT_SECONDS = 900

#: Directory / file names never copied into the sandbox nor treated as part of a
#: change-set: VCS metadata and tool caches (the target workspace is Databricks-native,
#: so ``.git`` is irrelevant, and caches are never a real source change). Applied
#: identically to the copy and to both sides of the change diff so they can never
#: register as an add/modify/delete.
_IGNORED_DIR_NAMES = frozenset(
    {".git", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", "node_modules", ".venv"}
)


def _is_ignored_file(name: str) -> bool:
    return name.endswith(".pyc") or name == ".DS_Store"


# ---------------------------------------------------------------------------
# Errors (fail-closed: a failure is never a returned partial success)
# ---------------------------------------------------------------------------


class ExecutorError(RuntimeError):
    """Base for an executor failure."""


class PreviewError(ExecutorError):
    """:func:`produce_preview` failed fail-closed — **no preview was written**.

    Raised for an unmet precondition (not an ``AGENT_TASK``, no plan, unset/unreadable
    ``target_workspace``), an agent error, a change that produced nothing, a change
    that deletes files (the L6 substrate cannot apply a deletion), or a snapshot that
    could not persist. In every case the live ``target_workspace`` is left
    byte-for-byte untouched (the agent only ever edits the sandbox copy) and the
    proposal carries no fabricated preview.
    """


class CommitRefused(ExecutorError):
    """:func:`commit_approved` refused fail-closed — **nothing was applied**.

    Raised when a precondition is unmet: the proposal is not APPROVED, is not an
    ``AGENT_TASK``, carries no ``produced_change_ref`` (the human never approved a
    concrete diff), the target workspace is unset/unreadable, or the stored
    change-set is missing/unreadable or targets a path outside the workspace.
    """


class CommitRecordError(ExecutorError):
    """The change WAS committed (live) but recording it to the timeline failed.

    Cross-system atomicity between applying the change and recording it is impossible,
    so the invariant is **fail-loud, never silently inconsistent**: once the change is
    applied live it stays applied. It carries the applied :class:`CommitResult`
    (``lineage_recorded=False``) so the caller surfaces *committed-but-unrecorded,
    reconcile* rather than rolling a live change back into a fake not-applied state.
    """

    def __init__(
        self, *, result: CommitResult, record: CommittedChangeRecord, cause: BaseException
    ) -> None:
        self.result = result
        self.record = record
        self.cause = cause
        super().__init__(
            f"proposal {result.proposal_id!r} was COMMITTED ({result.n_files} file(s) applied to "
            f"{result.target_workspace!r}) but recording it failed ({type(cause).__name__}: "
            f"{cause}) — the change is LIVE and the audit record must be reconciled "
            "(committed-but-unrecorded)."
        )


# ---------------------------------------------------------------------------
# Injectable seams (faked in tests → no live SDK / MLflow / Databricks call)
# ---------------------------------------------------------------------------


class AgentRunner(Protocol):
    """Runs the Claude Agent SDK against an :class:`~ail.ingest.base.AgentTask`.

    The exact surface of :class:`~ail.ingest.base.AgentAdapter`, so the shipped
    :class:`~ail.ingest.adapters.claude_code.ClaudeCodeAdapter` satisfies it directly
    (the default). Injected so a test can supply a fake that edits the sandbox and
    returns a synthetic result — no live SDK/model call. **Only** :func:`produce_preview`
    takes a runner; :func:`commit_approved` has none (it never re-runs the agent).
    """

    def run(self, task: AgentTask) -> AgentRunResult: ...


class PreviewWriter(Protocol):
    """Persists a produced preview back onto the proposal (its ``agent_proposed_actions``
    row) so the app shows the human the real diff. Injected so the core stays offline;
    the companion runner supplies a live UC ``UPDATE``.
    """

    def __call__(
        self, *, agent_name: str, proposal_id: str, preview_diff: str, produced_change_ref: str
    ) -> None: ...


class CommitRecorder(Protocol):
    """Records one committed change onto the audit timeline (the snapshot refs + summary
    + approver). Injected so the core stays offline; the companion runner persists it.
    """

    def __call__(self, record: CommittedChangeRecord) -> None: ...


# ---------------------------------------------------------------------------
# Typed results / records (pydantic, extra='forbid' — the repo's convention)
# ---------------------------------------------------------------------------


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FileChange(_Model):
    """One file the sandboxed agent added or modified (relative to the workspace root)."""

    path: str
    change_type: str  # "added" | "modified"


class CommittedChangeRecord(_Model):
    """The audit record of one committed open-ended change — *what, from where, by whom*.

    Handed to the :class:`CommitRecorder` seam. The two snapshot refs are load-bearing:
    ``produced_change_ref`` is exactly the previewed change the human approved, and
    ``pre_change_ref`` is the pre-commit snapshot of the live workspace — the revert
    point (revert = L6 restore of it). ``pre_change_ref`` is ``None`` only when the
    change is pure additions (no pre-existing files were overwritten, so there was
    nothing to snapshot).
    """

    proposal_id: str
    agent_name: str
    target_workspace: str
    produced_change_ref: str
    pre_change_ref: str | None = None
    n_files: int
    changed_paths: list[str] = Field(default_factory=list)
    summary: str
    approver: str
    committed_at: str


class PreviewResult(_Model):
    """The outcome of :func:`produce_preview` — the previewed change, ready for review."""

    proposal: ProposedAction
    produced_change_ref: str
    preview_diff: str
    changes: list[FileChange] = Field(default_factory=list)
    n_files: int


class CommitResult(_Model):
    """The outcome of :func:`commit_approved` — what was applied live."""

    proposal_id: str
    agent_name: str
    target_workspace: str
    produced_change_ref: str
    pre_change_ref: str | None = None
    applied_paths: list[str] = Field(default_factory=list)
    n_files: int
    approver: str
    committed_at: str
    lineage_recorded: bool = False


# ---------------------------------------------------------------------------
# (1) produce_preview — PRE-APPROVAL, no live effect
# ---------------------------------------------------------------------------


def produce_preview(
    proposal: ProposedAction,
    agent: Agent,
    *,
    volume_client: VolumeClient,
    volume_root: str,
    preview_writer: PreviewWriter,
    agent_runner: AgentRunner | None = None,
    sandbox_root: str | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    allowed_tools: list[str] | None = None,
    model: str | None = None,
) -> PreviewResult:
    """Produce the concrete preview of an ``AGENT_TASK`` proposal — no live effect.

    Copies ``agent.target_workspace`` into an isolated sandbox, runs the Claude Agent
    SDK agent **in the copy only** (cwd = the sandbox, edit tools allowed, the
    proposal's ``plan`` as the prompt), captures the produced change as a
    ``preview_diff`` and an L6 snapshot (``produced_change_ref``), and records both
    onto the proposal via ``preview_writer`` so the human reviews the real change.

    Fail-closed — on any of the following it writes **no** preview, raises
    :class:`PreviewError`, and leaves the live ``target_workspace`` byte-for-byte
    untouched (the agent only ever edits the sandbox copy):

    * the proposal is not an ``AGENT_TASK`` / carries no ``plan``;
    * ``target_workspace`` is unset or not a readable directory;
    * the agent run errors;
    * the agent produced no change;
    * the change deletes files (the L6 restore substrate cannot apply a deletion);
    * the snapshot could not fully persist.

    Args:
        proposal: The pending ``AGENT_TASK`` proposal (its ``plan`` drives the agent).
        agent: The registry entry — ``agent.target_workspace`` is the tree edited.
        volume_client: UC-Volume seam for the L6 snapshot (injected; no implicit live).
        volume_root: UC-Volume dir under ``/Volumes/…`` to snapshot the produced change into.
        preview_writer: Persists the ``preview_diff`` + ``produced_change_ref`` (called
            only after the snapshot succeeds).
        agent_runner: The SDK seam (defaults to a fresh :class:`ClaudeCodeAdapter`).
        sandbox_root: Optional explicit sandbox dir (a fresh temp dir is made and
            removed otherwise). When given it is left in place for the caller.
        timeout_seconds: Per-run ceiling passed to the agent task.
        allowed_tools: Tools allowed in the sandbox (``None`` → the adapter's default
            edit tool set).
        model: Optional model override for the agent run.

    Returns:
        The :class:`PreviewResult` with the updated proposal (its change now carrying
        ``preview_diff`` + ``produced_change_ref``).
    """
    if proposal.action_kind is not ActionKind.AGENT_TASK:
        raise PreviewError(
            f"produce_preview only handles AGENT_TASK proposals; proposal "
            f"{proposal.proposal_id!r} is {proposal.action_kind.value!r} (fail-closed)"
        )
    plan = proposal.change.plan
    if not plan or not plan.strip():
        raise PreviewError(
            f"AGENT_TASK proposal {proposal.proposal_id!r} carries no plan to execute (fail-closed)"
        )
    workspace = _resolve_workspace(agent, error=PreviewError)

    owns_sandbox = sandbox_root is None
    sandbox = Path(sandbox_root) if sandbox_root is not None else Path(_new_sandbox_dir())
    try:
        _copy_workspace(workspace, sandbox)

        runner = agent_runner if agent_runner is not None else _default_agent_runner()
        task = AgentTask(
            prompt=plan,
            system_prompt=EXECUTOR_SYSTEM_PROMPT,
            model=model,
            allowed_tools=allowed_tools,
            cwd=str(sandbox),
            timeout_seconds=timeout_seconds,
        )
        result = runner.run(task)
        if not result.success:
            raise PreviewError(
                f"the executor agent failed on proposal {proposal.proposal_id!r} "
                f"({result.error or 'no error detail'}) — writing no preview (fail-closed)"
            )

        changes, deleted = _diff_trees(workspace, sandbox)
        if deleted:
            raise PreviewError(
                f"the executor deleted {len(deleted)} file(s) ({', '.join(sorted(deleted)[:5])}"
                f"{'…' if len(deleted) > 5 else ''}) for proposal {proposal.proposal_id!r}; the "
                "Databricks-native L6 snapshot substrate cannot apply a file deletion on commit — "
                "writing no preview (fail-closed)"
            )
        if not changes:
            raise PreviewError(
                f"the executor produced no change for proposal {proposal.proposal_id!r} — writing "
                "no preview (never a fabricated preview) (fail-closed)"
            )

        produced_ref = _snapshot_produced(
            changes,
            sandbox_root=sandbox,
            live_root=workspace,
            volume_root=volume_root,
            change_id=proposal.proposal_id,
            client=volume_client,
        )
        preview_diff = _render_preview_diff(changes, live_root=workspace, sandbox_root=sandbox)
    finally:
        if owns_sandbox:
            shutil.rmtree(sandbox, ignore_errors=True)

    snapshot_dir = produced_ref.snapshot_dir
    updated_change = proposal.change.model_copy(
        update={"preview_diff": preview_diff, "produced_change_ref": snapshot_dir}
    )
    updated = proposal.model_copy(update={"change": updated_change})

    # Record the preview LAST: the snapshot has persisted, so the proposal now points at
    # a real, committable change-set. A record failure leaves the live tree untouched
    # (nothing was ever applied); the next run re-previews (ref still unset) idempotently.
    try:
        preview_writer(
            agent_name=proposal.agent_name,
            proposal_id=proposal.proposal_id,
            preview_diff=preview_diff,
            produced_change_ref=snapshot_dir,
        )
    except Exception as exc:  # noqa: BLE001 - surface honestly; live tree is untouched
        raise PreviewError(
            f"preview produced for proposal {proposal.proposal_id!r} (snapshot at "
            f"{snapshot_dir!r}) but recording it onto the proposal failed "
            f"({type(exc).__name__}: {exc}); the live workspace is untouched (fail-closed)"
        ) from exc

    return PreviewResult(
        proposal=updated,
        produced_change_ref=snapshot_dir,
        preview_diff=preview_diff,
        changes=changes,
        n_files=len(changes),
    )


# ---------------------------------------------------------------------------
# (2) commit_approved — POST-APPROVAL, live
# ---------------------------------------------------------------------------


def commit_approved(
    proposal: ProposedAction,
    agent: Agent,
    *,
    volume_client: VolumeClient,
    volume_root: str,
    commit_recorder: CommitRecorder,
    approver: str,
    committed_at: str,
) -> CommitResult:
    """Commit the **stored** produced change of an approved ``AGENT_TASK`` — live.

    **The load-bearing safety invariant:** this applies the change-set recorded at
    ``proposal.change.produced_change_ref`` — the *exact* one the human previewed and
    approved. It **never re-runs the agent** (the SDK is non-deterministic; a re-run
    could produce a different change than was approved). There is deliberately no
    agent-runner parameter here.

    Order (fail-closed): snapshot the live workspace **first** (the revert point), then
    apply the stored change via L6's all-or-nothing restore, then record the commit —
    so a failed apply cannot leave a half-applied tree (L6's restore verifies every
    object before writing any and rolls back a mid-swap failure), and the revert point
    exists before anything live changes.

    Fail-closed preconditions (any unmet ⇒ :class:`CommitRefused`, nothing applied):

    * the proposal is APPROVED and is an ``AGENT_TASK``;
    * it carries a ``produced_change_ref`` (the concrete diff the human approved);
    * ``target_workspace`` is set and a readable directory;
    * the stored change-set is readable and every target path is inside the workspace.

    Raises:
        CommitRefused: any precondition unmet (nothing applied).
        CommitRecordError: the change WAS applied live but recording it failed.
        SnapshotError: the pre-change snapshot or the restore failed (L6 leaves the
            live tree restorable/untouched); surfaced unchanged — never a fake commit.
    """
    if proposal.status is not ProposalStatus.APPROVED:
        raise CommitRefused(
            f"proposal {proposal.proposal_id!r} is {proposal.status.value!r}, not approved — "
            "refusing to commit an un-approved change (fail-closed)"
        )
    if proposal.action_kind is not ActionKind.AGENT_TASK:
        raise CommitRefused(
            f"commit_approved only handles AGENT_TASK proposals; proposal "
            f"{proposal.proposal_id!r} is {proposal.action_kind.value!r} (fail-closed)"
        )
    if proposal.change.kind is not ChangeKind.AGENT_TASK_PLAN:
        raise CommitRefused(
            f"proposal {proposal.proposal_id!r} carries change kind "
            f"{proposal.change.kind.value!r}, not an agent_task_plan (fail-closed)"
        )
    produced_ref_dir = proposal.change.produced_change_ref
    if not produced_ref_dir or not produced_ref_dir.strip():
        raise CommitRefused(
            f"approved proposal {proposal.proposal_id!r} carries no produced_change_ref — the "
            "human approved no concrete diff (missing/stale preview); refusing (fail-closed)"
        )
    workspace = _resolve_workspace(agent, error=CommitRefused)

    # Load the STORED produced change-set (never re-run the agent).
    try:
        produced = load_snapshot_ref(produced_ref_dir, client=volume_client)
    except SnapshotError as exc:
        raise CommitRefused(
            f"the produced change-set for proposal {proposal.proposal_id!r} at "
            f"{produced_ref_dir!r} is missing or unreadable ({exc}) — refusing (fail-closed)"
        ) from exc

    targets = [f.original_path for f in produced.files]
    _require_paths_within(targets, workspace, proposal_id=proposal.proposal_id)

    # 1) Snapshot the LIVE workspace FIRST — the revert point. Only the pre-existing
    #    files the change will overwrite are captured (added files have no pre-state; a
    #    pure-add change has no revert point, recorded honestly as pre_change_ref=None).
    existing = [p for p in targets if Path(p).is_file()]
    pre_change_ref: SnapshotRef | None = None
    if existing:
        pre_change_ref = snapshot_paths(
            existing,
            volume_root=volume_root,
            change_id=f"{proposal.proposal_id}-pre",
            client=volume_client,
        )

    # 2) Apply the STORED change to live — all-or-nothing (L6 restore verifies every
    #    object before writing any; a failure leaves the live tree restorable/untouched).
    restore_snapshot(produced, client=volume_client)

    # 3) Record the commit. The change is now LIVE; a record failure is surfaced
    #    fail-loud as committed-but-unrecorded, never rolled back into a fake refusal.
    applied_paths = sorted(targets)
    result = CommitResult(
        proposal_id=proposal.proposal_id,
        agent_name=proposal.agent_name,
        target_workspace=str(workspace),
        produced_change_ref=produced_ref_dir,
        pre_change_ref=pre_change_ref.snapshot_dir if pre_change_ref is not None else None,
        applied_paths=applied_paths,
        n_files=len(applied_paths),
        approver=approver,
        committed_at=committed_at,
        lineage_recorded=False,
    )
    record = CommittedChangeRecord(
        proposal_id=proposal.proposal_id,
        agent_name=proposal.agent_name,
        target_workspace=str(workspace),
        produced_change_ref=produced_ref_dir,
        pre_change_ref=pre_change_ref.snapshot_dir if pre_change_ref is not None else None,
        n_files=len(applied_paths),
        changed_paths=applied_paths,
        summary=(
            f"committed approved AGENT_TASK {proposal.proposal_id}: {len(applied_paths)} file(s) "
            f"applied to {workspace}"
        ),
        approver=approver,
        committed_at=committed_at,
    )
    try:
        commit_recorder(record)
    except Exception as exc:  # noqa: BLE001 - fail loud: the change is live, the record is not
        raise CommitRecordError(result=result, record=record, cause=exc) from exc

    return result.model_copy(update={"lineage_recorded": True})


# ---------------------------------------------------------------------------
# Workspace / sandbox helpers (pure; only ever read live, only ever write sandbox)
# ---------------------------------------------------------------------------


def _resolve_workspace(agent: Agent, *, error: type[ExecutorError]) -> Path:
    """Resolve + validate ``agent.target_workspace`` (required for the executor)."""
    ws = agent.target_workspace
    if not ws or not ws.strip():
        raise error(
            f"agent {agent.agent_name!r} has no target_workspace configured — the executor "
            "cannot run against an agent with no target workspace (fail-closed)"
        )
    path = Path(ws).expanduser()
    if not path.is_dir():
        raise error(
            f"target_workspace {ws!r} for agent {agent.agent_name!r} is not a readable directory "
            "(fail-closed)"
        )
    return path.resolve()


def _new_sandbox_dir() -> str:
    return tempfile.mkdtemp(prefix="ail-executor-sandbox-")


def _copytree_ignore(_dir: str, names: list[str]) -> set[str]:
    return {n for n in names if n in _IGNORED_DIR_NAMES or _is_ignored_file(n)}


def _copy_workspace(live_root: Path, sandbox: Path) -> None:
    """Copy the live workspace into ``sandbox`` (VCS/caches excluded, no symlink follow)."""
    # dirs_exist_ok so an explicit, pre-created sandbox_root is accepted.
    shutil.copytree(live_root, sandbox, ignore=_copytree_ignore, symlinks=True, dirs_exist_ok=True)


def _iter_relpaths(root: Path) -> set[str]:
    """Relative paths of every (non-ignored) file under ``root``.

    Ignores the same VCS/cache names as the sandbox copy, applied to both trees, so an
    ignored path can never register as an add/modify/delete.
    """
    out: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _IGNORED_DIR_NAMES]
        for fn in filenames:
            if _is_ignored_file(fn):
                continue
            out.add(str((Path(dirpath) / fn).relative_to(root)))
    return out


def _diff_trees(live_root: Path, sandbox_root: Path) -> tuple[list[FileChange], list[str]]:
    """Compare the sandbox against the live tree.

    Returns ``(changes, deleted)`` where ``changes`` are the added/modified files
    (sorted, relative) and ``deleted`` are files present live but gone from the sandbox.
    """
    before = _iter_relpaths(live_root)
    after = _iter_relpaths(sandbox_root)
    added = after - before
    deleted = sorted(before - after)
    modified = {
        rel
        for rel in (before & after)
        if (live_root / rel).read_bytes() != (sandbox_root / rel).read_bytes()
    }
    changes = [FileChange(path=rel, change_type="added") for rel in sorted(added)]
    changes += [FileChange(path=rel, change_type="modified") for rel in sorted(modified)]
    changes.sort(key=lambda c: c.path)
    return changes, deleted


def _snapshot_produced(
    changes: list[FileChange],
    *,
    sandbox_root: Path,
    live_root: Path,
    volume_root: str,
    change_id: str,
    client: VolumeClient,
) -> SnapshotRef:
    """Snapshot the produced (post-edit) bytes, keyed to the **live** paths.

    Snapshots the sandbox files' produced bytes via :func:`ail.versioning.snapshot_paths`
    (content-addressed blobs), then remaps each manifest entry's ``original_path`` from
    the sandbox back to the live workspace and re-writes the manifest, so a later
    :func:`ail.versioning.restore_snapshot` at commit writes the produced bytes to the
    **live** paths (the blobs are content-addressed, so they are valid regardless of
    which path they were read from). ``produced_change_ref`` is the returned
    ``snapshot_dir``; :func:`ail.versioning.load_snapshot_ref` reads the remapped
    manifest back on commit.
    """
    # Match snapshot_paths' own os.path.abspath (which does NOT resolve symlinks) so the
    # relpath below yields the bare relative path; live_root is already .resolve()d by the
    # caller, so the remapped live paths match the workspace commit validates against.
    sandbox_abs = os.path.abspath(str(sandbox_root))
    live_abs = str(live_root)
    sandbox_paths = [str(sandbox_root / c.path) for c in changes]
    ref = snapshot_paths(sandbox_paths, volume_root=volume_root, change_id=change_id, client=client)

    remapped_files: list[FileSnapshot] = []
    for f in ref.files:
        rel = os.path.relpath(f.original_path, sandbox_abs)
        remapped_files.append(
            FileSnapshot(
                original_path=os.path.join(live_abs, rel),
                volume_path=f.volume_path,
                sha256=f.sha256,
                size=f.size,
            )
        )
    remapped = ref.model_copy(update={"files": remapped_files})
    # Overwrite the manifest snapshot_paths just wrote with the live-path manifest, so
    # load_snapshot_ref(snapshot_dir) on commit returns the committable (live-path) ref.
    client.upload(remapped.manifest_path, remapped.model_dump_json(indent=2).encode("utf-8"))
    return remapped


def _render_preview_diff(changes: list[FileChange], *, live_root: Path, sandbox_root: Path) -> str:
    """A human-readable unified diff of the produced change (for review, not for apply).

    The concrete change the human reviews. The *apply* uses the snapshotted bytes, not
    this diff, so a binary file is rendered as a marker rather than a byte diff.
    """
    parts: list[str] = []
    for change in changes:
        rel = change.path
        old = b"" if change.change_type == "added" else (live_root / rel).read_bytes()
        new = (sandbox_root / rel).read_bytes()
        parts.append(_render_file_diff(rel, old, new, added=change.change_type == "added"))
    return "".join(parts)


def _render_file_diff(rel: str, old: bytes, new: bytes, *, added: bool) -> str:
    old_text = _decode(old)
    new_text = _decode(new)
    header = f"### {'added' if added else 'modified'}: {rel}\n"
    if old_text is None or new_text is None:
        return f"{header}Binary file changed ({len(old)} -> {len(new)} bytes)\n\n"
    body = list(
        difflib.unified_diff(
            old_text.splitlines(),
            new_text.splitlines(),
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
            lineterm="",
        )
    )
    return header + ("\n".join(body) + "\n\n" if body else "(no textual change)\n\n")


def _decode(data: bytes) -> str | None:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _require_paths_within(paths: list[str], workspace: Path, *, proposal_id: str) -> None:
    """Refuse (fail-closed) if any stored target path is outside ``workspace``.

    Defends the live commit against a tampered manifest: a restore writes to each
    entry's ``original_path``, so every one must resolve inside the workspace or the
    commit would write outside the agent's own source.
    """
    root = str(workspace)
    for p in paths:
        ap = os.path.abspath(p)
        if ap != root and not ap.startswith(root + os.sep):
            raise CommitRefused(
                f"the produced change-set for proposal {proposal_id!r} targets {ap!r}, outside "
                f"the target_workspace {root!r} — refusing to write outside the workspace "
                "(fail-closed)"
            )


def _default_agent_runner(mlflow_experiment: str | None = None) -> AgentRunner:
    """Build the default live runner — a :class:`ClaudeCodeAdapter`.

    Built only when :func:`produce_preview` is called with no injected runner; the
    adapter lazy-imports the Claude Agent SDK, so importing this module stays offline.
    """
    return ClaudeCodeAdapter(mlflow_experiment=mlflow_experiment)
