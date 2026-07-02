"""Lane L6 — Databricks-native versioning for arbitrary file / code change-sets.

The open-ended executor (L7) makes arbitrary changes to improve an agent. Each
applied change must be **versioned and revertible**, Databricks-native, with no git
dependency (``docs/PRODUCT_ARCHITECTURE.md`` §8). The three change types map to
three revert mechanisms:

* prompt / skill / instruction -> MLflow Prompt Registry (new version + champion
  alias; re-point the alias to revert). Owned by
  :mod:`ail.optimize.prompt_registry` — reused, not reimplemented.
* UC asset (metric view / table / function) -> created in UC; revert = drop/recreate.
* **arbitrary file / code change-set -> UC-Volume snapshot/restore.** This module.

What it does
------------
:func:`snapshot_paths` copies the *current bytes* of a set of files to a UC Volume
location keyed by ``change_id`` and returns a :class:`SnapshotRef` (the Volume
location + a per-file manifest of path + sha256 + size). :func:`restore_snapshot`
restores the exact snapshotted bytes back to their original paths. Together they
make an executor change to arbitrary files fully revertible and auditable.

Fail-closed by construction (this framework's whole point is anti-fake-good)
----------------------------------------------------------------------------
* **A snapshot that cannot be fully written raises** — a missing ``WRITE_VOLUME``
  grant, an unreachable Volume, or an unreadable source file all raise
  :class:`SnapshotWriteError`. A :class:`SnapshotRef` is returned **only** after
  every blob *and* the manifest have persisted, so a change is never reported
  "versioned / revertible" when the snapshot did not persist.
* **Restore is all-or-nothing.** Every manifested file is downloaded and its
  sha256 + size verified against the manifest *before any bytes are written back*;
  a missing or corrupt object raises :class:`RestoreError` and nothing is written
  (no half-reverted tree). The verified bytes are then staged to sibling temp files
  and swapped in, so a mid-restore local I/O error still cannot leave a partially
  reverted set.

Injectable client (no live Databricks call on import or in tests)
-----------------------------------------------------------------
All Volume I/O goes through the small :class:`VolumeClient` Protocol (mirrors the
seam in :mod:`ail.optimize.prompt_registry` / :mod:`ail.loop.apply_service`). The
core :func:`snapshot_paths` / :func:`restore_snapshot` require an explicit
``client`` — there is no implicit live path — so tests inject a fake and this module
never touches Databricks unless a caller builds a live client with
:func:`new_volume_client`.

Recording the ref (additive; no DDL / column change)
----------------------------------------------------
:func:`snapshot_ref_tags` renders the ref as ``ail.snapshot.*`` pointer tags
(mirroring :meth:`ail.optimize.prompt_registry.PromptProvenance.as_tags`), so the
L7 apply path can stamp the *pointer* onto the applied change's existing record (a
registered version's tags or a decision-audit field) with no schema change. Only
the pointer is recorded; the full per-file manifest lives in the Volume
(``manifest.json``) and is read back on revert via :func:`load_snapshot_ref`. This
module deliberately does **not** wire snapshot/restore into the apply path — that
is L7's job; L6 provides the capability.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import tempfile
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

__all__ = [
    "VOLUME_ROOT_PREFIX",
    "SNAPSHOT_TAG_PREFIX",
    "MANIFEST_FILENAME",
    "SnapshotError",
    "SnapshotWriteError",
    "RestoreError",
    "FileSnapshot",
    "SnapshotRef",
    "VolumeClient",
    "snapshot_paths",
    "restore_snapshot",
    "snapshot_ref_tags",
    "load_snapshot_ref",
    "new_volume_client",
]

#: A UC Volume path always lives under ``/Volumes/<catalog>/<schema>/<volume>/…``.
#: :func:`snapshot_paths` fails closed if ``volume_root`` is not one — writing a
#: "snapshot" to a non-Volume path would be a lie about being Databricks-native.
VOLUME_ROOT_PREFIX = "/Volumes/"

#: Namespace prefix for the additive pointer tags :func:`snapshot_ref_tags` emits
#: (``ail.snapshot.<field>``). Deliberately distinct from ``ail.prompt.*`` (owned by
#: :mod:`ail.optimize.prompt_registry`) so the two provenance schemas never collide.
SNAPSHOT_TAG_PREFIX = "ail.snapshot"

#: Leaf name of the self-describing manifest written into each snapshot directory.
MANIFEST_FILENAME = "manifest.json"

_BLOBS_DIRNAME = "blobs"
#: A ``change_id`` keys a Volume directory, so it must be a safe slug: alphanumeric
#: start, then alphanumerics / dot / dash / underscore — no ``/`` and no ``..``.
_CHANGE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


# ---------------------------------------------------------------------------
# Errors (fail-closed: a failure is never a returned partial success)
# ---------------------------------------------------------------------------


class SnapshotError(RuntimeError):
    """Base for a snapshot/restore failure."""


class SnapshotWriteError(SnapshotError):
    """A snapshot could not be *fully* written — no :class:`SnapshotRef` is returned.

    Raised when a source file is unreadable, the Volume is unreachable, or a write
    is rejected (e.g. a missing ``WRITE_VOLUME`` grant). Never returned as a partial
    snapshot: a change must never be reported versioned/revertible when the snapshot
    did not persist.
    """


class RestoreError(SnapshotError):
    """A restore could not be completed safely — nothing is written back.

    Raised when a manifested object is missing or its bytes do not match the
    manifest's sha256/size (corrupt). Verification of *every* file completes before
    any write, so this leaves the local tree untouched.
    """


# ---------------------------------------------------------------------------
# Typed contracts (pydantic, extra='forbid' — the repo's contract convention)
# ---------------------------------------------------------------------------


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FileSnapshot(_Contract):
    """One snapshotted file's manifest entry: where it came from, where it is, its hash."""

    original_path: str
    volume_path: str
    sha256: str
    size: int


class SnapshotRef(_Contract):
    """A completed snapshot: the Volume location + the per-file manifest.

    Returned by :func:`snapshot_paths` **only** once every blob and the manifest have
    persisted. ``snapshot_dir`` is the recordable pointer (see
    :func:`snapshot_ref_tags`); ``manifest_path`` is the self-describing manifest read
    back by :func:`load_snapshot_ref` on revert. ``files`` is what
    :func:`restore_snapshot` verifies and restores.
    """

    change_id: str
    volume_root: str
    snapshot_dir: str
    manifest_path: str
    files: list[FileSnapshot]
    created_at: str


# ---------------------------------------------------------------------------
# The injectable Volume seam (faked in tests → no live Databricks call)
# ---------------------------------------------------------------------------


class VolumeClient(Protocol):
    """The slice of UC Volume file I/O this module needs.

    The default implementation (:class:`_FilesVolumeClient`) delegates to the
    Databricks SDK Files API against a UC Volume; tests inject a fake exposing these
    two methods so no live call is ever made. Both must **raise** on failure — a
    silent partial write/read would defeat the fail-closed contract.
    """

    def upload(self, volume_path: str, contents: bytes) -> None:
        """Write ``contents`` to ``volume_path``, overwriting; raise on any failure."""
        ...

    def download(self, volume_path: str) -> bytes:
        """Read the bytes at ``volume_path``; raise if missing or unreadable."""
        ...


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


def snapshot_paths(
    paths: Iterable[str | Path],
    *,
    volume_root: str,
    change_id: str,
    client: VolumeClient,
    created_at: str | None = None,
) -> SnapshotRef:
    """Snapshot the current bytes of ``paths`` to a UC Volume, keyed by ``change_id``.

    Reads each file's bytes, content-addresses them (a blob per sha256 under
    ``<volume_root>/<change_id>/blobs/``), writes a self-describing ``manifest.json``,
    and returns a :class:`SnapshotRef`. Content addressing dedupes identical files and
    makes each blob name its own integrity check.

    Fail-closed: any unreadable source, an unreachable Volume, or a rejected write
    raises :class:`SnapshotWriteError`; the ref is constructed and returned **only**
    after every blob and the manifest have persisted, so a partial/failed snapshot
    never yields a ref. (A failed snapshot may leave orphan blobs no ref points at —
    harmless, and overwritten if the same ``change_id`` is re-snapshotted.)

    Args:
        paths: The files to snapshot (absolute or relative; resolved and de-duplicated).
        volume_root: A UC Volume directory under ``/Volumes/…`` to snapshot into.
        change_id: A safe slug keying this snapshot's directory (no ``/`` or ``..``).
        client: Injectable Volume client — the only path to the Volume (no live call
            unless the caller passes a live one from :func:`new_volume_client`).
        created_at: Optional ISO timestamp override (defaults to now, UTC).

    Returns:
        The :class:`SnapshotRef` describing the persisted snapshot.

    Raises:
        ValueError: for an empty path set, a bad ``change_id``, or a non-Volume root.
        SnapshotWriteError: if any source is unreadable or any Volume write fails.
    """
    root = _validate_volume_root(volume_root)
    _validate_change_id(change_id)
    resolved = _dedupe_paths(paths)
    if not resolved:
        raise ValueError("snapshot_paths requires at least one path; got an empty set")

    snapshot_dir = f"{root}/{change_id}"
    blobs_dir = f"{snapshot_dir}/{_BLOBS_DIRNAME}"
    manifest_path = f"{snapshot_dir}/{MANIFEST_FILENAME}"

    entries: list[FileSnapshot] = []
    uploaded: set[str] = set()
    for abspath in resolved:
        data = _read_source(abspath)
        digest = hashlib.sha256(data).hexdigest()
        blob_path = f"{blobs_dir}/{digest}"
        if blob_path not in uploaded:
            _upload(client, blob_path, data)
            uploaded.add(blob_path)
        entries.append(
            FileSnapshot(
                original_path=abspath,
                volume_path=blob_path,
                sha256=digest,
                size=len(data),
            )
        )

    ref = SnapshotRef(
        change_id=change_id,
        volume_root=root,
        snapshot_dir=snapshot_dir,
        manifest_path=manifest_path,
        files=entries,
        created_at=created_at or datetime.now(UTC).isoformat(),
    )
    # The manifest is written LAST: its presence is what makes the snapshot complete
    # and self-describing. If this write fails, we raise and return no ref.
    _upload(client, manifest_path, ref.model_dump_json(indent=2).encode("utf-8"))
    return ref


def _read_source(abspath: str) -> bytes:
    """Read a source file's bytes, or fail closed if it is not a readable file."""
    path = Path(abspath)
    if not path.is_file():
        raise SnapshotWriteError(
            f"cannot snapshot {abspath!r}: not a readable file "
            "(missing, a directory, or a broken symlink)"
        )
    try:
        return path.read_bytes()
    except OSError as exc:
        raise SnapshotWriteError(
            f"cannot read source file {abspath!r} for snapshot: {exc}"
        ) from exc


def _upload(client: VolumeClient, volume_path: str, data: bytes) -> None:
    """Write one object to the Volume, wrapping any failure as :class:`SnapshotWriteError`."""
    try:
        client.upload(volume_path, data)
    except Exception as exc:
        raise SnapshotWriteError(
            f"failed to write snapshot object to {volume_path!r} "
            f"(Volume unreachable or missing WRITE_VOLUME grant?): {type(exc).__name__}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Restore (verify EVERYTHING before writing anything back)
# ---------------------------------------------------------------------------


def restore_snapshot(ref: SnapshotRef, *, client: VolumeClient) -> None:
    """Restore the exact snapshotted bytes in ``ref`` to their original paths.

    All-or-nothing: every manifested object is downloaded and its sha256 + size
    verified against the manifest **before any byte is written back**; a missing or
    corrupt object raises :class:`RestoreError` and the local tree is left untouched.
    The verified bytes are then staged to sibling temp files and swapped in with
    ``os.replace``, so even a mid-restore local I/O error cannot leave a half-reverted
    tree. Restoring a file whose original path no longer exists recreates it (and any
    missing parent directories).

    Raises:
        RestoreError: if the ref is empty, or any object is missing/corrupt, or the
            verified bytes cannot be staged.
    """
    if not ref.files:
        raise RestoreError("snapshot ref carries no files; nothing to restore (invalid/empty ref)")

    # Phase 1 — download + verify every file. Nothing local is touched here.
    verified: list[tuple[str, bytes]] = []
    cache: dict[str, bytes] = {}
    for entry in ref.files:
        data = cache.get(entry.volume_path)
        if data is None:
            data = _download(client, entry)
            cache[entry.volume_path] = data
        _verify(entry, data)
        verified.append((entry.original_path, data))

    # Phase 2 — verification passed for ALL files: stage each to a sibling temp, then
    # swap them in. Staging fully before the first swap keeps a mid-restore I/O error
    # from leaving a partially reverted set.
    staged: list[tuple[str, str]] = []  # (tmp_path, target_path)
    try:
        for original_path, data in verified:
            staged.append((_stage(Path(original_path), data), original_path))
    except OSError as exc:
        for tmp, _ in staged:
            _silent_unlink(tmp)
        raise RestoreError(f"failed to stage restored files (nothing written back): {exc}") from exc
    for tmp, target in staged:
        os.replace(tmp, target)


def _download(client: VolumeClient, entry: FileSnapshot) -> bytes:
    """Fetch one snapshotted object, wrapping a missing/unreadable object as a raise."""
    try:
        return client.download(entry.volume_path)
    except Exception as exc:
        raise RestoreError(
            f"snapshotted object for {entry.original_path!r} is missing or unreadable at "
            f"{entry.volume_path!r} (refusing to write a half-reverted tree): "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _verify(entry: FileSnapshot, data: bytes) -> None:
    """Fail closed unless ``data`` matches the manifest's sha256 *and* size."""
    actual = hashlib.sha256(data).hexdigest()
    if len(data) != entry.size or actual != entry.sha256:
        raise RestoreError(
            f"snapshotted object for {entry.original_path!r} at {entry.volume_path!r} is corrupt: "
            f"expected sha256 {entry.sha256} size {entry.size}, "
            f"got sha256 {actual} size {len(data)}"
        )


def _stage(path: Path, data: bytes) -> str:
    """Write ``data`` to a sibling temp file (creating parents); return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".ail-restore")
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
    return tmp


def _silent_unlink(tmp: str) -> None:
    """Best-effort cleanup of a staged temp file; never masks the original error."""
    try:
        os.unlink(tmp)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _validate_volume_root(volume_root: str) -> str:
    """Return the trailing-slash-trimmed root, or raise if it is not a UC Volume path."""
    root = volume_root.rstrip("/")
    if not root.startswith(VOLUME_ROOT_PREFIX) or root == VOLUME_ROOT_PREFIX.rstrip("/"):
        raise ValueError(
            f"volume_root must be a UC Volume path under {VOLUME_ROOT_PREFIX!r} "
            "(e.g. /Volumes/<catalog>/<schema>/<volume>/ail_snapshots); "
            f"got {volume_root!r}"
        )
    return root


def _validate_change_id(change_id: str) -> None:
    """Raise unless ``change_id`` is a safe directory slug (no ``/``, no ``..``)."""
    if not _CHANGE_ID_RE.match(change_id):
        raise ValueError(
            f"change_id must be a non-empty slug matching {_CHANGE_ID_RE.pattern!r} "
            "(no '/' or '..' — it keys a Volume directory); "
            f"got {change_id!r}"
        )


def _dedupe_paths(paths: Iterable[str | Path]) -> list[str]:
    """Resolve to absolute paths, de-duplicated, preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for path in paths:
        abspath = os.path.abspath(os.fspath(path))
        if abspath not in seen:
            seen.add(abspath)
            out.append(abspath)
    return out


# ---------------------------------------------------------------------------
# Recording the ref (additive pointer) + reading it back on revert
# ---------------------------------------------------------------------------


def snapshot_ref_tags(ref: SnapshotRef) -> dict[str, str]:
    """Render ``ref`` as additive ``ail.snapshot.*`` pointer tags (no DDL/column change).

    Mirrors :meth:`ail.optimize.prompt_registry.PromptProvenance.as_tags`: the L7
    apply path stamps these onto the applied change's *existing* record (a registered
    version's tags or a decision-audit field), so the lineage / revert surface can
    find the snapshot without any schema change. Only the **pointer** is recorded —
    the full per-file manifest lives in the Volume (``manifest_path``) and is read
    back via :func:`load_snapshot_ref` on revert, so a large change-set never has to
    be squeezed into a tag or column.
    """
    return {
        f"{SNAPSHOT_TAG_PREFIX}.change_id": ref.change_id,
        f"{SNAPSHOT_TAG_PREFIX}.snapshot_dir": ref.snapshot_dir,
        f"{SNAPSHOT_TAG_PREFIX}.manifest_path": ref.manifest_path,
        f"{SNAPSHOT_TAG_PREFIX}.n_files": str(len(ref.files)),
        f"{SNAPSHOT_TAG_PREFIX}.created_at": ref.created_at,
    }


def load_snapshot_ref(snapshot_dir: str, *, client: VolumeClient) -> SnapshotRef:
    """Reconstruct a :class:`SnapshotRef` from its ``manifest.json`` in the Volume.

    The revert counterpart to :func:`snapshot_ref_tags`: given only the recorded
    pointer (``snapshot_dir``), read the manifest back and return the full ref that
    :func:`restore_snapshot` needs. Fail-closed: a missing or unparseable manifest
    raises :class:`RestoreError` rather than returning a partial ref.
    """
    manifest_path = f"{snapshot_dir.rstrip('/')}/{MANIFEST_FILENAME}"
    try:
        raw = client.download(manifest_path)
    except Exception as exc:
        raise RestoreError(
            f"snapshot manifest missing or unreadable at {manifest_path!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    try:
        return SnapshotRef.model_validate_json(raw.decode("utf-8"))
    except Exception as exc:
        raise RestoreError(
            f"snapshot manifest at {manifest_path!r} is not a valid SnapshotRef: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Live client (built only when a caller passes no injected client)
# ---------------------------------------------------------------------------


class _FilesVolumeClient:
    """Default :class:`VolumeClient` delegating to the Databricks SDK Files API."""

    def __init__(self, workspace_client: Any) -> None:
        self._ws = workspace_client

    def upload(self, volume_path: str, contents: bytes) -> None:
        self._ws.files.upload(volume_path, io.BytesIO(contents), overwrite=True)

    def download(self, volume_path: str) -> bytes:
        response = self._ws.files.download(volume_path)
        data = response.contents.read()
        return data if isinstance(data, bytes) else bytes(data)


def new_volume_client(profile: str | None = None) -> VolumeClient:
    """Build a live :class:`VolumeClient` against UC Volumes.

    Uses the static-token-matched-to-host auth pattern by reusing
    :func:`ail.publish._build_workspace_client`: a PAT in ``DATABRICKS_HOST`` /
    ``DATABRICKS_TOKEN`` is preferred, else the CLI ``profile`` — never a ``--profile``
    OAuth refresh for long-running work (the concern documented in
    :func:`ail.jobs.publish_job.resolve_job_auth`). A missing ``WRITE_VOLUME`` grant
    is a deploy-time prerequisite; it surfaces as an honest :class:`SnapshotWriteError`
    on first write, never a fake success. Built **only** when a caller wants live I/O;
    never touched on import or in tests.
    """
    from ail.publish import _build_workspace_client

    return _FilesVolumeClient(_build_workspace_client(profile))
