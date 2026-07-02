"""Databricks-native versioning (Lane L6): UC-Volume snapshot/restore.

Snapshot the current bytes of an arbitrary file change-set to a UC Volume before an
executor change, and restore them on revert — no git dependency. See
:mod:`ail.versioning.snapshot` for the full contract and ``docs/VERSIONING.md``.
"""

from __future__ import annotations

from ail.versioning.snapshot import (
    MANIFEST_FILENAME,
    SNAPSHOT_TAG_PREFIX,
    VOLUME_ROOT_PREFIX,
    FileSnapshot,
    RestoreCleanupError,
    RestoreError,
    RestoreRollbackError,
    SnapshotError,
    SnapshotRef,
    SnapshotWriteError,
    VolumeClient,
    load_snapshot_ref,
    new_volume_client,
    restore_snapshot,
    snapshot_paths,
    snapshot_ref_tags,
)

__all__ = [
    "VOLUME_ROOT_PREFIX",
    "SNAPSHOT_TAG_PREFIX",
    "MANIFEST_FILENAME",
    "SnapshotError",
    "SnapshotWriteError",
    "RestoreError",
    "RestoreRollbackError",
    "RestoreCleanupError",
    "FileSnapshot",
    "SnapshotRef",
    "VolumeClient",
    "snapshot_paths",
    "restore_snapshot",
    "snapshot_ref_tags",
    "load_snapshot_ref",
    "new_volume_client",
]
