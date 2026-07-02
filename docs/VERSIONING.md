# UC-Volume snapshot/restore (arbitrary file change-sets)

Databricks-native versioning for the **third** change type in
`docs/PRODUCT_ARCHITECTURE.md` §8: an executor change to **arbitrary files / code**.
Prompt/skill changes version in the MLflow Prompt Registry and UC-asset changes
revert by drop/recreate; a change to a set of arbitrary files versions by
**snapshotting its current bytes to a UC Volume before the change and restoring them
on revert** — no git dependency.

- **Module:** `src/ail/versioning/` (offline-tested; the Volume client is injected —
  no live Databricks call on import or in tests)
- **Storage:** a UC Volume path under `/Volumes/<catalog>/<schema>/<volume>/…`
- **Revert primitive:** restore the snapshot (whole change-set, byte-for-byte)

| Change | Mechanism | Revert |
|---|---|---|
| Prompt / skill / instruction | MLflow Prompt Registry (`ail.optimize.prompt_registry`) | re-point the champion alias |
| UC asset (metric view / table / function) | created directly in UC | drop / recreate |
| **Arbitrary file / code change-set** | **UC-Volume snapshot** (this module) | **restore the snapshot** |

**Tradeoff (from the architecture doc):** a Volume snapshot is coarser than git —
whole-set restore, not line-level diff/revert. Git can be added later as an *option*
for heavy code edits; the Databricks-native snapshot is the default.

## API

```python
from ail.versioning import snapshot_paths, restore_snapshot, new_volume_client

client = new_volume_client(profile="dais-demo")  # or inject your own VolumeClient

# Before an executor change: snapshot the files it will touch.
ref = snapshot_paths(
    ["src/agent/tools.py", "skills/plan.md"],
    volume_root="/Volumes/austin_choi_omni_agent_catalog/agent_improvement_loop/snapshots",
    change_id="chg-2026-07-02-abc123",
    client=client,
)

# ... executor mutates those files ...

# On revert: restore the exact snapshotted bytes to their original paths.
restore_snapshot(ref, client=client)
```

- `snapshot_paths(paths, *, volume_root, change_id, client) -> SnapshotRef` — reads
  each file's current bytes, content-addresses them (one blob per sha256 under
  `<volume_root>/<change_id>/blobs/`), writes a self-describing `manifest.json`, and
  returns a `SnapshotRef`.
- `restore_snapshot(ref, *, client) -> None` — verifies **every** manifested object
  against its sha256 + size, then restores the exact bytes to their original paths.
- `SnapshotRef` (pydantic, `extra="forbid"`) carries the Volume location
  (`snapshot_dir`, `manifest_path`) and a per-file manifest — each `FileSnapshot`
  is `original_path` + `volume_path` + `sha256` + `size`.

## Layout in the Volume

```
<volume_root>/<change_id>/
    manifest.json          # the full SnapshotRef (JSON) — the completing write
    blobs/<sha256>         # one content-addressed blob per unique file body
```

Content addressing dedupes identical files and makes each blob name its own
integrity check. The manifest is written **last**; its presence marks the snapshot
complete.

## Fail-closed contract (this framework's whole point is anti-fake-good)

- **A snapshot that cannot be fully written raises `SnapshotWriteError`** — an
  unreadable source file, an unreachable Volume, or a rejected write (e.g. a missing
  `WRITE_VOLUME` grant). A `SnapshotRef` is returned **only** after every blob *and*
  the manifest persist, so a change is never reported "versioned / revertible" when
  the snapshot did not persist. (A failed snapshot may leave orphan blobs no ref
  points at — harmless, and overwritten if the same `change_id` is re-snapshotted.)
- **Restore is transactional — never a silent partial.** Every manifested object is
  downloaded and its sha256 + size verified against the manifest **before any byte is
  written back**; a missing or corrupt object raises `RestoreError` and the local
  tree is left untouched. The verified bytes are then **staged reversibly** (a staging
  failure removes its temps *and* any newly-created directories, leaving the tree
  byte-for-byte unchanged), then swapped in with `os.replace`. True cross-file
  atomicity of N renames is not physically guaranteed on POSIX, so the swap is made
  **recoverable**: the pre-restore bytes of every target are captured first, and if an
  `os.replace` fails mid-swap the already-swapped files are rolled back to that state
  (raising `RestoreError`, tree == pre-restore). If the rollback itself fails, a
  distinct, loud `RestoreRollbackError` names exactly which files hold restored /
  rolled-back / original content — so a mid-restore I/O error is never a silent
  half-reverted tree.
- **Round-trip integrity:** restored bytes equal snapshotted bytes exactly (tested
  across text / binary / unicode / empty content).

`WRITE_VOLUME` on the target Volume is a **deploy-time prerequisite**; its absence
surfaces as an honest `SnapshotWriteError` on the first write, never a fake success.

## Auth

`new_volume_client()` builds a live client using the **static-token-matched-to-host**
pattern (reuses `ail.publish._build_workspace_client`): a PAT in `DATABRICKS_HOST` /
`DATABRICKS_TOKEN` is preferred, otherwise the CLI `profile` — never a `--profile`
OAuth refresh for long-running work (the concern documented in
`ail.jobs.publish_job.resolve_job_auth`). The core `snapshot_paths` /
`restore_snapshot` take an **injectable** `VolumeClient` Protocol (mirroring the
seams in `ail.optimize.prompt_registry` and `ail.loop.apply_service`), so tests and
import make no live call; the live client is built only when a caller asks for it.

## Recording the ref for the lineage / revert surface (additive, no DDL)

The snapshot ref is recorded **additively** — no schema/column change to the
existing lineage table (`ail.publish_lineage`):

- `snapshot_ref_tags(ref) -> dict[str, str]` renders the ref as `ail.snapshot.*`
  **pointer** tags (mirroring `PromptProvenance.as_tags`'s `ail.prompt.*` convention).
  The L7 apply path stamps these onto the applied change's *existing* record — a
  registered version's tags or a decision-audit field — with no DDL. Only the
  pointer is recorded, so a large change-set never has to be squeezed into a tag or
  column.
- `load_snapshot_ref(snapshot_dir, *, client) -> SnapshotRef` reconstructs the full
  ref from the Volume's `manifest.json` given only the recorded pointer — the revert
  counterpart, fail-closed on a missing/unparseable manifest.

## Scope

This module provides the **capability + tests**. The executor wiring that actually
calls `snapshot_paths` before a change and `restore_snapshot`/`load_snapshot_ref` on
revert is **L7's** job — it is deliberately not wired into the apply path here.

## Tests

`tests/test_versioning.py` — all offline (an in-memory `FakeVolumeClient`; the
default SDK-backed client is exercised against a fake workspace object). Proves:
round-trip byte-identical; snapshot partial-write / unreadable-source failure raises
and returns no ref; restore of a missing/corrupt object raises and writes nothing;
all Volume I/O goes through the injected client (no implicit live path); input
validation (empty set, bad `change_id`, non-Volume root); dedup; deleted-file
recreate; and the `snapshot_ref_tags` → `load_snapshot_ref` → `restore_snapshot`
revert loop.
