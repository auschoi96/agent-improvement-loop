"""Tests for the L6 UC-Volume snapshot/restore versioning (``ail.versioning``).

Every test here is **offline**: the Volume client is replaced by an injected
:class:`FakeVolumeClient` that stores bytes in a dict, so no live Databricks call is
ever made (no ``live`` marker). The one test that exercises the default SDK-backed
client (:class:`_FilesVolumeClient`) uses a fake workspace object exposing the
Files API surface, proving the seam wiring without a network call.

The suite proves each fail-closed / integrity property the L6 contract requires:

* ``test_snapshot_restore_roundtrip_byte_identical`` — restored bytes == snapshotted
  bytes exactly, across text / binary / unicode / empty content.
* ``test_snapshot_partial_write_failure_raises_and_returns_no_ref`` and
  ``test_snapshot_source_file_unreadable_raises`` — a snapshot that cannot fully
  persist raises and yields no ref.
* ``test_restore_missing_object_raises_and_writes_nothing`` and
  ``test_restore_corrupt_object_raises_and_writes_nothing`` — restore verifies
  everything first; a missing/corrupt object raises and the local tree is untouched.
* ``test_core_api_makes_no_live_calls_only_injected_client`` — all Volume I/O goes
  through the injected fake; the core functions have no implicit live path.
"""

from __future__ import annotations

import hashlib
import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from ail.versioning import (
    MANIFEST_FILENAME,
    SNAPSHOT_TAG_PREFIX,
    FileSnapshot,
    RestoreError,
    SnapshotRef,
    SnapshotWriteError,
    load_snapshot_ref,
    restore_snapshot,
    snapshot_paths,
    snapshot_ref_tags,
)
from ail.versioning.snapshot import _FilesVolumeClient

VOLUME_ROOT = "/Volumes/cat/sch/vol/ail_snapshots"


# ---------------------------------------------------------------------------
# Fakes (in-memory; record calls — no live Databricks I/O)
# ---------------------------------------------------------------------------


@dataclass
class FakeVolumeClient:
    """In-memory stand-in for a UC Volume: a ``{volume_path: bytes}`` store."""

    store: dict[str, bytes] = field(default_factory=dict)
    upload_calls: list[str] = field(default_factory=list)
    download_calls: list[str] = field(default_factory=list)
    #: Volume paths whose upload should fail (simulate a rejected/failed write).
    fail_upload_on: set[str] = field(default_factory=set)

    def upload(self, volume_path: str, contents: bytes) -> None:
        self.upload_calls.append(volume_path)
        if volume_path in self.fail_upload_on:
            raise PermissionError("PERMISSION_DENIED: missing WRITE_VOLUME grant")
        self.store[volume_path] = bytes(contents)

    def download(self, volume_path: str) -> bytes:
        self.download_calls.append(volume_path)
        if volume_path not in self.store:
            raise FileNotFoundError(f"NOT_FOUND: {volume_path}")
        return self.store[volume_path]


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _sample_files(root: Path) -> dict[Path, bytes]:
    """A change-set spanning text, binary, unicode, empty, and a nested path."""
    files = {
        root / "a.py": b"print('hello')\n",
        root / "pkg" / "b.txt": "café — naïve résumé\n".encode(),
        root / "bin.dat": bytes(range(256)),
        root / "empty": b"",
        root / "pkg" / "deep" / "c.md": b"line1\nline2\n",
    }
    for path, data in files.items():
        _write(path, data)
    return files


# ---------------------------------------------------------------------------
# Round-trip integrity
# ---------------------------------------------------------------------------


def test_snapshot_restore_roundtrip_byte_identical(tmp_path: Path) -> None:
    files = _sample_files(tmp_path)
    client = FakeVolumeClient()

    ref = snapshot_paths(files.keys(), volume_root=VOLUME_ROOT, change_id="chg-1", client=client)

    assert isinstance(ref, SnapshotRef)
    assert ref.snapshot_dir == f"{VOLUME_ROOT}/chg-1"
    assert ref.manifest_path == f"{VOLUME_ROOT}/chg-1/{MANIFEST_FILENAME}"
    assert len(ref.files) == len(files)
    # The manifest was persisted as the completing write.
    assert ref.manifest_path in client.store
    # Every manifest entry hashes/sizes the true source bytes.
    by_abspath = {entry.original_path: entry for entry in ref.files}
    for path, data in files.items():
        entry = by_abspath[os.path.abspath(str(path))]
        assert entry.sha256 == hashlib.sha256(data).hexdigest()
        assert entry.size == len(data)

    # Mutate every file and delete one, then restore -> exact original bytes back.
    for path in files:
        path.write_bytes(b"CORRUPTED-BY-EXECUTOR")
    (tmp_path / "a.py").unlink()

    restore_snapshot(ref, client=client)

    for path, data in files.items():
        assert path.read_bytes() == data


def test_restore_recreates_a_deleted_file(tmp_path: Path) -> None:
    target = _write(tmp_path / "nested" / "deep" / "f.txt", b"original\n")
    client = FakeVolumeClient()
    ref = snapshot_paths([target], volume_root=VOLUME_ROOT, change_id="chg-2", client=client)

    # Remove the whole subtree the file lived in.
    target.unlink()
    target.parent.rmdir()

    restore_snapshot(ref, client=client)
    assert target.read_bytes() == b"original\n"


def test_identical_content_dedupes_to_one_blob(tmp_path: Path) -> None:
    same = b"identical bytes\n"
    f1 = _write(tmp_path / "one.txt", same)
    f2 = _write(tmp_path / "two.txt", same)
    client = FakeVolumeClient()

    ref = snapshot_paths([f1, f2], volume_root=VOLUME_ROOT, change_id="dedupe", client=client)

    blob_uploads = [p for p in client.upload_calls if "/blobs/" in p]
    assert len(blob_uploads) == 1  # deduped: one blob for identical content
    assert ref.files[0].volume_path == ref.files[1].volume_path

    f1.write_bytes(b"x")
    f2.write_bytes(b"y")
    restore_snapshot(ref, client=client)
    assert f1.read_bytes() == same
    assert f2.read_bytes() == same


# ---------------------------------------------------------------------------
# Fail-closed: snapshot that cannot fully persist raises + returns no ref
# ---------------------------------------------------------------------------


def test_snapshot_partial_write_failure_raises_and_returns_no_ref(tmp_path: Path) -> None:
    files = _sample_files(tmp_path)
    client = FakeVolumeClient()
    # Fail the manifest write specifically: blobs land, but the snapshot is NOT complete.
    client.fail_upload_on = {f"{VOLUME_ROOT}/chg/{MANIFEST_FILENAME}"}

    with pytest.raises(SnapshotWriteError) as exc:
        snapshot_paths(files.keys(), volume_root=VOLUME_ROOT, change_id="chg", client=client)

    assert "WRITE_VOLUME" in str(exc.value)
    # No completed snapshot: the manifest (the completion marker) is absent.
    assert f"{VOLUME_ROOT}/chg/{MANIFEST_FILENAME}" not in client.store


def test_snapshot_blob_write_failure_raises(tmp_path: Path) -> None:
    f1 = _write(tmp_path / "a.txt", b"aaa")
    client = FakeVolumeClient()
    digest = hashlib.sha256(b"aaa").hexdigest()
    client.fail_upload_on = {f"{VOLUME_ROOT}/chg/blobs/{digest}"}

    with pytest.raises(SnapshotWriteError):
        snapshot_paths([f1], volume_root=VOLUME_ROOT, change_id="chg", client=client)
    # The manifest was never even attempted after the blob failure.
    assert f"{VOLUME_ROOT}/chg/{MANIFEST_FILENAME}" not in client.store


def test_snapshot_source_file_unreadable_raises(tmp_path: Path) -> None:
    good = _write(tmp_path / "good.txt", b"ok")
    missing = tmp_path / "does-not-exist.txt"
    client = FakeVolumeClient()

    with pytest.raises(SnapshotWriteError) as exc:
        snapshot_paths([good, missing], volume_root=VOLUME_ROOT, change_id="chg", client=client)
    assert "not a readable file" in str(exc.value)


def test_snapshot_source_directory_raises(tmp_path: Path) -> None:
    a_dir = tmp_path / "subdir"
    a_dir.mkdir()
    client = FakeVolumeClient()
    with pytest.raises(SnapshotWriteError):
        snapshot_paths([a_dir], volume_root=VOLUME_ROOT, change_id="chg", client=client)


# ---------------------------------------------------------------------------
# Fail-closed: restore verifies EVERYTHING before writing anything
# ---------------------------------------------------------------------------


def test_restore_missing_object_raises_and_writes_nothing(tmp_path: Path) -> None:
    files = _sample_files(tmp_path)
    client = FakeVolumeClient()
    ref = snapshot_paths(files.keys(), volume_root=VOLUME_ROOT, change_id="chg", client=client)

    # A snapshotted blob vanishes from the Volume (partial loss).
    del client.store[ref.files[0].volume_path]

    # Mutate the local tree so we can prove restore wrote nothing.
    for path in files:
        path.write_bytes(b"LOCAL-STATE")

    with pytest.raises(RestoreError) as exc:
        restore_snapshot(ref, client=client)
    assert "missing or unreadable" in str(exc.value)

    for path in files:
        assert path.read_bytes() == b"LOCAL-STATE"  # nothing restored


def test_restore_corrupt_object_raises_and_writes_nothing(tmp_path: Path) -> None:
    files = _sample_files(tmp_path)
    client = FakeVolumeClient()
    ref = snapshot_paths(files.keys(), volume_root=VOLUME_ROOT, change_id="chg", client=client)

    # Corrupt one blob's bytes so its sha256/size no longer match the manifest.
    victim = ref.files[1].volume_path
    client.store[victim] = client.store[victim] + b"tampered"

    for path in files:
        path.write_bytes(b"LOCAL-STATE")

    with pytest.raises(RestoreError) as exc:
        restore_snapshot(ref, client=client)
    assert "corrupt" in str(exc.value)

    for path in files:
        assert path.read_bytes() == b"LOCAL-STATE"  # nothing restored


def test_restore_empty_ref_raises() -> None:
    empty = SnapshotRef(
        change_id="x",
        volume_root=VOLUME_ROOT,
        snapshot_dir=f"{VOLUME_ROOT}/x",
        manifest_path=f"{VOLUME_ROOT}/x/{MANIFEST_FILENAME}",
        files=[],
        created_at="2026-07-02T00:00:00+00:00",
    )
    with pytest.raises(RestoreError):
        restore_snapshot(empty, client=FakeVolumeClient())


def test_restore_detects_size_mismatch(tmp_path: Path) -> None:
    """Different-length bytes at the blob path are caught as corrupt."""
    f1 = _write(tmp_path / "a.txt", b"abc")
    client = FakeVolumeClient()
    ref = snapshot_paths([f1], volume_root=VOLUME_ROOT, change_id="chg", client=client)
    client.store[ref.files[0].volume_path] = b"abcd"
    with pytest.raises(RestoreError, match="corrupt"):
        restore_snapshot(ref, client=client)


# ---------------------------------------------------------------------------
# Injected client: the core has no implicit live path
# ---------------------------------------------------------------------------


def test_core_api_makes_no_live_calls_only_injected_client(tmp_path: Path) -> None:
    files = _sample_files(tmp_path)
    client = FakeVolumeClient()

    ref = snapshot_paths(files.keys(), volume_root=VOLUME_ROOT, change_id="chg", client=client)
    for path in files:
        path.write_bytes(b"changed")
    restore_snapshot(ref, client=client)

    # Every Volume touch went through the injected fake — proof there is no hidden
    # live path in the core snapshot/restore functions.
    assert client.upload_calls, "all writes must go through the injected client"
    assert client.download_calls, "all reads must go through the injected client"
    assert all(p.startswith(VOLUME_ROOT) for p in client.upload_calls)
    assert all(p.startswith(VOLUME_ROOT) for p in client.download_calls)


def test_core_functions_require_explicit_client(tmp_path: Path) -> None:
    """``client`` is keyword-only and required — no default that could go live."""
    f1 = _write(tmp_path / "a.txt", b"a")
    with pytest.raises(TypeError):
        snapshot_paths([f1], volume_root=VOLUME_ROOT, change_id="chg")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Input validation (fail-closed on bad inputs)
# ---------------------------------------------------------------------------


def test_empty_path_set_raises() -> None:
    with pytest.raises(ValueError, match="at least one path"):
        snapshot_paths([], volume_root=VOLUME_ROOT, change_id="chg", client=FakeVolumeClient())


@pytest.mark.parametrize("bad_id", ["", "../evil", "a/b", ".hidden", "with space", ".."])
def test_bad_change_id_raises(tmp_path: Path, bad_id: str) -> None:
    f1 = _write(tmp_path / "a.txt", b"a")
    with pytest.raises(ValueError, match="change_id"):
        snapshot_paths([f1], volume_root=VOLUME_ROOT, change_id=bad_id, client=FakeVolumeClient())


@pytest.mark.parametrize("bad_root", ["/tmp/snapshots", "relative/path", "/Volumes", "/Volumes/"])
def test_non_volume_root_raises(tmp_path: Path, bad_root: str) -> None:
    f1 = _write(tmp_path / "a.txt", b"a")
    with pytest.raises(ValueError, match="UC Volume path"):
        snapshot_paths([f1], volume_root=bad_root, change_id="chg", client=FakeVolumeClient())


def test_trailing_slash_root_normalized(tmp_path: Path) -> None:
    f1 = _write(tmp_path / "a.txt", b"a")
    client = FakeVolumeClient()
    ref = snapshot_paths([f1], volume_root=f"{VOLUME_ROOT}/", change_id="chg", client=client)
    assert ref.volume_root == VOLUME_ROOT  # trailing slash stripped
    assert ref.snapshot_dir == f"{VOLUME_ROOT}/chg"


# ---------------------------------------------------------------------------
# Recording the ref (additive pointer) + reading it back on revert
# ---------------------------------------------------------------------------


def test_snapshot_ref_tags_are_additive_pointer(tmp_path: Path) -> None:
    f1 = _write(tmp_path / "a.txt", b"a")
    ref = snapshot_paths([f1], volume_root=VOLUME_ROOT, change_id="chg", client=FakeVolumeClient())

    tags = snapshot_ref_tags(ref)
    assert all(k.startswith(f"{SNAPSHOT_TAG_PREFIX}.") for k in tags)
    assert all(isinstance(v, str) for v in tags.values())  # MLflow tags are strings
    assert tags[f"{SNAPSHOT_TAG_PREFIX}.snapshot_dir"] == ref.snapshot_dir
    assert tags[f"{SNAPSHOT_TAG_PREFIX}.manifest_path"] == ref.manifest_path
    assert tags[f"{SNAPSHOT_TAG_PREFIX}.n_files"] == "1"
    # Only the pointer is recorded — the (potentially large) manifest is NOT inlined.
    assert not any("sha256" in k or "original_path" in k for k in tags)


def test_load_snapshot_ref_reconstructs_and_reverts(tmp_path: Path) -> None:
    """Pointer -> reconstruct-from-Volume -> restore: the full L7 revert loop, offline."""
    files = _sample_files(tmp_path)
    client = FakeVolumeClient()
    ref = snapshot_paths(files.keys(), volume_root=VOLUME_ROOT, change_id="chg", client=client)

    # A downstream revert has only the recorded pointer (snapshot_dir).
    pointer = snapshot_ref_tags(ref)[f"{SNAPSHOT_TAG_PREFIX}.snapshot_dir"]
    reloaded = load_snapshot_ref(pointer, client=client)
    assert reloaded == ref

    for path in files:
        path.write_bytes(b"changed")
    restore_snapshot(reloaded, client=client)
    for path, data in files.items():
        assert path.read_bytes() == data


def test_load_snapshot_ref_missing_manifest_raises() -> None:
    with pytest.raises(RestoreError, match="missing or unreadable"):
        load_snapshot_ref(f"{VOLUME_ROOT}/nope", client=FakeVolumeClient())


def test_load_snapshot_ref_corrupt_manifest_raises() -> None:
    client = FakeVolumeClient()
    client.store[f"{VOLUME_ROOT}/x/{MANIFEST_FILENAME}"] = b"not json at all"
    with pytest.raises(RestoreError, match="not a valid SnapshotRef"):
        load_snapshot_ref(f"{VOLUME_ROOT}/x", client=client)


# ---------------------------------------------------------------------------
# Default SDK-backed client wiring (offline: fake workspace object)
# ---------------------------------------------------------------------------


@dataclass
class _FakeDownloadResponse:
    contents: Any


@dataclass
class _FakeFilesApi:
    uploaded: dict[str, bytes] = field(default_factory=dict)
    upload_overwrite: list[bool] = field(default_factory=list)

    def upload(self, file_path: str, contents: Any, *, overwrite: bool = False) -> None:
        self.upload_overwrite.append(overwrite)
        self.uploaded[file_path] = contents.read()

    def download(self, file_path: str) -> _FakeDownloadResponse:
        return _FakeDownloadResponse(contents=io.BytesIO(self.uploaded[file_path]))


@dataclass
class _FakeWorkspace:
    files: _FakeFilesApi = field(default_factory=_FakeFilesApi)


def test_files_volume_client_delegates_to_sdk_files_api() -> None:
    ws = _FakeWorkspace()
    client = _FilesVolumeClient(ws)

    client.upload("/Volumes/c/s/v/x", b"payload")
    assert ws.files.uploaded["/Volumes/c/s/v/x"] == b"payload"
    assert ws.files.upload_overwrite == [True]  # snapshots always overwrite

    assert client.download("/Volumes/c/s/v/x") == b"payload"


def test_files_volume_client_end_to_end_offline(tmp_path: Path) -> None:
    """The default client, backed by a fake Files API, satisfies the same round-trip."""
    files = _sample_files(tmp_path)
    client = _FilesVolumeClient(_FakeWorkspace())

    ref = snapshot_paths(files.keys(), volume_root=VOLUME_ROOT, change_id="chg", client=client)
    for path in files:
        path.write_bytes(b"changed")
    restore_snapshot(ref, client=client)
    for path, data in files.items():
        assert path.read_bytes() == data


def test_file_snapshot_contract_is_strict() -> None:
    with pytest.raises(ValidationError):
        FileSnapshot(  # type: ignore[call-arg]
            original_path="/a", volume_path="/b", sha256="x", size=1, extra="nope"
        )
