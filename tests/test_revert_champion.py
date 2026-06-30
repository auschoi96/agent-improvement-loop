"""Unit tests for the guarded champion-revert CLI (:mod:`ail.jobs.revert_champion`).

No network/registry access: the lineage registry seam is faked. These cover the
guard contract — fail-closed on an unknown version/agent, the WAS->BECOMES audit
line, dry-run by default, and the alias write only on ``--yes``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

import ail.jobs.revert_champion as rc
from ail.jobs.revert_champion import EXIT_REFUSED, main, revert_champion

CATALOG = "cat"
SCHEMA = "sch"
LEAF = "token_efficient_execution"
FULL = f"{CATALOG}.{SCHEMA}.{LEAF}"


@dataclass
class FakeVersion:
    version: int
    uri: str


@dataclass
class FakeClient:
    versions: list[FakeVersion] = field(default_factory=list)
    alias_to_version: dict[str, int] = field(default_factory=dict)
    set_alias_calls: list[tuple[str, str, int]] = field(default_factory=list)

    def search_prompt_versions(self, name: str) -> list[FakeVersion]:
        return list(self.versions)

    def get_prompt_version_by_alias(self, name: str, alias: str) -> FakeVersion | None:
        v = self.alias_to_version.get(alias)
        if v is None:
            return None
        return next((fv for fv in self.versions if fv.version == v), None)

    def set_prompt_alias(self, name: str, alias: str, version: int) -> None:
        self.set_alias_calls.append((name, alias, version))


def _client() -> FakeClient:
    return FakeClient(
        versions=[FakeVersion(1, "prompts:/p/1"), FakeVersion(2, "prompts:/p/2")],
        alias_to_version={"champion": 2},  # current champion is v2
    )


def _run(client: FakeClient, *, to_version: int, apply: bool) -> tuple[int, list[str]]:
    lines: list[str] = []
    code = revert_champion(
        agent_name="claude_code",
        to_version=to_version,
        client=client,
        prompt_name=LEAF,
        catalog=CATALOG,
        schema=SCHEMA,
        apply=apply,
        out=lines.append,
    )
    return code, lines


# ---------------------------------------------------------------------------
# Dry run (default)
# ---------------------------------------------------------------------------


def test_dry_run_prints_plan_and_writes_nothing() -> None:
    client = _client()
    code, lines = _run(client, to_version=1, apply=False)
    assert code == 0
    assert client.set_alias_calls == []  # no write in a dry run
    text = "\n".join(lines)
    assert "DRY RUN" in text
    # explicit WAS -> BECOMES audit, with version + uri on each side
    assert "champion WAS : v2 (prompts:/p/2)" in text
    assert "champion BECOMES: v1 (prompts:/p/1)" in text


# ---------------------------------------------------------------------------
# Apply (--yes)
# ---------------------------------------------------------------------------


def test_apply_repoints_alias_and_reminds_to_republish() -> None:
    client = _client()
    code, lines = _run(client, to_version=1, apply=True)
    assert code == 0
    assert client.set_alias_calls == [(FULL, "champion", 1)]
    text = "\n".join(lines)
    assert "APPLIED" in text
    assert "publish_lineage" in text  # reminder to re-publish (no auto-publish)


# ---------------------------------------------------------------------------
# Fail-closed
# ---------------------------------------------------------------------------


def test_refuses_unknown_version_even_with_yes() -> None:
    client = _client()
    code, lines = _run(client, to_version=99, apply=True)
    assert code == EXIT_REFUSED
    assert client.set_alias_calls == []  # never points champion at a missing version
    assert "REFUSED" in "\n".join(lines)


def test_no_op_when_already_champion() -> None:
    client = _client()
    code, lines = _run(client, to_version=2, apply=True)  # 2 is already champion
    assert code == 0
    assert client.set_alias_calls == []
    assert "No change" in "\n".join(lines)


def test_audit_handles_unset_current_champion() -> None:
    client = FakeClient(versions=[FakeVersion(1, "prompts:/p/1")], alias_to_version={})
    code, lines = _run(client, to_version=1, apply=False)
    assert code == 0
    assert "champion WAS : (no champion alias set)" in "\n".join(lines)


# ---------------------------------------------------------------------------
# main wiring
# ---------------------------------------------------------------------------


def test_main_refuses_unknown_agent_without_building_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _no_client(_profile: str | None = None) -> object:
        raise AssertionError("must not build a live client for an unknown agent")

    monkeypatch.setattr(rc, "new_lineage_client", _no_client)
    code = main(["does_not_exist", "--to-version", "1", "--registry", "/nonexistent.yaml"])
    assert code == EXIT_REFUSED


def test_main_wires_args_into_revert(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    sentinel = object()

    monkeypatch.setattr(rc, "new_lineage_client", lambda _profile=None: sentinel)

    def fake_revert(**kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(rc, "revert_champion", fake_revert)

    code = main(
        [
            "claude_code",
            "--to-version",
            "3",
            "--yes",
            "--registry",
            "/nonexistent.yaml",  # -> falls back to the in-code seed (has claude_code)
            "--catalog",
            CATALOG,
            "--schema",
            SCHEMA,
        ]
    )
    assert code == 0
    assert captured["agent_name"] == "claude_code"
    assert captured["to_version"] == 3
    assert captured["client"] is sentinel
    assert captured["apply"] is True
