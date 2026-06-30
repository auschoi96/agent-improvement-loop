"""Unit tests for the prompt-lineage publish (Tier A, Phase C).

These exercise the pure provenance-tags -> rows mapping (incl. ``is_champion`` and
``is_forced_non_improving``) and the agent-scoped ``REPLACE WHERE`` write path. The
tag dicts are produced by the **real** ``PromptProvenance.as_tags()`` the promote
step writes, so the read parser is tested against the write schema (drift fails
loud). No network/warehouse access — the registry read seam and the warehouse
client are faked, exactly like ``test_publish_versions.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from databricks.sdk.service.sql import StatementState

import ail.publish_lineage as pl
from ail.optimize.prompt_registry import PromptProvenance, PromptSource
from ail.publish_lineage import (
    LINEAGE_COLUMNS,
    PromptLineageRow,
    _lineage_row,
    build_lineage_rows,
    champion_versions,
    publish_agent_lineage,
    publish_lineage,
)
from ail.registry import Agent, AgentRegistry

CATALOG = "cat"
SCHEMA = "sch"
PROMPT = f"{CATALOG}.{SCHEMA}.token_efficient_execution"
AGENT = "claude_code"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class FakeVersion:
    version: int
    tags: dict[str, str]
    uri: str
    creation_timestamp: int | None = 1_700_000_000_000  # ms epoch


@dataclass
class FakeRegistryClient:
    """In-memory :class:`LineageRegistryClient` (records alias writes)."""

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


# -- fake warehouse client (records statements) ----------------------------


class _FakeStatus:
    def __init__(self, state: StatementState) -> None:
        self.state = state
        self.error = None


class _FakeResp:
    def __init__(self, state: StatementState) -> None:
        self.statement_id = "stmt-1"
        self.status = _FakeStatus(state)


class _FakeStatementExecution:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def execute_statement(self, *, warehouse_id, statement, wait_timeout=None):  # type: ignore[no-untyped-def]
        self.statements.append(statement)
        return _FakeResp(StatementState.SUCCEEDED)

    def get_statement(self, statement_id):  # type: ignore[no-untyped-def]
        return _FakeResp(StatementState.SUCCEEDED)


class _FakeClient:
    def __init__(self) -> None:
        self.statement_execution = _FakeStatementExecution()


# ---------------------------------------------------------------------------
# Version builders (tags produced by the REAL provenance writer)
# ---------------------------------------------------------------------------


def _seed_version(version: int = 1) -> FakeVersion:
    tags = PromptProvenance(
        source=PromptSource.SEED, changed=False, suite_version="phase2-mini"
    ).as_tags()
    return FakeVersion(version=version, tags=tags, uri=f"prompts:/{PROMPT}/{version}")


def _improving_version(version: int = 2) -> FakeVersion:
    tags = PromptProvenance(
        source=PromptSource.GEPA_EVOLVED,
        changed=True,
        gepa_best_val_score=0.82,
        gepa_num_candidates=4,
        holdout_evolved_savings_pct=45.0,
        holdout_seed_savings_pct=30.0,
        holdout_savings_delta_pct=15.0,
        candidate_artifact="artifacts/gepa_candidate.json",
        suite_version="phase2-mini",
        improving=True,
        registration_reason="held-out savings delta +15.0 pct-pts beats seed",
    ).as_tags()
    return FakeVersion(version=version, tags=tags, uri=f"prompts:/{PROMPT}/{version}")


def _forced_version(version: int = 3) -> FakeVersion:
    tags = PromptProvenance(
        source=PromptSource.GEPA_EVOLVED,
        changed=True,
        holdout_evolved_savings_pct=20.0,
        holdout_seed_savings_pct=30.0,
        holdout_savings_delta_pct=-10.0,
        candidate_artifact="artifacts/gepa_candidate_forced.json",
        suite_version="phase2-mini",
        improving=False,
        registration_reason="held-out savings delta -10.0 pct-pts does not beat seed",
        forced=True,
    ).as_tags()
    return FakeVersion(version=version, tags=tags, uri=f"prompts:/{PROMPT}/{version}")


# ---------------------------------------------------------------------------
# build_lineage_rows — provenance -> rows
# ---------------------------------------------------------------------------


def _rows(champion: int | None = 2) -> list[PromptLineageRow]:
    versions = [_seed_version(1), _improving_version(2), _forced_version(3)]
    champs: set[int] = set() if champion is None else {champion}
    return build_lineage_rows(AGENT, PROMPT, versions, champion_versions=champs, generated_at="t0")


def test_rows_are_newest_version_first() -> None:
    assert [r.version for r in _rows()] == [3, 2, 1]


def test_seed_version_maps_provenance() -> None:
    seed = next(r for r in _rows() if r.version == 1)
    assert seed.source == "seed"
    assert seed.changed is False
    assert seed.suite_version == "phase2-mini"
    # a seed is never a forced non-improvement (it has no `forced` tag).
    assert seed.is_forced_non_improving is False
    assert seed.gepa_best_val_score is None
    assert seed.registered_at == "2023-11-14T22:13:20+00:00"  # 1_700_000_000_000 ms


def test_improving_version_maps_full_gepa_provenance() -> None:
    v = next(r for r in _rows() if r.version == 2)
    assert v.source == "gepa-evolved"
    assert v.changed is True
    assert v.gepa_best_val_score == 0.82
    assert v.gepa_num_candidates == 4
    assert v.holdout_evolved_savings_pct == 45.0
    assert v.holdout_seed_savings_pct == 30.0
    assert v.holdout_savings_delta_pct == 15.0
    assert v.candidate_artifact == "artifacts/gepa_candidate.json"
    assert v.is_forced_non_improving is False


def test_forced_non_improving_is_flagged_with_reason() -> None:
    v = next(r for r in _rows() if r.version == 3)
    assert v.is_forced_non_improving is True
    assert v.holdout_savings_delta_pct == -10.0  # a negative (worse) delta
    assert "does not beat seed" in (v.registration_reason or "")


def test_is_champion_only_for_aliased_version() -> None:
    rows = _rows(champion=2)
    champs = [r.version for r in rows if r.is_champion]
    assert champs == [2]
    # and no champion when the alias is unset
    assert all(r.is_champion is False for r in _rows(champion=None))


def test_row_builder_matches_column_order() -> None:
    assert len(_lineage_row(_rows()[0])) == len(LINEAGE_COLUMNS)


def test_missing_creation_timestamp_is_honest_none() -> None:
    v = FakeVersion(version=5, tags={"ail.prompt.source": "seed"}, uri="u", creation_timestamp=None)
    (row,) = build_lineage_rows(AGENT, PROMPT, [v], champion_versions=set(), generated_at="t")
    assert row.registered_at is None


# ---------------------------------------------------------------------------
# champion_versions — authoritative alias resolution
# ---------------------------------------------------------------------------


def test_champion_versions_resolves_aliases_and_skips_unset() -> None:
    client = FakeRegistryClient(
        versions=[_seed_version(1), _improving_version(2)],
        alias_to_version={"champion": 2},  # 'production' unset -> skipped
    )
    assert champion_versions(client, PROMPT) == {2}

    none = FakeRegistryClient(versions=[_seed_version(1)], alias_to_version={})
    assert champion_versions(none, PROMPT) == set()


# ---------------------------------------------------------------------------
# Write path: agent-scoped REPLACE WHERE
# ---------------------------------------------------------------------------


def _agent(name: str) -> Agent:
    return Agent(agent_name=name, experiment_id="123")


def test_publish_agent_lineage_uses_agent_scoped_replace() -> None:
    registry_client = FakeRegistryClient(
        versions=[_seed_version(1), _improving_version(2), _forced_version(3)],
        alias_to_version={"champion": 2},
    )
    wh = _FakeClient()
    rows = publish_agent_lineage(
        _agent(AGENT),
        prompt_name="token_efficient_execution",
        registry_client=registry_client,
        warehouse_client=wh,
        warehouse_id="wh",
        catalog=CATALOG,
        schema=SCHEMA,
    )
    assert [r.version for r in rows] == [3, 2, 1]

    swaps = [s for s in wh.statement_execution.statements if "REPLACE WHERE" in s]
    assert len(swaps) == 1
    assert "agent_name = 'claude_code'" in swaps[0]
    # agent-scoped, not version-scoped: a re-publish drops versions that vanished.
    assert "version =" not in swaps[0]


def test_publish_lineage_scopes_each_agent_independently() -> None:
    registry = AgentRegistry(agents=[_agent("claude_code"), _agent("codex_cli")])
    registry_client = FakeRegistryClient(
        versions=[_seed_version(1), _improving_version(2)],
        alias_to_version={"champion": 2},
    )
    wh = _FakeClient()
    published = publish_lineage(
        registry,
        registry_client=registry_client,
        warehouse_client=wh,
        warehouse_id="wh",
        catalog=CATALOG,
        schema=SCHEMA,
    )
    assert set(published) == {"claude_code", "codex_cli"}

    swaps = [s for s in wh.statement_execution.statements if "REPLACE WHERE" in s]
    assert len(swaps) == 2
    assert any("agent_name = 'claude_code'" in s for s in swaps)
    assert any("agent_name = 'codex_cli'" in s for s in swaps)
    # no swap touches both agents — each replaces exactly its own slice.
    assert all(not ("claude_code" in s and "codex_cli" in s) for s in swaps)


def test_publish_with_no_versions_still_clears_the_slice() -> None:
    registry_client = FakeRegistryClient(versions=[], alias_to_version={})
    wh = _FakeClient()
    rows = publish_agent_lineage(
        _agent(AGENT),
        prompt_name="token_efficient_execution",
        registry_client=registry_client,
        warehouse_client=wh,
        warehouse_id="wh",
        catalog=CATALOG,
        schema=SCHEMA,
    )
    assert rows == []
    # The REPLACE WHERE still runs (clears any prior rows for the agent) — honest
    # empty state, not a no-op that leaves stale rows behind.
    swaps = [s for s in wh.statement_execution.statements if "REPLACE WHERE" in s]
    assert len(swaps) == 1
    assert "agent_name = 'claude_code'" in swaps[0]


def test_ddl_creates_the_lineage_table() -> None:
    ddl = "\n".join(pl._ddl("cat", "sch"))
    assert ".agent_prompt_lineage (" in ddl


# ---------------------------------------------------------------------------
# Live client wiring (offline: fake underlying MlflowClient, no network)
# ---------------------------------------------------------------------------


def test_mlflow_lineage_client_alias_failsoft() -> None:
    class _Raising:
        def get_prompt_version_by_alias(self, name: str, alias: str) -> Any:
            raise RuntimeError("alias not found")

    client = pl._MlflowLineageClient(_Raising())
    # An unset alias on the UC store raises; "no champion yet" is legitimate, so the
    # client must fail soft to None rather than crash the publish.
    assert client.get_prompt_version_by_alias(PROMPT, "champion") is None
