"""Tests for the UC prompt-registry promote step (``ail.optimize.prompt_registry``).

Every test here is **offline**: the MLflow prompt-registry client is replaced by an
injected :class:`FakeRegistryClient` that records calls and returns canned version
objects, so no live Databricks/MLflow call is ever made (no ``live`` marker). The
one test that exercises the default ``mlflow.genai``-backed client monkeypatches the
``mlflow.genai`` functions, so it proves the seam wiring without a network call.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from ail.optimize.gepa_runner import GepaOptimizationResult
from ail.optimize.lever import token_efficiency_skill
from ail.optimize.phase2 import Phase2Artifact
from ail.optimize.prompt_registry import (
    DEFAULT_CATALOG,
    DEFAULT_PROMPT_NAME,
    DEFAULT_SCHEMA,
    NonImprovingCandidateError,
    PromptProvenance,
    PromptSource,
    candidate_improvement,
    register_gepa_candidate,
    register_prompt_body,
    register_seed_prompt,
    resolve_prompt_name,
    search_registered_prompts,
)
from ail.workspace_config import CATALOG_ENV, SCHEMA_ENV

TEST_CATALOG = "test_catalog"
TEST_SCHEMA = "test_schema"
EXPECTED_FULL_NAME = f"{TEST_CATALOG}.{TEST_SCHEMA}.{DEFAULT_PROMPT_NAME}"


@pytest.fixture(autouse=True)
def _configured_test_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CATALOG_ENV, TEST_CATALOG)
    monkeypatch.setenv(SCHEMA_ENV, TEST_SCHEMA)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class FakeVersion:
    version: int
    uri: str


@dataclass
class FakePrompt:
    name: str


@dataclass
class FakeRegistryClient:
    """In-memory stand-in for the MLflow prompt-registry client (records calls)."""

    next_version: int = 1
    register_calls: list[dict[str, Any]] = field(default_factory=list)
    alias_calls: list[tuple[str, str, int]] = field(default_factory=list)
    search_calls: list[str | None] = field(default_factory=list)
    load_calls: list[str] = field(default_factory=list)
    search_result: list[FakePrompt] = field(default_factory=list)

    def register_prompt(
        self,
        name: str,
        template: str,
        commit_message: str | None,
        tags: dict[str, str] | None,
    ) -> FakeVersion:
        self.register_calls.append(
            {"name": name, "template": template, "commit_message": commit_message, "tags": tags}
        )
        version = self.next_version
        self.next_version += 1
        return FakeVersion(version=version, uri=f"prompts:/{name}/{version}")

    def set_prompt_alias(self, name: str, alias: str, version: int) -> None:
        self.alias_calls.append((name, alias, version))

    def search_prompts(self, filter_string: str | None) -> list[FakePrompt]:
        self.search_calls.append(filter_string)
        return self.search_result

    def load_prompt(self, name_or_uri: str) -> FakePrompt:
        self.load_calls.append(name_or_uri)
        return FakePrompt(name=name_or_uri)


# ---------------------------------------------------------------------------
# Result builders
# ---------------------------------------------------------------------------


def _artifact(*, n_tasks: int, n_promote: int, savings_pct: float | None) -> Phase2Artifact:
    return Phase2Artifact(
        n_tasks=n_tasks,
        n_promote=n_promote,
        realized_token_savings_pct=savings_pct,
    )


def _result(
    *,
    changed: bool = True,
    evolved_body: str = "# Evolved skill\n\nDo less re-reading.",
    evolved_savings: float | None = 45.0,
    seed_savings: float | None = 30.0,
    with_holdout: bool = True,
) -> GepaOptimizationResult:
    holdout_evolved = (
        _artifact(n_tasks=3, n_promote=2, savings_pct=evolved_savings) if with_holdout else None
    )
    holdout_seed = (
        _artifact(n_tasks=3, n_promote=1, savings_pct=seed_savings) if with_holdout else None
    )
    return GepaOptimizationResult(
        component_name="token-efficient-execution",
        seed_skill_body="# Seed skill\n\nAvoid re-reading.",
        evolved_skill_body=evolved_body,
        changed=changed,
        reflection_lm="databricks:/databricks-claude-sonnet-4-6",
        gepa_num_candidates=4,
        gepa_best_val_score=0.82,
        suite_version="phase2-mini",
        suite_content_hash="deadbeefcafe0001",
        holdout_task_ids=["ts-04", "ts-05"],
        train_task_ids=["ts-01", "ts-02", "ts-03"],
        holdout_evolved=holdout_evolved,
        holdout_seed_baseline=holdout_seed,
    )


def _write_candidate(tmp_path: Path, result: GepaOptimizationResult) -> Path:
    path = tmp_path / "gepa_candidate.json"
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# resolve_prompt_name
# ---------------------------------------------------------------------------


class TestResolvePromptName:
    def test_bare_leaf_is_qualified(self) -> None:
        assert resolve_prompt_name("foo") == f"{DEFAULT_CATALOG}.{DEFAULT_SCHEMA}.foo"

    def test_full_name_passthrough(self) -> None:
        assert resolve_prompt_name("cat.sch.leaf") == "cat.sch.leaf"

    def test_custom_catalog_schema(self) -> None:
        assert resolve_prompt_name("leaf", catalog="c", schema="s") == "c.s.leaf"

    def test_partially_qualified_rejected(self) -> None:
        with pytest.raises(ValueError, match="partially qualified"):
            resolve_prompt_name("schema.leaf")


# ---------------------------------------------------------------------------
# register_prompt_body
# ---------------------------------------------------------------------------


class TestRegisterPromptBody:
    def test_stamps_provenance_tags_and_versions(self) -> None:
        client = FakeRegistryClient(next_version=7)
        prov = PromptProvenance(source=PromptSource.SEED, changed=False, suite_version="v1")
        out = register_prompt_body(body="# body\n\ntext", provenance=prov, client=client)

        assert out.name == EXPECTED_FULL_NAME
        assert out.version == 7
        assert out.uri == f"prompts:/{EXPECTED_FULL_NAME}/7"
        assert out.source is PromptSource.SEED

        call = client.register_calls[0]
        assert call["name"] == EXPECTED_FULL_NAME
        assert call["template"] == "# body\n\ntext"  # body registered verbatim
        assert call["tags"]["ail.prompt.source"] == "seed"
        assert call["tags"]["ail.prompt.changed"] == "false"
        assert call["tags"]["ail.prompt.suite_version"] == "v1"

    def test_alias_set_only_when_requested(self) -> None:
        client = FakeRegistryClient()
        prov = PromptProvenance(source=PromptSource.SEED)
        register_prompt_body(body="x", provenance=prov, client=client)
        assert client.alias_calls == []

        client2 = FakeRegistryClient(next_version=3)
        register_prompt_body(body="x", provenance=prov, client=client2, alias="champion")
        assert client2.alias_calls == [(EXPECTED_FULL_NAME, "champion", 3)]

    def test_empty_body_refused(self) -> None:
        client = FakeRegistryClient()
        prov = PromptProvenance(source=PromptSource.SEED)
        with pytest.raises(ValueError, match="empty prompt body"):
            register_prompt_body(body="   \n ", provenance=prov, client=client)
        assert client.register_calls == []

    def test_none_provenance_fields_omitted_from_tags(self) -> None:
        client = FakeRegistryClient()
        prov = PromptProvenance(source=PromptSource.SEED)
        register_prompt_body(body="x", provenance=prov, client=client)
        tags = client.register_calls[0]["tags"]
        assert tags == {"ail.prompt.source": "seed"}


class TestRegisterSeedPrompt:
    def test_defaults_to_on_disk_seed_body(self) -> None:
        client = FakeRegistryClient()
        out = register_seed_prompt(client=client)
        assert out.source is PromptSource.SEED
        call = client.register_calls[0]
        assert call["template"] == token_efficiency_skill().body
        assert call["tags"]["ail.prompt.source"] == "seed"
        assert call["tags"]["ail.prompt.changed"] == "false"


# ---------------------------------------------------------------------------
# candidate_improvement (the fail-closed gate)
# ---------------------------------------------------------------------------


class TestCandidateImprovement:
    def test_improving_when_delta_positive(self) -> None:
        ok, reason = candidate_improvement(_result(evolved_savings=45.0, seed_savings=30.0))
        assert ok is True
        assert "beats seed" in reason

    def test_unchanged_is_not_improving(self) -> None:
        ok, reason = candidate_improvement(_result(changed=False))
        assert ok is False
        assert "changed=False" in reason

    def test_missing_holdout_is_not_improving(self) -> None:
        ok, reason = candidate_improvement(_result(with_holdout=False))
        assert ok is False
        assert "no held-out validation" in reason

    def test_delta_not_beating_seed_is_not_improving(self) -> None:
        ok, reason = candidate_improvement(_result(evolved_savings=20.0, seed_savings=30.0))
        assert ok is False
        assert "does not beat seed" in reason

    def test_equal_savings_is_not_improving(self) -> None:
        ok, _ = candidate_improvement(_result(evolved_savings=30.0, seed_savings=30.0))
        assert ok is False

    def test_evolved_savings_none_is_not_improving(self) -> None:
        ok, reason = candidate_improvement(_result(evolved_savings=None, seed_savings=30.0))
        assert ok is False
        assert "cannot prove improvement" in reason

    def test_nan_delta_is_not_improving(self) -> None:
        # nan <= 0 is False, so a NaN delta must be trapped explicitly or it slips
        # through and registers as a fake improvement (`+nan pct-pts beats seed`).
        result = _result(evolved_savings=float("nan"), seed_savings=30.0)
        assert not math.isfinite(result.holdout_savings_delta_pct)  # precondition
        ok, reason = candidate_improvement(result)
        assert ok is False
        assert "does not beat seed" in reason

    def test_positive_inf_delta_is_not_improving(self) -> None:
        # inf <= 0 is also False; +inf must not pass the guard either.
        result = _result(evolved_savings=float("inf"), seed_savings=30.0)
        ok, reason = candidate_improvement(result)
        assert ok is False
        assert "does not beat seed" in reason


# ---------------------------------------------------------------------------
# register_gepa_candidate
# ---------------------------------------------------------------------------


class TestRegisterGepaCandidate:
    def test_happy_path_registers_evolved_body_with_provenance(self, tmp_path: Path) -> None:
        result = _result(evolved_body="# Evolved\n\nReuse context.")
        path = _write_candidate(tmp_path, result)
        client = FakeRegistryClient(next_version=2)

        out = register_gepa_candidate(path, client=client)

        assert out.source is PromptSource.GEPA_EVOLVED
        assert out.version == 2
        assert out.forced is False
        call = client.register_calls[0]
        assert call["template"] == "# Evolved\n\nReuse context."
        tags = call["tags"]
        assert tags["ail.prompt.source"] == "gepa-evolved"
        assert tags["ail.prompt.changed"] == "true"
        assert tags["ail.prompt.suite_content_hash"] == "deadbeefcafe0001"
        assert tags["ail.prompt.gepa_best_val_score"] == "0.82"
        assert tags["ail.prompt.gepa_num_candidates"] == "4"
        assert tags["ail.prompt.holdout_evolved_promote"] == "2/3"
        assert tags["ail.prompt.holdout_seed_promote"] == "1/3"
        assert tags["ail.prompt.holdout_savings_delta_pct"] == "15.0"
        assert tags["ail.prompt.candidate_artifact"] == str(path)
        assert tags["ail.prompt.improving"] == "true"
        assert "ail.prompt.forced" not in tags
        # provenance summary commit message, no human-authored override
        assert "Promote GEPA candidate" in call["commit_message"]

    def test_alias_passthrough(self, tmp_path: Path) -> None:
        path = _write_candidate(tmp_path, _result())
        client = FakeRegistryClient(next_version=5)
        register_gepa_candidate(path, client=client, alias="production")
        assert client.alias_calls == [(EXPECTED_FULL_NAME, "production", 5)]

    def test_refuses_unchanged_candidate(self, tmp_path: Path) -> None:
        path = _write_candidate(tmp_path, _result(changed=False))
        client = FakeRegistryClient()
        with pytest.raises(NonImprovingCandidateError, match="changed=False"):
            register_gepa_candidate(path, client=client)
        assert client.register_calls == []  # nothing registered

    def test_refuses_without_holdout(self, tmp_path: Path) -> None:
        path = _write_candidate(tmp_path, _result(with_holdout=False))
        client = FakeRegistryClient()
        with pytest.raises(NonImprovingCandidateError, match="no held-out validation"):
            register_gepa_candidate(path, client=client)
        assert client.register_calls == []

    def test_refuses_non_beating_candidate(self, tmp_path: Path) -> None:
        path = _write_candidate(tmp_path, _result(evolved_savings=20.0, seed_savings=30.0))
        client = FakeRegistryClient()
        with pytest.raises(NonImprovingCandidateError, match="does not beat seed"):
            register_gepa_candidate(path, client=client)
        assert client.register_calls == []

    def test_force_registers_and_records_reason(self, tmp_path: Path) -> None:
        path = _write_candidate(tmp_path, _result(evolved_savings=20.0, seed_savings=30.0))
        client = FakeRegistryClient(next_version=9)

        out = register_gepa_candidate(path, client=client, force=True)

        assert out.forced is True
        assert out.version == 9
        tags = client.register_calls[0]["tags"]
        assert tags["ail.prompt.forced"] == "true"
        assert tags["ail.prompt.improving"] == "false"
        assert "does not beat seed" in tags["ail.prompt.registration_reason"]
        assert "FORCE-registered non-improving" in client.register_calls[0]["commit_message"]

    def test_force_on_improving_candidate_is_not_marked_forced(self, tmp_path: Path) -> None:
        path = _write_candidate(tmp_path, _result(evolved_savings=45.0, seed_savings=30.0))
        client = FakeRegistryClient()
        out = register_gepa_candidate(path, client=client, force=True)
        assert out.forced is False
        assert "ail.prompt.forced" not in client.register_calls[0]["tags"]

    @pytest.mark.parametrize("delta_value", [float("nan"), float("inf")])
    def test_refuses_nonfinite_delta_and_never_registers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, delta_value: float
    ) -> None:
        # pydantic null-ifies NaN/inf through JSON, so to exercise the in-memory
        # non-finite trap we parse to a result whose computed delta is non-finite.
        result = _result(evolved_savings=delta_value, seed_savings=30.0)
        assert not math.isfinite(result.holdout_savings_delta_pct)  # precondition
        monkeypatch.setattr(
            GepaOptimizationResult,
            "model_validate_json",
            classmethod(lambda cls, data: result),
        )
        path = tmp_path / "candidate.json"
        path.write_text("{}", encoding="utf-8")
        client = FakeRegistryClient()
        with pytest.raises(NonImprovingCandidateError, match="does not beat seed"):
            register_gepa_candidate(path, client=client)
        assert client.register_calls == []  # register_prompt NEVER called

    def test_force_with_custom_commit_message_keeps_force_prefix(self, tmp_path: Path) -> None:
        path = _write_candidate(tmp_path, _result(evolved_savings=20.0, seed_savings=30.0))
        client = FakeRegistryClient()
        register_gepa_candidate(
            path, client=client, force=True, commit_message="Ship for the demo per Austin"
        )
        msg = client.register_calls[0]["commit_message"]
        assert msg.startswith("FORCE-registered non-improving GEPA candidate:")
        assert "Ship for the demo per Austin" in msg  # caller text preserved, not dropped
        assert client.register_calls[0]["tags"]["ail.prompt.forced"] == "true"

    def test_custom_commit_message_passthrough_when_not_forced(self, tmp_path: Path) -> None:
        path = _write_candidate(tmp_path, _result(evolved_savings=45.0, seed_savings=30.0))
        client = FakeRegistryClient()
        register_gepa_candidate(path, client=client, commit_message="Promote per review")
        # a genuine (non-forced) promotion keeps the caller's message verbatim
        assert client.register_calls[0]["commit_message"] == "Promote per review"


# ---------------------------------------------------------------------------
# search_registered_prompts
# ---------------------------------------------------------------------------


class TestSearchRegisteredPrompts:
    def test_builds_catalog_schema_filter(self) -> None:
        client = FakeRegistryClient(search_result=[FakePrompt(name="p1")])
        out = search_registered_prompts(client=client, catalog="c", schema="s")
        assert client.search_calls == ["catalog = 'c' AND schema = 's'"]
        assert [p.name for p in out] == ["p1"]


# ---------------------------------------------------------------------------
# Default client wiring (offline: monkeypatch mlflow.genai, no network)
# ---------------------------------------------------------------------------


class TestDefaultClientWiring:
    def test_delegates_to_public_mlflow_genai_api(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import mlflow.genai

        from ail.optimize.prompt_registry import _GenAIPromptRegistryClient

        calls: dict[str, Any] = {}

        def fake_register(**kw: Any) -> FakeVersion:
            calls["register"] = kw
            return FakeVersion(1, "prompts:/x/1")

        def fake_alias(*a: Any) -> None:
            calls["alias"] = a

        def fake_search(**kw: Any) -> list[FakePrompt]:
            calls["search"] = kw
            return [FakePrompt("p")]

        monkeypatch.setattr(mlflow.genai, "register_prompt", fake_register)
        monkeypatch.setattr(mlflow.genai, "set_prompt_alias", fake_alias)
        monkeypatch.setattr(mlflow.genai, "search_prompts", fake_search)

        c = _GenAIPromptRegistryClient()
        v = c.register_prompt("cat.sch.leaf", "body", "msg", {"k": "v"})
        c.set_prompt_alias("cat.sch.leaf", "champion", 1)
        found = c.search_prompts("name = 'x'")

        assert v.version == 1
        assert calls["register"] == {
            "name": "cat.sch.leaf",
            "template": "body",
            "commit_message": "msg",
            "tags": {"k": "v"},
        }
        assert calls["alias"] == ("cat.sch.leaf", "champion", 1)
        assert calls["search"] == {"filter_string": "name = 'x'"}
        assert [p.name for p in found] == ["p"]


def test_candidate_json_is_parsed_via_real_schema(tmp_path: Path) -> None:
    """The candidate file is read through the real GepaOptimizationResult schema."""
    path = _write_candidate(tmp_path, _result())
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["human_gate_required"] is True  # the artifact is a human-gated candidate
    # round-trips cleanly through the schema the runner emits
    GepaOptimizationResult.model_validate_json(path.read_text(encoding="utf-8"))
