"""Offline coverage for catalog/schema resolution on live write seams."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from ail.l3.contract import RankedAsset
from ail.loop import apply_service
from ail.optimize.assets.metric_view import generate_metric_view
from ail.optimize.prompt_registry import PromptProvenance, PromptSource, register_prompt_body
from ail.workspace_config import (
    ALLOW_REFERENCE_ENV,
    CATALOG_ENV,
    SCHEMA_ENV,
    resolve_catalog_schema,
)
from ail.workspace_guards import REFERENCE_WORKSPACE_DEFAULTS


@dataclass
class _Version:
    version: int = 1
    uri: str = "prompts:/unused/1"


@dataclass
class _Registry:
    register_calls: list[str] = field(default_factory=list)

    def register_prompt(
        self,
        name: str,
        template: str,
        commit_message: str | None,
        tags: dict[str, str] | None,
    ) -> _Version:
        self.register_calls.append(name)
        return _Version(uri=f"prompts:/{name}/1")

    def set_prompt_alias(self, name: str, alias: str, version: int) -> None:
        raise AssertionError("alias should not be set in this test")

    def search_prompts(self, filter_string: str | None) -> list[Any]:
        raise AssertionError("search should not be called in this test")

    def load_prompt(self, name_or_uri: str) -> Any:
        raise AssertionError("load should not be called in this test")


def _metric_view_asset() -> RankedAsset:
    return RankedAsset(
        asset_type="metric_view",
        title="Token waste by tool",
        rank=1,
        n_traces=3,
        occurrences=3,
        rationales=["track redundant calls and token waste"],
    )


def test_explicit_prompt_registration_uses_explicit_catalog_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(CATALOG_ENV, raising=False)
    monkeypatch.delenv(SCHEMA_ENV, raising=False)
    registry = _Registry()

    register_prompt_body(
        body="body",
        provenance=PromptProvenance(source=PromptSource.SEED),
        catalog="other_cat",
        schema="other",
        client=registry,
    )

    assert registry.register_calls == ["other_cat.other.token_efficient_execution"]


def test_env_catalog_schema_reaches_write_seams(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CATALOG_ENV, "other_cat")
    monkeypatch.setenv(SCHEMA_ENV, "other")

    registry = _Registry()
    register_prompt_body(
        body="body",
        provenance=PromptProvenance(source=PromptSource.SEED),
        client=registry,
    )
    assert registry.register_calls == ["other_cat.other.token_efficient_execution"]

    lineage_calls: list[dict[str, Any]] = []

    def spy_publish_agent_lineage(*args: Any, **kwargs: Any) -> None:
        lineage_calls.append(kwargs)

    monkeypatch.setattr(apply_service, "publish_agent_lineage", spy_publish_agent_lineage)
    recorder = apply_service.build_lineage_recorder(
        agent=object(),
        prompt_name="prompt",
        registry_client=object(),
        warehouse_client=object(),
        warehouse_id="wh",
    )
    recorder(object())
    assert lineage_calls == [
        {
            "prompt_name": "prompt",
            "registry_client": lineage_calls[0]["registry_client"],
            "warehouse_client": lineage_calls[0]["warehouse_client"],
            "warehouse_id": "wh",
            "catalog": "other_cat",
            "schema": "other",
        }
    ]

    generated = generate_metric_view(_metric_view_asset())
    assert generated.spec.full_name.startswith("other_cat.other.")
    assert generated.spec.source.startswith("other_cat.other.")


def test_missing_catalog_schema_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CATALOG_ENV, raising=False)
    monkeypatch.delenv(SCHEMA_ENV, raising=False)
    monkeypatch.delenv(ALLOW_REFERENCE_ENV, raising=False)

    with pytest.raises(RuntimeError, match="catalog/schema"):
        resolve_catalog_schema()


def test_reference_catalog_schema_requires_explicit_allow(monkeypatch: pytest.MonkeyPatch) -> None:
    reference_catalog = next(iter(REFERENCE_WORKSPACE_DEFAULTS["catalog"]))
    reference_schema = next(iter(REFERENCE_WORKSPACE_DEFAULTS["schema"]))
    monkeypatch.setenv(CATALOG_ENV, reference_catalog)
    monkeypatch.setenv(SCHEMA_ENV, reference_schema)
    monkeypatch.delenv(ALLOW_REFERENCE_ENV, raising=False)

    with pytest.raises(RuntimeError, match="reference workspace"):
        resolve_catalog_schema()

    monkeypatch.setenv(ALLOW_REFERENCE_ENV, "1")
    assert resolve_catalog_schema() == (reference_catalog, reference_schema)


def test_l0_contract_publish_invariant_still_holds() -> None:
    from ail.optimize.assets.l0_contract import verify_against_publish

    verify_against_publish()
