"""Tests for the Stage 6 helper-asset generator (:mod:`ail.optimize.assets`).

All offline / non-live (no ``live`` marker, no Databricks call): the generator
builds and validates specs statically against the L0 column contract. Coverage:

* the L0 column registry matches :mod:`ail.publish` exactly (drift guard);
* the registry/interface dispatches by ``asset_type`` and unimplemented types
  raise a clear ``next`` signal;
* generation from a sample ``ranked_assets`` list (the fixture mirrors
  ``artifacts/rlm_batch_report.json``) yields a valid ``metric_view`` spec;
* the fabrication guard drops a measure with no backing L0 column, with a reason;
* spec validity — well-formed YAML and only-real-column references.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from ail.l3.contract import CohortReviewReport, RankedAsset
from ail.optimize.assets import (
    AssetGenerator,
    AssetGeneratorNotImplemented,
    GeneratedMetricView,
    MetricViewDimension,
    MetricViewMeasure,
    MetricViewSpec,
    generate_asset,
    generate_metric_view,
    generate_metric_views_from_report,
    get_generator,
    registered_asset_types,
    validate_spec,
    verify_against_publish,
)
from ail.optimize.assets.l0_contract import L0_CONTRACT, SESSION_TABLE
from ail.optimize.assets.metric_view import SpecValidationError

FIXTURE = Path(__file__).parent / "fixtures" / "rlm_batch_report_sample.json"


@pytest.fixture(autouse=True)
def _allow_reference_workspace_for_default_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIL_CATALOG", "austin_choi_omni_agent_catalog")
    monkeypatch.setenv("AIL_SCHEMA", "agent_improvement_loop")
    monkeypatch.setenv("AIL_ALLOW_REFERENCE_WORKSPACE", "1")


def _report() -> CohortReviewReport:
    return CohortReviewReport.model_validate(json.loads(FIXTURE.read_text()))


def _metric_view_rec() -> RankedAsset:
    return next(a for a in _report().ranked_assets if a.asset_type == "metric_view")


def _rec(asset_type: str, title: str = "x", **kw: object) -> RankedAsset:
    return RankedAsset(
        asset_type=asset_type,  # type: ignore[arg-type]
        title=title,
        rank=kw.pop("rank", 1),  # type: ignore[arg-type]
        n_traces=kw.pop("n_traces", 1),  # type: ignore[arg-type]
        occurrences=kw.pop("occurrences", 1),  # type: ignore[arg-type]
        **kw,  # type: ignore[arg-type]
    )


# --- L0 column contract is the real one (drift guard) ----------------------


class TestL0Contract:
    def test_registry_matches_publish(self) -> None:
        # The source of truth for column names is ail.publish; this raises if the
        # generator's registry has drifted from it (fabrication / staleness guard).
        verify_against_publish()

    def test_session_table_resolves_by_fqn_and_short_name(self) -> None:
        table = L0_CONTRACT.get_table(SESSION_TABLE)
        assert table is not None
        assert L0_CONTRACT.table_for(table.fqn) is table
        assert table.has("redundancy_rate")
        assert not table.has("not_a_real_column")


# --- registry / dispatch ----------------------------------------------------


class TestRegistryDispatch:
    def test_metric_view_is_registered_and_dispatches(self) -> None:
        gen = get_generator("metric_view")
        assert isinstance(gen, AssetGenerator)
        assert gen.asset_type == "metric_view"
        result = generate_asset(_metric_view_rec())
        assert isinstance(result, GeneratedMetricView)

    @pytest.mark.parametrize("asset_type", ["skill", "tool", "prompt_change"])
    def test_unimplemented_types_raise_next_signal(self, asset_type: str) -> None:
        # Registered as explicit placeholders, so dispatch names the type.
        assert asset_type in registered_asset_types()
        with pytest.raises(AssetGeneratorNotImplemented) as exc:
            generate_asset(_rec(asset_type))
        assert exc.value.asset_type == asset_type
        assert exc.value.status == "next"

    def test_unknown_type_also_raises_next_signal(self) -> None:
        # Even a type with no placeholder fails with the same explicit signal,
        # never a bare KeyError.
        with pytest.raises(AssetGeneratorNotImplemented) as exc:
            get_generator("semantic_layer")
        assert exc.value.asset_type == "semantic_layer"


# --- metric_view generation from the sample ranked_assets list -------------


class TestMetricViewGeneration:
    def test_generates_valid_spec_from_recommendation(self) -> None:
        rec = _metric_view_rec()
        gen = generate_metric_view(rec)

        # Provenance back to the recommendation.
        assert gen.asset_type == "metric_view"
        assert gen.source_rank == rec.rank
        assert gen.n_source_traces == rec.n_traces
        assert gen.source_trace_ids == rec.trace_ids

        spec = gen.spec
        # Source is the real L0 session table; name lives in its schema.
        table = L0_CONTRACT.get_table(SESSION_TABLE)
        assert table is not None
        assert spec.source == table.fqn
        assert spec.full_name.startswith(f"{table.catalog}.{table.schema}.mv_")

        # The waste the RLM flagged is reflected as measures.
        measure_names = {m.name for m in spec.measures}
        assert "Redundant Tool Call Rate" in measure_names
        assert "Tokens per Trace" in measure_names
        assert "Total Tool Calls" in measure_names
        # Baseline COUNT(1) is always present so the view is queryable.
        assert any(m.expr == "COUNT(1)" for m in spec.measures)
        # No measures were dropped (all backed by real columns).
        assert gen.dropped_measures == []

        # The spec passes full offline validation.
        validate_spec(spec)

    def test_every_referenced_column_is_real(self) -> None:
        spec = generate_metric_view(_metric_view_rec()).spec
        table = L0_CONTRACT.get_table(SESSION_TABLE)
        assert table is not None
        real = set(table.column_names)
        # Re-derive referenced columns from each rendered expression independently.
        from ail.optimize.assets.metric_view import _referenced_columns

        for entry in [*spec.measures, *spec.dimensions]:
            assert _referenced_columns(entry.expr) <= real, entry.expr

    def test_recommendation_with_no_concept_uses_default_set(self) -> None:
        rec = _rec("metric_view", title="Build a governed metric view please")
        gen = generate_metric_view(rec)
        assert any("default token-efficiency" in n for n in gen.notes)
        # Default set still yields a usable, valid spec.
        validate_spec(gen.spec)
        assert len(gen.spec.measures) >= 2

    def test_from_report_generates_one_view_per_metric_view_rec(self) -> None:
        report = _report()
        views = generate_metric_views_from_report(report)
        n_mv = sum(1 for a in report.ranked_assets if a.asset_type == "metric_view")
        assert len(views) == n_mv
        assert all(isinstance(v, GeneratedMetricView) for v in views)
        for v in views:
            validate_spec(v.spec)

    def test_write_emits_deployable_sql_and_json(self, tmp_path: Path) -> None:
        gen = generate_metric_view(_metric_view_rec())
        paths = gen.write(tmp_path)
        sql = Path(paths["sql"]).read_text()
        assert sql.startswith("CREATE OR REPLACE VIEW ")
        assert "WITH METRICS" in sql and "LANGUAGE YAML" in sql
        # The JSON round-trips back into the typed object.
        reloaded = GeneratedMetricView.model_validate_json(Path(paths["json"]).read_text())
        assert reloaded.spec.full_name == gen.spec.full_name


# --- fabrication guard ------------------------------------------------------


class TestFabricationGuard:
    def test_measure_without_backing_column_is_dropped_with_reason(self) -> None:
        # An L0 contract that lacks the redundancy columns: a recommendation asking
        # for the redundancy measures must DROP them with a reason, never fabricate.
        restricted = L0_CONTRACT.restricted(
            SESSION_TABLE, drop={"redundant_tool_calls", "redundancy_rate"}
        )
        rec = _rec(
            "metric_view",
            title="Redundant tool-call and token-waste view",
            rationales=["track redundancy and tokens per task"],
            expected_benefits=["cut redundant repeated calls"],
        )
        gen = generate_metric_view(rec, contract=restricted)

        dropped_names = {d.name for d in gen.dropped_measures}
        assert {"Redundant Tool Calls", "Redundant Tool Call Rate"} <= dropped_names
        for d in gen.dropped_measures:
            assert d.missing_columns  # the specific real column(s) that were absent
            assert "no backing" in d.reason.lower() or "no fabrication" in d.reason.lower()

        # Token measures, backed by real columns, still come through.
        kept = {m.name for m in gen.spec.measures}
        assert "Tokens per Trace" in kept
        # The dropped measures are NOT emitted...
        assert "Redundant Tool Call Rate" not in kept
        # ...and the surviving spec is still valid against the restricted contract.
        validate_spec(gen.spec, restricted)
        assert any("fabrication guard" in n for n in gen.notes)


# --- spec validity ----------------------------------------------------------


class TestSpecValidation:
    def test_well_formed_yaml_round_trips(self) -> None:
        spec = generate_metric_view(_metric_view_rec()).spec
        body = spec.to_yaml()
        doc = yaml.safe_load(body)
        assert doc["source"] == spec.source
        assert isinstance(doc["measures"], list) and doc["measures"]
        assert isinstance(doc["dimensions"], list) and doc["dimensions"]
        # The CREATE statement embeds exactly this YAML body.
        assert body.rstrip() in spec.to_create_sql()

    def test_fabricated_column_is_rejected(self) -> None:
        # A hand-built spec referencing a non-existent column must fail validation
        # (the only-real-columns gate), not be silently accepted.
        spec = MetricViewSpec(
            full_name="austin_choi_omni_agent_catalog.agent_improvement_loop.mv_bad",
            source=L0_CONTRACT.get_table(SESSION_TABLE).fqn,  # type: ignore[union-attr]
            dimensions=[MetricViewDimension(name="Model", expr="model")],
            measures=[MetricViewMeasure(name="Bogus", expr="SUM(made_up_column)")],
        )
        with pytest.raises(SpecValidationError) as exc:
            validate_spec(spec)
        assert any("made_up_column" in p for p in exc.value.problems)

    def test_measure_without_aggregate_is_rejected(self) -> None:
        spec = MetricViewSpec(
            full_name="austin_choi_omni_agent_catalog.agent_improvement_loop.mv_bad2",
            source=L0_CONTRACT.get_table(SESSION_TABLE).fqn,  # type: ignore[union-attr]
            dimensions=[MetricViewDimension(name="Model", expr="model")],
            measures=[MetricViewMeasure(name="NotAgg", expr="total_tokens")],
        )
        with pytest.raises(SpecValidationError) as exc:
            validate_spec(spec)
        assert any("aggregate" in p for p in exc.value.problems)

    def test_unknown_source_table_is_rejected(self) -> None:
        spec = MetricViewSpec(
            full_name="c.s.mv_bad3",
            source="c.s.not_an_l0_table",
            dimensions=[MetricViewDimension(name="Model", expr="model")],
            measures=[MetricViewMeasure(name="Cnt", expr="COUNT(1)")],
        )
        with pytest.raises(SpecValidationError) as exc:
            validate_spec(spec)
        assert any("not a known L0 table" in p for p in exc.value.problems)

    def test_filter_referencing_unknown_column_is_rejected(self) -> None:
        # A global filter is also column-checked against the L0 allow-list, so a
        # filter cannot smuggle a fabricated column past the guard.
        spec = MetricViewSpec(
            full_name="austin_choi_omni_agent_catalog.agent_improvement_loop.mv_filt",
            source=L0_CONTRACT.get_table(SESSION_TABLE).fqn,  # type: ignore[union-attr]
            filter="ghost_column > 0 AND status = 'OK'",
            dimensions=[MetricViewDimension(name="Model", expr="model")],
            measures=[MetricViewMeasure(name="Cnt", expr="COUNT(1)")],
        )
        with pytest.raises(SpecValidationError) as exc:
            validate_spec(spec)
        assert any("filter" in p and "ghost_column" in p for p in exc.value.problems)

    def test_filter_referencing_real_columns_is_accepted(self) -> None:
        # A filter over real columns (with literals/operators) passes — the check
        # is column-membership, not a blanket ban on filters.
        spec = MetricViewSpec(
            full_name="austin_choi_omni_agent_catalog.agent_improvement_loop.mv_filt_ok",
            source=L0_CONTRACT.get_table(SESSION_TABLE).fqn,  # type: ignore[union-attr]
            filter="status = 'OK' AND total_tokens > 1000",
            dimensions=[MetricViewDimension(name="Model", expr="model")],
            measures=[MetricViewMeasure(name="Cnt", expr="COUNT(1)")],
        )
        validate_spec(spec)  # does not raise


# --- SQL-injection / dollar-quote safety (untrusted LLM title) -------------


class TestDollarQuoteSafety:
    def test_title_with_dollar_dollar_produces_valid_sql(self) -> None:
        # asset.title is free-text LLM output; a literal "$$" must not break out of
        # the CREATE ... AS $$ ... $$ dollar-quoted block.
        rec = _rec(
            "metric_view",
            title="Cut $$ waste tracking view",
            rationales=["track tokens and redundancy"],
        )
        gen = generate_metric_view(rec)
        sql = gen.spec.to_create_sql()

        # Exactly two "$$" sequences remain: the opening and closing delimiters,
        # i.e. none survive inside the body.
        assert sql.count("$$") == 2
        assert sql.startswith("CREATE OR REPLACE VIEW ")
        assert "\nAS $$\n" in sql and sql.endswith("\n$$")

        # The body between the delimiters is free of any "$$".
        body = sql.split("\nAS $$\n", 1)[1].rsplit("\n$$", 1)[0]
        assert "$$" not in body
        # The comment kept the recommendation, just with the "$$" run collapsed.
        assert "Cut $ waste tracking view" in body

    def test_to_create_sql_sanitizes_manually_built_spec(self) -> None:
        # The render boundary is defensive even for a hand-built spec whose comment
        # was never run through the generator's source-level sanitize.
        spec = MetricViewSpec(
            full_name="austin_choi_omni_agent_catalog.agent_improvement_loop.mv_dq",
            source=L0_CONTRACT.get_table(SESSION_TABLE).fqn,  # type: ignore[union-attr]
            comment="spend $$ here $$$ now",
            dimensions=[MetricViewDimension(name="Model", expr="model")],
            measures=[MetricViewMeasure(name="Cnt", expr="COUNT(1)")],
        )
        sql = spec.to_create_sql()
        assert sql.count("$$") == 2
