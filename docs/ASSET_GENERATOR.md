# Helper-Asset Generator (Stage 6)

**Status:** `metric_view` implemented end-to-end · other types are `next` stubs.

Stage 6 turns the L3/RLM ranked recommendations into real, deployable Databricks
assets. The L3 cohort review (`ail.l3.cohort_review`) produces a
`CohortReviewReport` whose `ranked_assets` is a recurrence-ranked list of
`RankedAsset` recommendations (`metric_view`, `skill`, `tool`, `prompt_change`,
…). This generator consumes that list and emits concrete artifacts.

Scope is deliberately tight: **one asset type end-to-end** (`metric_view`) plus an
extensible seam. The other types raise a clear `next` signal rather than emitting
half-formed assets.

## The seam — `ail.optimize.assets.base`

| Piece | What it is |
|-------|-----------|
| `AssetGenerator` (ABC) | One generator per `asset_type`; `generate(asset, **opts) -> GeneratedAsset`. |
| `register` / `get_generator` / `registered_asset_types` | The registry keyed by `asset_type`. |
| `generate_asset(asset, **opts)` | Dispatch a `RankedAsset` to its generator. |
| `AssetGeneratorNotImplemented` | The `next` signal (`.status == "next"`, `.asset_type`) for a recognised-but-unbuilt type. |

Importing the package registers the `metric_view` generator and `next`
placeholders for `skill` / `tool` / `prompt_change`. Dispatching any other type
(e.g. `semantic_layer`) raises the same `AssetGeneratorNotImplemented` rather than
a bare `KeyError`.

```python
from ail.optimize.assets import generate_asset, generate_metric_views_from_report

# Single recommendation:
asset = generate_asset(ranked_asset)            # metric_view -> GeneratedMetricView
# Whole report:
views = generate_metric_views_from_report(cohort_review_report)
```

## The `metric_view` generator — `ail.optimize.assets.metric_view`

Given a `metric_view` recommendation, it emits a valid Unity Catalog metric-view
spec over the published **L0** Delta tables
(`austin_choi_omni_agent_catalog.agent_improvement_loop.l0_*`, see
`docs/L0_METRICS_CONTRACT.md`).

**Recommendation → measures.** A catalog of measure *templates*
(`MEASURE_CATALOG`) maps a token-waste concept to a concrete aggregate expression
and the exact L0 columns it needs. The recommendation's free text (title +
rationales + expected benefits) is keyword-matched against those templates, so the
view reflects the waste the RLM actually flagged. A recommendation that names no
concept falls back to a default token-efficiency set (recorded in `notes`), and a
baseline `Trace Count` (`COUNT(1)`) measure is always present so the view is
queryable and the per-trace ratios have a denominator.

The view is built over `l0_session_metrics` (one row per trace — the per-trace
fact carrying token, tool-call, and redundancy signals). Ratios are written as
`SUM(...) / NULLIF(SUM(...), 0)` so the metric view re-aggregates them safely at
query time.

### Measures (over `l0_session_metrics`)

| Concept | Measure | Expression |
|---------|---------|------------|
| `trace_count` | Trace Count | `COUNT(1)` |
| `total_tokens` | Total Tokens | `SUM(total_tokens)` |
| `tokens_per_trace` | Tokens per Trace | `SUM(total_tokens) / NULLIF(COUNT(1), 0)` |
| `output_tokens` | Output Tokens | `SUM(output_tokens)` |
| `cache_tokens` | Cache Tokens | `SUM(cache_total_tokens)` |
| `total_tool_calls` | Total Tool Calls | `SUM(total_tool_calls)` |
| `tool_calls_per_trace` | Tool Calls per Trace | `SUM(total_tool_calls) / NULLIF(COUNT(1), 0)` |
| `redundant_tool_calls` | Redundant Tool Calls | `SUM(redundant_tool_calls)` |
| `redundancy_rate` | Redundant Tool Call Rate | `SUM(redundant_tool_calls) / NULLIF(SUM(total_tool_calls), 0)` |
| `est_cost_usd` | Estimated Cost (USD) | `SUM(est_cost_usd) FILTER (WHERE cost_priced)` |
| `avg_duration` | Avg Duration (seconds) | `AVG(duration_seconds)` |

Dimensions: `Model`, `Producer`, `Status`, `Request Time` (all real session
columns).

## Fail-closed / no fabrication

Two layers, both static/offline — **no live Databricks call is made in code or
tests.** Deploying the view is a separate operational step the orchestrator runs.

1. **Real-column allow-list.** `ail.optimize.assets.l0_contract` is the typed
   registry of the real L0 columns. `verify_against_publish()` asserts its column
   names are *exactly* `ail.publish`'s `*_COLUMNS` (the source of truth) and that
   the catalog/schema/table constants agree; a test calls it, so any drift in
   `publish` fails loudly rather than letting the generator reference a stale
   column.
2. **Fabrication guard.** Each measure template declares its `required_columns`.
   A measure whose backing column is absent from the contract is **dropped with a
   recorded reason** (`DroppedMeasure`), never emitted against an invented column.
3. **Spec validation.** `validate_spec` re-derives the columns referenced by every
   expression and rejects any not in the contract; it also checks a supported YAML
   version, a 3-level `source` naming a real L0 table, ≥1 dimension and ≥1 measure
   with unique names, that every measure carries an aggregate and no dimension
   does, and that the rendered YAML round-trips. The generator runs this before
   returning, so an unusable spec is never handed back.

## Output

`generate_metric_view` returns a `GeneratedMetricView` (typed, JSON round-trips):

- `spec` — the deployable `MetricViewSpec` (`to_yaml()`, `to_create_sql()`).
- `dropped_measures` — fabrication-guard omissions, each with a reason.
- `matched_concepts`, `source_rank`/`source_title`/`source_trace_ids`, `notes`.

`GeneratedMetricView.write(out_dir)` writes the operator-runnable `<slug>.sql`
(the `CREATE OR REPLACE VIEW ... WITH METRICS LANGUAGE YAML` statement) and the
typed `<slug>.metric_view.json`. A generated example is checked in at
`artifacts/generated/`.

## Deploying (operational, outside this module)

The generated SQL is run by an operator / orchestrator against a SQL warehouse —
e.g. via the `manage_metric_views` capability or the `databricks-metric-views`
patterns — after the L0 tables have been published (`ail.publish`). This module
deliberately makes **no** live calls.

## Next

`skill`, `tool`, and `prompt_change` generators are registered as `next` stubs.
A diagnosis-sourced metric view (path-level repeated reads over `l0_diagnosis`,
keyed on `signature_kind = 'path'`) is a natural extension of the metric-view
generator.
