"""The ``metric_view`` asset generator — end-to-end, fail-closed.

Given an L3/RLM :class:`~ail.l3.contract.RankedAsset` of type ``metric_view``, this
emits a **valid, deployable** Unity Catalog metric-view spec over the published L0
Delta tables (``docs/L0_METRICS_CONTRACT.md``): a token-efficiency / tool-call
redundancy view whose measures reflect the waste the RLM flagged.

How a recommendation becomes measures:

* a **catalog** of measure templates (:data:`MEASURE_CATALOG`) maps a token-waste
  *concept* (redundancy, tokens-per-task, tool-call volume, cost, …) to a concrete
  aggregate expression and the exact L0 columns it needs;
* the recommendation's free text (title + rationales + expected benefits) is
  keyword-matched against those templates, so the view reflects what the RLM
  actually flagged; a recommendation that names no concept falls back to the
  default token-efficiency set (recorded in notes), and a baseline ``Trace Count``
  measure is always present so the view is queryable.

**Fabrication guard.** Every template declares its ``required_columns``. Before a
measure is emitted, those columns are checked against the real L0 column contract
(:data:`ail.optimize.assets.l0_contract.L0_CONTRACT`). A measure whose backing
column is absent is **dropped with a recorded reason** (:class:`DroppedMeasure`),
never emitted against an invented column. The finished spec is then run through
:func:`validate_spec`, which re-derives the columns referenced by every
expression and rejects any that the contract does not contain — so a spec that
reaches a caller is well-formed and references only real columns. Validation is
static/offline; deploying the view is a separate operational step (no live
Databricks call is made here).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime

import yaml

from ail.l3.contract import CohortReviewReport, RankedAsset
from ail.optimize.assets.asset_contract import (
    DroppedMeasure,
    GeneratedMetricView,
    MetricViewDimension,
    MetricViewMeasure,
    MetricViewSpec,
)
from ail.optimize.assets.base import AssetGenerator
from ail.optimize.assets.l0_contract import (
    L0_CONTRACT,
    SESSION_TABLE,
    ColumnKind,
    L0ColumnContract,
)

__all__ = [
    "GENERATOR_VERSION",
    "MeasureTemplate",
    "MEASURE_CATALOG",
    "DEFAULT_CONCEPTS",
    "SpecValidationError",
    "validate_spec",
    "generate_metric_view",
    "generate_metric_views_from_report",
    "MetricViewGenerator",
]

#: Version of this generator, recorded on every emitted asset for provenance.
GENERATOR_VERSION = "assets.metric_view/v1"


@dataclass(frozen=True, slots=True)
class MeasureTemplate:
    """A token-waste *concept* mapped to a concrete metric-view measure.

    ``required_columns`` are the L0 columns the ``expr`` reads; the fabrication
    guard checks them against the live column contract before emitting the measure.
    ``keywords`` are lowercase substrings that, found in the recommendation text,
    select this measure (so the view reflects the RLM's actual findings).
    """

    concept: str
    name: str
    expr: str
    comment: str
    required_columns: tuple[str, ...]
    keywords: tuple[str, ...] = ()

    def matches(self, text: str) -> bool:
        return any(kw in text for kw in self.keywords)


# Measure catalog over l0_session_metrics (one row per trace): the per-trace fact
# table that carries the token, tool-call, and redundancy signals the L3 rubric is
# about. Ratios are written as SUM/SUM so the metric view re-aggregates them safely
# at query time (a metric view's core advantage over a plain view).
_SESSION_CATALOG: tuple[MeasureTemplate, ...] = (
    MeasureTemplate(
        "trace_count",
        "Trace Count",
        "COUNT(1)",
        "Number of traces (the denominator for per-trace efficiency measures).",
        (),
        ("trace", "task", "session", "volume", "throughput"),
    ),
    MeasureTemplate(
        "total_tokens",
        "Total Tokens",
        "SUM(total_tokens)",
        "Total tokens consumed across traces.",
        ("total_tokens",),
        ("token", "spend", "usage"),
    ),
    MeasureTemplate(
        "tokens_per_trace",
        "Tokens per Trace",
        "SUM(total_tokens) / NULLIF(COUNT(1), 0)",
        "Average tokens per trace — the headline tokens-per-task efficiency measure.",
        ("total_tokens",),
        ("token", "per task", "per-task", "per trace", "efficien"),
    ),
    MeasureTemplate(
        "output_tokens",
        "Output Tokens",
        "SUM(output_tokens)",
        "Total model output (generation) tokens across traces.",
        ("output_tokens",),
        ("output token", "generation token"),
    ),
    MeasureTemplate(
        "cache_tokens",
        "Cache Tokens",
        "SUM(cache_total_tokens)",
        "Total cache read+write tokens — churn here signals re-ingested context.",
        ("cache_total_tokens",),
        ("cache",),
    ),
    MeasureTemplate(
        "total_tool_calls",
        "Total Tool Calls",
        "SUM(total_tool_calls)",
        "Total tool calls across traces.",
        ("total_tool_calls",),
        ("tool call", "tool-call", "tool use", "tool invocation"),
    ),
    MeasureTemplate(
        "tool_calls_per_trace",
        "Tool Calls per Trace",
        "SUM(total_tool_calls) / NULLIF(COUNT(1), 0)",
        "Average tool calls per trace.",
        ("total_tool_calls",),
        ("tool call", "tool-call", "per trace"),
    ),
    MeasureTemplate(
        "redundant_tool_calls",
        "Redundant Tool Calls",
        "SUM(redundant_tool_calls)",
        "Total byte-identical repeated tool calls (re-reads / re-run boilerplate).",
        ("redundant_tool_calls",),
        ("redundan", "repeat", "re-run", "rerun", "re-read", "reread", "duplicat"),
    ),
    MeasureTemplate(
        "redundancy_rate",
        "Redundant Tool Call Rate",
        "SUM(redundant_tool_calls) / NULLIF(SUM(total_tool_calls), 0)",
        "Share of tool calls that were byte-identical repeats (re-aggregatable).",
        ("redundant_tool_calls", "total_tool_calls"),
        ("redundan", "waste", "repeat", "avoidable"),
    ),
    MeasureTemplate(
        "est_cost_usd",
        "Estimated Cost (USD)",
        "SUM(est_cost_usd) FILTER (WHERE cost_priced)",
        "Estimated USD cost over priced traces only (unpriced traces excluded).",
        ("est_cost_usd", "cost_priced"),
        ("cost", "dollar", "spend", "usd", "price"),
    ),
    MeasureTemplate(
        "avg_duration",
        "Avg Duration (seconds)",
        "AVG(duration_seconds)",
        "Average wall-clock trace duration in seconds.",
        ("duration_seconds",),
        ("latency", "duration", "slow", "wall clock", "wall-clock"),
    ),
)

#: The single measure catalog (only ``l0_session_metrics`` is generated against in
#: this stage; a diagnosis-sourced view for path-level repeated reads is a natural
#: extension — see ``docs/ASSET_GENERATOR.md``).
MEASURE_CATALOG: tuple[MeasureTemplate, ...] = _SESSION_CATALOG

#: Fallback measures when a recommendation names no recognisable concept — a
#: sensible token-efficiency default rather than an empty or guessed view.
DEFAULT_CONCEPTS: tuple[str, ...] = (
    "tokens_per_trace",
    "total_tool_calls",
    "redundancy_rate",
)

_CATALOG_BY_CONCEPT: dict[str, MeasureTemplate] = {t.concept: t for t in MEASURE_CATALOG}
_BASELINE_CONCEPT = "trace_count"


@dataclass(frozen=True, slots=True)
class _DimensionTemplate:
    name: str
    column: str
    comment: str


# Default slicing dimensions, all real categorical/temporal session columns.
_SESSION_DIMENSIONS: tuple[_DimensionTemplate, ...] = (
    _DimensionTemplate("Model", "model", "Model that produced the trace."),
    _DimensionTemplate("Producer", "producer", "Agent runtime that produced the trace."),
    _DimensionTemplate("Status", "status", "Trace terminal status (OK/ERROR/...)."),
    _DimensionTemplate("Request Time", "request_time", "Trace request time (ISO-8601)."),
)


# --- validation -------------------------------------------------------------

_AGG_FUNCS = frozenset({"sum", "count", "avg", "min", "max"})
#: Non-column SQL tokens that may appear in our generated expressions. Anything
#: left after removing these must be a real column (the fabrication check).
_SQL_KEYWORDS = _AGG_FUNCS | frozenset(
    {
        "distinct",
        "filter",
        "where",
        "nullif",
        "coalesce",
        "case",
        "when",
        "then",
        "else",
        "end",
        "and",
        "or",
        "not",
        "null",
        "cast",
        "as",
        "date_trunc",
        "extract",
        "year",
        "month",
        "day",
        "true",
        "false",
    }
)

_STR_LITERAL = re.compile(r"'[^']*'")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _expr_tokens(expr: str) -> set[str]:
    """Lowercased identifier tokens in ``expr`` with string literals stripped."""
    cleaned = _STR_LITERAL.sub(" ", expr)
    return {m.lower() for m in _IDENT.findall(cleaned)}


def _referenced_columns(expr: str) -> set[str]:
    """Identifier tokens that are not SQL keywords — i.e. column references."""
    return _expr_tokens(expr) - _SQL_KEYWORDS


class SpecValidationError(ValueError):
    """A generated spec failed offline validation; carries every problem found."""

    def __init__(self, problems: Iterable[str]) -> None:
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


def validate_spec(spec: MetricViewSpec, contract: L0ColumnContract = L0_CONTRACT) -> None:
    """Validate ``spec`` offline; raise :class:`SpecValidationError` if unusable.

    Checks (all static — no Databricks call): a supported YAML version; a
    3-level ``source`` that names a real L0 table; at least one dimension and one
    measure with unique names; every measure carries an aggregate and every
    dimension does not; **every column referenced by any expression exists in the
    L0 contract** (the fabrication gate); and the rendered YAML round-trips.
    """
    problems: list[str] = []

    if spec.version not in {"0.1", "1.1"}:
        problems.append(f"unsupported metric-view YAML version {spec.version!r}")

    if len(spec.source.split(".")) != 3:
        problems.append(f"source {spec.source!r} must be a 3-level catalog.schema.table name")
    table = contract.table_for(spec.source)
    if table is None:
        problems.append(f"source {spec.source!r} is not a known L0 table")

    if not spec.dimensions:
        problems.append("at least one dimension is required")
    if not spec.measures:
        problems.append("at least one measure is required")

    dim_names = [d.name for d in spec.dimensions]
    if len(set(dim_names)) != len(dim_names):
        problems.append("dimension names must be unique")
    measure_names = [m.name for m in spec.measures]
    if len(set(measure_names)) != len(measure_names):
        problems.append("measure names must be unique")

    if table is not None:
        valid = set(table.column_names)
        for m in spec.measures:
            if not (_AGG_FUNCS & _expr_tokens(m.expr)):
                problems.append(f"measure {m.name!r} has no aggregate function")
            bad = _referenced_columns(m.expr) - valid
            if bad:
                problems.append(
                    f"measure {m.name!r} references unknown column(s) {sorted(bad)} "
                    f"not in {table.name}"
                )
        for d in spec.dimensions:
            if _AGG_FUNCS & _expr_tokens(d.expr):
                problems.append(f"dimension {d.name!r} must not use an aggregate function")
            bad = _referenced_columns(d.expr) - valid
            if bad:
                problems.append(
                    f"dimension {d.name!r} references unknown column(s) {sorted(bad)} "
                    f"not in {table.name}"
                )

    try:
        reparsed = yaml.safe_load(spec.to_yaml())
    except yaml.YAMLError as exc:  # pragma: no cover - safe_dump output is valid YAML
        problems.append(f"rendered YAML is not well-formed: {exc}")
    else:
        if not isinstance(reparsed, dict) or "measures" not in reparsed:
            problems.append("rendered YAML did not round-trip to a mapping with measures")

    if problems:
        raise SpecValidationError(problems)


# --- generation -------------------------------------------------------------


@dataclass(slots=True)
class _Selection:
    """Outcome of resolving a recommendation's text to measure templates."""

    templates: list[MeasureTemplate] = field(default_factory=list)
    used_default: bool = False


def _select_templates(text: str) -> _Selection:
    """Pick measure templates from recommendation ``text`` (keyword match)."""
    matched = [t for t in MEASURE_CATALOG if t.matches(text)]
    meaningful = [t for t in matched if t.concept != _BASELINE_CONCEPT]
    used_default = False
    if not meaningful:
        matched = [_CATALOG_BY_CONCEPT[c] for c in DEFAULT_CONCEPTS]
        used_default = True
    # Always include the baseline trace count so the view is queryable and the
    # per-trace ratios have a denominator.
    if all(t.concept != _BASELINE_CONCEPT for t in matched):
        matched = [_CATALOG_BY_CONCEPT[_BASELINE_CONCEPT], *matched]
    return _Selection(templates=matched, used_default=used_default)


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(title: str) -> str:
    base = _SLUG_RE.sub("_", title.lower()).strip("_")
    if not base:
        base = "token_efficiency"
    if not base.startswith("mv_"):
        base = f"mv_{base}"
    return base[:60].rstrip("_")


def _recommendation_text(asset: RankedAsset) -> str:
    return " ".join([asset.title, *asset.rationales, *asset.expected_benefits]).lower()


def generate_metric_view(
    asset: RankedAsset,
    *,
    contract: L0ColumnContract = L0_CONTRACT,
    source_table: str = SESSION_TABLE,
    full_name: str | None = None,
    generated_at: str | None = None,
) -> GeneratedMetricView:
    """Generate a validated metric-view asset from a ranked ``metric_view`` rec.

    Args:
        asset: The L3/RLM ranked recommendation (must be ``asset_type ==
            "metric_view"``).
        contract: The L0 column contract to build and validate against. Defaults
            to the real published contract; inject a restricted one to exercise the
            fabrication guard offline.
        source_table: Which L0 table the view aggregates (default
            ``l0_session_metrics``).
        full_name: Override the generated view's UC name; defaults to
            ``<catalog>.<schema>.mv_<slug-of-title>`` in the source table's schema.
        generated_at: ISO-8601 stamp (defaults to now, UTC).

    Returns:
        A :class:`~ail.optimize.assets.asset_contract.GeneratedMetricView` whose
        ``spec`` has passed :func:`validate_spec`.

    Raises:
        ValueError: if ``asset`` is not a ``metric_view`` or ``source_table`` is
            unknown to ``contract``.
        SpecValidationError: if the resulting spec is not usable (fail-closed:
            an invalid spec is never returned).
    """
    if asset.asset_type != "metric_view":
        raise ValueError(
            f"metric_view generator received asset_type {asset.asset_type!r}; "
            "expected 'metric_view'"
        )
    table = contract.get_table(source_table)
    if table is None:
        raise ValueError(f"unknown L0 source table {source_table!r}")

    selection = _select_templates(_recommendation_text(asset))

    measures: list[MetricViewMeasure] = []
    dropped: list[DroppedMeasure] = []
    for template in selection.templates:
        missing = [c for c in template.required_columns if not table.has(c)]
        if missing:
            dropped.append(
                DroppedMeasure(
                    name=template.name,
                    concept=template.concept,
                    reason=(
                        f"L0 table {table.name} has no backing column(s) "
                        f"{missing}; measure omitted (no fabrication)"
                    ),
                    missing_columns=missing,
                )
            )
            continue
        measures.append(
            MetricViewMeasure(name=template.name, expr=template.expr, comment=template.comment)
        )

    dimensions = [
        MetricViewDimension(name=d.name, expr=d.column, comment=d.comment)
        for d in _SESSION_DIMENSIONS
        if table.has(d.column)
        and table.get(d.column).kind  # type: ignore[union-attr]
        in (ColumnKind.CATEGORICAL, ColumnKind.TEMPORAL, ColumnKind.IDENTIFIER)
    ]

    view_name = full_name or f"{table.catalog}.{table.schema}.{_slug(asset.title)}"
    comment = (
        f"Auto-generated (Stage 6) from L3/RLM recommendation rank {asset.rank}"
        f" (recurs across {asset.n_traces} trace(s)): {asset.title}."
        f" Token-efficiency / tool-call-redundancy metrics over {table.name}."
    )
    spec = MetricViewSpec(
        version="1.1",
        full_name=view_name,
        source=table.fqn,
        comment=comment,
        dimensions=dimensions,
        measures=measures,
    )

    # Fail-closed: validate before returning; never hand back an unusable spec.
    validate_spec(spec, contract)

    notes: list[str] = []
    if selection.used_default:
        notes.append(
            "recommendation named no recognisable measure concept; generated the "
            "default token-efficiency measure set"
        )
    if dropped:
        notes.append(
            f"{len(dropped)} requested measure(s) dropped: no backing L0 column (fabrication guard)"
        )

    return GeneratedMetricView(
        generator_version=GENERATOR_VERSION,
        source_rank=asset.rank,
        source_title=asset.title,
        source_trace_ids=list(asset.trace_ids),
        n_source_traces=asset.n_traces,
        matched_concepts=[t.concept for t in selection.templates],
        notes=notes,
        generated_at=generated_at or datetime.now(UTC).isoformat(),
        spec=spec,
        dropped_measures=dropped,
    )


def generate_metric_views_from_report(
    report: CohortReviewReport,
    *,
    contract: L0ColumnContract = L0_CONTRACT,
    source_table: str = SESSION_TABLE,
    generated_at: str | None = None,
) -> list[GeneratedMetricView]:
    """Generate a metric view for every ``metric_view`` recommendation in a report."""
    return [
        generate_metric_view(
            asset,
            contract=contract,
            source_table=source_table,
            generated_at=generated_at,
        )
        for asset in report.ranked_assets
        if asset.asset_type == "metric_view"
    ]


class MetricViewGenerator(AssetGenerator):
    """The registered ``metric_view`` generator (see :func:`generate_metric_view`)."""

    asset_type = "metric_view"

    def generate(self, asset: RankedAsset, **options: object) -> GeneratedMetricView:
        return generate_metric_view(asset, **options)  # type: ignore[arg-type]
