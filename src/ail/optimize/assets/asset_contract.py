"""Typed output contract for generated helper assets.

The generator turns an L3 :class:`~ail.l3.contract.RankedAsset` into a concrete,
deployable artifact. These are the typed shapes it returns. Like the L0/L2/L3
contracts they are pydantic v2 models that forbid unknown fields (drift is loud)
and round-trip through JSON without custom serialization, so a generated asset can
be persisted and re-read by the orchestrator that deploys it.

:class:`GeneratedAsset` is the small base every generated artifact shares (its
type, the generator version, provenance back to the ranked recommendation, and
free-text notes). :class:`GeneratedMetricView` is the metric-view specialization:
it carries the validated :class:`MetricViewSpec` (the deployable Unity Catalog
metric-view definition) plus the :class:`DroppedMeasure` records — measures the
recommendation asked for but that no real L0 column backs, omitted **with a
reason** rather than fabricated.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "MetricViewDimension",
    "MetricViewMeasure",
    "MetricViewSpec",
    "DroppedMeasure",
    "GeneratedAsset",
    "GeneratedMetricView",
    "strip_dollar_quote",
]

#: A run of two-or-more ``$`` — the token that closes a SQL ``$$``-quoted block.
_DOLLAR_QUOTE_RUN = re.compile(r"\${2,}")


def strip_dollar_quote(text: str) -> str:
    """Collapse any ``$$`` run to a single ``$`` so it cannot close a dollar-quote.

    The metric-view DDL embeds the YAML body in a ``CREATE ... AS $$ ... $$``
    dollar-quoted block. Free-text that lands inside it (a ``comment`` derived from
    an L3/RLM *recommendation title* — untrusted LLM output) could contain ``$$``
    (e.g. "Cut $$ waste") and prematurely terminate the quote, emitting broken SQL.
    Collapsing every ``$$+`` run to one ``$`` removes the delimiter while keeping a
    harmless single ``$``; it is idempotent and generalises to ``$$$`` etc.
    """
    return _DOLLAR_QUOTE_RUN.sub("$", text)


class _Contract(BaseModel):
    """Base for every contract model: forbid unknown fields so drift is loud."""

    model_config = ConfigDict(extra="forbid")


class MetricViewDimension(_Contract):
    """One metric-view dimension: a categorical/temporal attribute to group by."""

    name: str
    expr: str
    comment: str = ""


class MetricViewMeasure(_Contract):
    """One metric-view measure: an aggregate expression queried via ``MEASURE()``."""

    name: str
    expr: str
    comment: str = ""


class MetricViewSpec(_Contract):
    """A Unity Catalog metric-view definition — the deployable artifact.

    ``source`` is the fully-qualified L0 table the view aggregates;
    ``dimensions`` / ``measures`` are the YAML body. :meth:`to_yaml` renders the
    ``WITH METRICS LANGUAGE YAML`` body and :meth:`to_create_sql` wraps it in the
    full ``CREATE OR REPLACE VIEW`` statement an operator runs to deploy it. The
    spec is validated (:func:`ail.optimize.assets.metric_view.validate_spec`)
    before a :class:`GeneratedMetricView` is returned, so a spec that reaches a
    caller is always well-formed and references only real columns.
    """

    version: str = "1.1"
    full_name: str
    source: str
    comment: str = ""
    filter: str = ""
    dimensions: list[MetricViewDimension] = Field(default_factory=list)
    measures: list[MetricViewMeasure] = Field(default_factory=list)

    @property
    def qualified_name_sql(self) -> str:
        """``full_name`` with each identifier back-tick quoted for SQL."""
        return ".".join(f"`{part}`" for part in self.full_name.split("."))

    def to_doc(self) -> dict[str, Any]:
        """The metric-view YAML as a plain dict (ordered as UC expects)."""
        doc: dict[str, Any] = {"version": self.version, "source": self.source}
        if self.comment:
            doc["comment"] = self.comment
        if self.filter:
            doc["filter"] = self.filter
        doc["dimensions"] = [_entry(d.name, d.expr, d.comment) for d in self.dimensions]
        doc["measures"] = [_entry(m.name, m.expr, m.comment) for m in self.measures]
        return doc

    def to_yaml(self) -> str:
        """Render the metric-view definition as YAML (the ``LANGUAGE YAML`` body)."""
        return yaml.safe_dump(
            self.to_doc(),
            sort_keys=False,
            default_flow_style=False,
            width=4096,
            allow_unicode=True,
        )

    def to_create_sql(self) -> str:
        """The full ``CREATE OR REPLACE VIEW ... WITH METRICS LANGUAGE YAML`` DDL.

        The rendered YAML body is run through :func:`strip_dollar_quote` so no
        ``$$`` it may contain (e.g. from a recommendation-derived ``comment``) can
        prematurely close the ``$$``-quoted block — a final safety net independent
        of how the spec was constructed.
        """
        body = strip_dollar_quote(self.to_yaml().rstrip())
        return (
            f"CREATE OR REPLACE VIEW {self.qualified_name_sql}\n"
            "WITH METRICS\n"
            "LANGUAGE YAML\n"
            "AS $$\n"
            f"{body}\n"
            "$$"
        )


def _entry(name: str, expr: str, comment: str) -> dict[str, str]:
    entry = {"name": name, "expr": expr}
    if comment:
        entry["comment"] = comment
    return entry


class DroppedMeasure(_Contract):
    """A measure the recommendation implied but that no real L0 column backs.

    The fabrication guard: rather than invent a column to satisfy the
    recommendation, the generator omits the measure and records *why* here
    (``missing_columns`` names the L0 columns that were required but absent). A
    consumer sees exactly what was asked for and could not be honoured.
    """

    name: str
    concept: str
    reason: str
    missing_columns: list[str] = Field(default_factory=list)


class GeneratedAsset(_Contract):
    """Base for any generated helper asset: type, generator version, provenance."""

    asset_type: str
    generator_version: str
    source_rank: int | None = None
    source_title: str = ""
    source_trace_ids: list[str] = Field(default_factory=list)
    n_source_traces: int = 0
    notes: list[str] = Field(default_factory=list)
    generated_at: str | None = None  # ISO-8601


class GeneratedMetricView(GeneratedAsset):
    """A generated, validated Unity Catalog metric view ready to deploy.

    ``spec`` is the deployable definition; ``dropped_measures`` records the
    fabrication-guard omissions; ``matched_concepts`` names which measure concepts
    the recommendation's text selected (or that the default set supplied).
    """

    asset_type: str = "metric_view"
    spec: MetricViewSpec
    dropped_measures: list[DroppedMeasure] = Field(default_factory=list)
    matched_concepts: list[str] = Field(default_factory=list)

    def write(self, out_dir: str | Path) -> dict[str, str]:
        """Write the deployable ``.sql`` and the typed ``.json`` to ``out_dir``.

        Returns a ``{"sql": path, "json": path}`` map of what was written. The SQL
        is the operator-runnable ``CREATE`` statement; the JSON is this object
        (spec + provenance + dropped measures) for the orchestrator's records.
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        slug = self.spec.full_name.split(".")[-1]
        sql_path = out / f"{slug}.sql"
        json_path = out / f"{slug}.metric_view.json"
        sql_path.write_text(self.spec.to_create_sql() + "\n", encoding="utf-8")
        json_path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return {"sql": str(sql_path), "json": str(json_path)}
