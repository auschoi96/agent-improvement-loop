"""Tier A — publish the **unified, per-agent-version** comparison tables.

This is the publish extension behind the observability app's priority visual: a
**live baseline-vs-new VERSION comparison** (``docs/OBSERVABILITY_APP.md`` Phase
B). It federates at publish time and presents a single pane at query time:

* The L0 numbers are **not recomputed here** — they are the already-Python-computed
  outputs carried in a Phase-2 :class:`~ail.optimize.phase2.Phase2Artifact` (which
  the comparison harness produced via :func:`ail.metrics.compute_l0` per arm). This
  module only *aggregates* those per-task L0 values to the version level (the exact
  extensive sums / rate formula :mod:`ail.metrics` itself uses) and shapes flat,
  analytics-friendly rows. The app (Tier B) only ``SELECT``s from these tables.
* It is **honest per the readiness/trust model** (``docs/READINESS_AND_TRUST.md``).
  It wires :func:`ail.readiness.compute_readiness` so the comparison reflects *real*
  readiness, and emits a single Python-decided :class:`VersionComparisonStatus` so
  the app never has to decide trust in SQL/TS. A version whose organic readiness
  wall has not cleared is **not** shown as a green "improvement" — it is shown as
  ``CONTROLLED_PROOF_COLLECTING`` (a real, measured controlled-comparison delta
  whose organic confirmation is still collecting) or ``COLLECTING``. The delta is
  measured, never fabricated; the *trust treatment* is gated by readiness.

Unified tables written to ``<catalog>.<schema>`` (keyed by ``agent_name`` +
``agent_version`` — one set of tables for all agents, segmented in SQL):

* ``agent_registry`` — the registered agents (:mod:`ail.registry`): name →
  experiment + optional judge/tag config. The app's agent switcher reads this.
* ``agent_version_l0`` — one row per (agent, version): the version's L0 aggregate
  (tokens, tokens/trace, tool calls, redundancy, cost) over its counted traces.
* ``agent_version_comparison`` — one row per (agent, baseline_version,
  candidate_version, metric): the version-over-version :class:`MetricDelta`.
* ``agent_version_readiness`` — one row per (agent, baseline_version,
  candidate_version): the readiness tier, the trust-gated display status, the
  controlled-proof header (PROMOTE/BLOCK counts, correctness-held), and the
  headline objective delta.

Writes reuse :mod:`ail.publish`'s atomic, idempotent staging→``REPLACE WHERE``
swap, here scoped by a **composite** ``agent_name``/``agent_version`` predicate so a
re-publish of one version never disturbs another (or another agent). The seed
source is a **committed artifact**, so the comparison view renders real data even
when live trace auth is unavailable.

Run (seed the Phase-2 controlled result for the Claude Code agent)::

    python -m ail.publish_versions \\
        --artifact artifacts/phase2_token_lever.json \\
        --agent claude_code \\
        --baseline-version v0-baseline-no-skill \\
        --candidate-version v1-token-efficiency-skill \\
        --warehouse-id <SQL_WAREHOUSE_ID> --profile dais-demo
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ail.cohorts import TAG_AGENT, Cohort
from ail.compare.contract import MetricDelta, Recommendation
from ail.optimize.phase2 import L1Outcome, Phase2Artifact, TaskOutcome
from ail.publish import (
    DEFAULT_CATALOG,
    DEFAULT_SCHEMA,
    _atomic_replace_table,
    _build_workspace_client,
    _execute,
    _lit,
)
from ail.readiness import (
    ReadinessFacts,
    ReadinessStatus,
    ReadinessThresholds,
    compute_readiness,
)
from ail.registry import (
    TAG_AGENT_VERSION,
    Agent,
    AgentRegistry,
    load_registry,
)

SCHEMA_VERSION = "ail.observability/v1"

REGISTRY_TABLE = "agent_registry"
VERSION_L0_TABLE = "agent_version_l0"
VERSION_COMPARISON_TABLE = "agent_version_comparison"
VERSION_READINESS_TABLE = "agent_version_readiness"

#: How the Phase-2 controlled comparison was produced — recorded as provenance so
#: a reader knows the delta came from the frozen-suite, L1-gated harness, not an
#: organic trace trend.
PROOF_SOURCE_PHASE2 = "controlled_comparison_frozen_suite"
SOURCE_PHASE2 = "phase2_controlled_comparison"
#: The L0 aggregate is over the PROMOTE (objective-met + correctness-held) tasks
#: only — the same set the artifact's ``realized_*`` headline is summed over. A
#: blocked/crashed task's tokens are never counted as a win.
BASIS_PROMOTE = "promote_correctness_held"


# ---------------------------------------------------------------------------
# Output contract (typed, JSON-shaped; the app reads the flat tables below)
# ---------------------------------------------------------------------------


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VersionComparisonStatus(StrEnum):
    """The trust-gated display status for one version-over-version comparison.

    Decided **in Python** (never in SQL/TS) from the measured delta *and* the
    readiness wall, so the app renders a status, not a verdict it computed:

    * :attr:`PROVEN` — the objective improved, correctness held, **and** the
      readiness wall cleared (``READY_TO_PROVE``). The only green state.
    * :attr:`CONTROLLED_PROOF_COLLECTING` — the controlled comparison proved the
      improvement on the frozen suite with correctness held, but organic readiness
      has **not** cleared yet. The real delta is shown; it is *not* green —
      organic confirmation is still collecting (the honest dual-signal state).
    * :attr:`COLLECTING` — no controlled proof yet (no PROMOTE task), or not enough
      data. No improvement claim.
    * :attr:`REGRESSED` — the candidate is worse (objective not improved, or a
      correctness regression). Never reads as a win.
    """

    PROVEN = "proven_improvement"
    CONTROLLED_PROOF_COLLECTING = "controlled_proof_collecting"
    COLLECTING = "collecting"
    REGRESSED = "regressed"


class AgentVersionAggregate(_Contract):
    """One agent version's L0 aggregate over its counted (PROMOTE) traces.

    Every number is an extensive sum (or the ``redundant/total`` rate) of the
    per-task L0 values :func:`ail.metrics.compute_l0` already produced — no metric
    is recomputed here. ``cost_priced`` is ``False`` when any counted trace was
    unpriced; the dollar total is then a floor, surfaced honestly (never inflated).
    """

    schema_version: str = SCHEMA_VERSION
    agent_name: str
    agent_version: str
    experiment_id: str | None = None
    source: str = SOURCE_PHASE2
    basis: str = BASIS_PROMOTE
    n_traces: int = 0
    n_traces_total: int = 0
    input_tokens: float = 0.0
    output_tokens: float = 0.0
    total_tokens: float = 0.0
    cache_total_tokens: float = 0.0
    tokens_per_trace: float = 0.0
    total_tool_calls: float = 0.0
    redundant_tool_calls: float = 0.0
    redundancy_rate: float = 0.0
    total_cost_usd: float = 0.0
    cost_priced: bool = False
    generated_at: str | None = None


class VersionComparison(_Contract):
    """The headline of one baseline-vs-candidate version comparison.

    Carries the trust-gated :attr:`status`, the embedded real
    :class:`~ail.readiness.contract.ReadinessStatus`, the controlled-proof header
    (PROMOTE/BLOCK counts, correctness-held), the headline objective delta, and the
    per-metric :class:`MetricDelta` list the comparison cards render.
    """

    schema_version: str = SCHEMA_VERSION
    agent_name: str
    baseline_version: str
    candidate_version: str
    objective_metric: str = "total_tokens"
    status: VersionComparisonStatus = VersionComparisonStatus.COLLECTING
    proof_source: str = PROOF_SOURCE_PHASE2
    n_promote: int = 0
    n_block: int = 0
    n_errored: int = 0
    correctness_held: bool = False
    frozen_suite_present: bool = False
    headline_metric: str = "total_tokens"
    headline_baseline: float = 0.0
    headline_candidate: float = 0.0
    headline_delta_absolute: float = 0.0
    headline_delta_pct: float | None = None
    headline_improved: bool = False
    readiness: ReadinessStatus
    deltas: list[MetricDelta] = Field(default_factory=list)
    generated_at: str | None = None
    notes: list[str] = Field(default_factory=list)


class VersionPublishBundle(_Contract):
    """Everything one version comparison contributes to the unified tables."""

    aggregates: list[AgentVersionAggregate] = Field(default_factory=list)
    comparison: VersionComparison


# ---------------------------------------------------------------------------
# Builders: Phase-2 artifact -> unified records (pure; no I/O)
# ---------------------------------------------------------------------------

#: Comparison metrics the view renders, in display order. ``tokens_per_trace`` is
#: derived (total/​n_traces); the rest are read straight from the per-task
#: ``comparison.deltas``. All are lower-is-better.
_COMPARISON_METRICS: tuple[tuple[str, str], ...] = (
    ("total_tokens", "tokens"),
    ("tokens_per_trace", "tokens"),
    ("total_tool_calls", "calls"),
    ("redundancy_rate", "rate"),
    ("total_usd", "usd"),
)


@dataclass(frozen=True)
class _DeterministicGoal:
    """A minimal :class:`ail.readiness.goal.GoalView` for a token/cost objective.

    The Phase-2 objective is a deterministic L0 token reduction guarded by a
    programmatic L1 correctness check (``NO_LLM_JUDGE``), so ``requires_quality`` is
    ``False`` — readiness evaluates the universal trace + frozen-suite gates, not
    the judged-quality gates.
    """

    objective_metric: str
    guardrail_names: tuple[str, ...] = ()
    requires_quality: bool = False


def _promote_outcomes(artifact: Phase2Artifact) -> list[TaskOutcome]:
    """The objective-met + correctness-held tasks — the counted set."""
    return [o for o in artifact.outcomes if o.recommendation is Recommendation.PROMOTE]


def _arm_metric(outcome: TaskOutcome, metric: str, *, candidate: bool) -> float:
    """One arm's value of ``metric`` for ``outcome`` from its carried comparison.

    Reads the already-Python-computed L0 value out of the per-task
    :class:`~ail.compare.contract.ComparisonResult` (never recomputed). Returns
    ``0.0`` when the comparison or the metric is absent (a PROMOTE outcome always
    carries both; the guard keeps a malformed artifact from raising).
    """
    if outcome.comparison is None:
        return 0.0
    delta = outcome.comparison.delta_for(metric)
    if delta is None:
        return 0.0
    return delta.candidate if candidate else delta.baseline


def _reconstruct_redundant(rate: float, total_tool_calls: float) -> float:
    """Recover a trace's redundant-call count from its rate × total.

    :attr:`~ail.metrics.contract.ToolRedundancy.redundancy_rate` is
    ``redundant_tool_calls / total_tool_calls`` by construction, so ``rate × total``
    recovers the integer count exactly. Summing these and re-dividing reproduces the
    *same* aggregate formula :func:`ail.metrics.compute_l0` uses — this is
    re-aggregation of L0 outputs, not a new metric.
    """
    return round(rate * total_tool_calls)


def _aggregate_version(
    outcomes: list[TaskOutcome],
    *,
    candidate: bool,
    agent_name: str,
    agent_version: str,
    experiment_id: str | None,
    n_traces_total: int,
    generated_at: str | None,
) -> AgentVersionAggregate:
    """Aggregate one arm's per-task L0 values across ``outcomes`` to version level."""
    n = len(outcomes)
    input_tokens = sum(_arm_metric(o, "input_tokens", candidate=candidate) for o in outcomes)
    output_tokens = sum(_arm_metric(o, "output_tokens", candidate=candidate) for o in outcomes)
    total_tokens = sum(_arm_metric(o, "total_tokens", candidate=candidate) for o in outcomes)
    cache_total = sum(_arm_metric(o, "cache_total_tokens", candidate=candidate) for o in outcomes)
    tool_calls = sum(_arm_metric(o, "total_tool_calls", candidate=candidate) for o in outcomes)
    redundant = sum(
        _reconstruct_redundant(
            _arm_metric(o, "redundancy_rate", candidate=candidate),
            _arm_metric(o, "total_tool_calls", candidate=candidate),
        )
        for o in outcomes
    )
    cost = sum(_arm_metric(o, "total_usd", candidate=candidate) for o in outcomes)
    # cost_priced is True only if EVERY counted comparison priced this arm. The
    # comparison notes flag an unpriced side; absent a positive cost we treat the
    # arm as unpriced rather than asserting a $0.00 truth.
    cost_priced = n > 0 and cost > 0.0
    return AgentVersionAggregate(
        agent_name=agent_name,
        agent_version=agent_version,
        experiment_id=experiment_id,
        n_traces=n,
        n_traces_total=n_traces_total,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cache_total_tokens=cache_total,
        tokens_per_trace=(total_tokens / n) if n else 0.0,
        total_tool_calls=tool_calls,
        redundant_tool_calls=redundant,
        redundancy_rate=(redundant / tool_calls) if tool_calls else 0.0,
        total_cost_usd=cost,
        cost_priced=cost_priced,
        generated_at=generated_at,
    )


def _metric_delta(
    metric: str, unit: str, baseline: float, candidate: float, *, lower_is_better: bool = True
) -> MetricDelta:
    """A :class:`MetricDelta` with the same semantics the comparison contract documents.

    ``delta_absolute = candidate - baseline``; ``delta_pct = 100 * delta / baseline``
    (``None`` when ``baseline == 0`` — undefined, not fabricated); ``improved`` is a
    **strict** move in the desired direction (a tie is not an improvement).
    """
    delta = candidate - baseline
    pct = (100.0 * delta / baseline) if baseline else None
    improved = (candidate < baseline) if lower_is_better else (candidate > baseline)
    return MetricDelta(
        metric=metric,
        unit=unit,
        lower_is_better=lower_is_better,
        baseline=baseline,
        candidate=candidate,
        delta_absolute=delta,
        delta_pct=round(pct, 4) if pct is not None else None,
        improved=improved,
    )


def _metric_worse(d: MetricDelta) -> bool:
    """Whether ``d`` is a **strict** move in the metric's *worse* direction.

    The exact mirror of :func:`_metric_delta`'s ``improved`` (negated direction),
    so it respects ``lower_is_better`` rather than hardcoding a direction: a
    lower-is-better metric is worse when it went *up*, a higher-is-better metric
    when it went *down*. A tie (no measurable change) is neither improved nor
    worse — so a genuine no-change never reads as a regression.
    """
    return (d.candidate > d.baseline) if d.lower_is_better else (d.candidate < d.baseline)


def _version_metric_value(agg: AgentVersionAggregate, metric: str) -> float:
    """Pull one comparison metric off a version aggregate."""
    return {
        "total_tokens": agg.total_tokens,
        "tokens_per_trace": agg.tokens_per_trace,
        "total_tool_calls": agg.total_tool_calls,
        "redundancy_rate": agg.redundancy_rate,
        "total_usd": agg.total_cost_usd,
    }[metric]


def _decide_status(
    *,
    n_promote: int,
    correctness_held: bool,
    headline_improved: bool,
    headline_worse: bool,
    any_regressed: bool,
    can_prove_improvement: bool,
) -> VersionComparisonStatus:
    """Trust-gate the comparison into a display status (the wall, in one place).

    Never green unless the controlled proof holds **and** the readiness wall has
    cleared. A measured-but-not-organically-ready improvement is amber
    (``CONTROLLED_PROOF_COLLECTING``), not green. A regression is **never** a win:
    both a correctness regression (``any_regressed``) *and* an actively-worse
    objective metric (``headline_worse`` — e.g. tokens went up) surface as
    ``REGRESSED``, before the no-win ``COLLECTING`` fallthrough. Only a genuine
    tie / no-change (neither improved nor worse, improvement unprovable) stays
    ``COLLECTING`` — that is honest: no regression, just no win yet.
    """
    if any_regressed or headline_worse:
        return VersionComparisonStatus.REGRESSED
    if not headline_improved:
        return VersionComparisonStatus.COLLECTING
    if n_promote == 0 or not correctness_held:
        return VersionComparisonStatus.COLLECTING
    if can_prove_improvement:
        return VersionComparisonStatus.PROVEN
    return VersionComparisonStatus.CONTROLLED_PROOF_COLLECTING


def build_phase2_version_bundle(
    artifact: Phase2Artifact,
    *,
    agent_name: str,
    baseline_version: str,
    candidate_version: str,
    experiment_id: str | None = None,
    thresholds: ReadinessThresholds | None = None,
    generated_at: str | None = None,
) -> VersionPublishBundle:
    """Build the unified per-version records from a Phase-2 comparison artifact.

    The baseline arm becomes version ``baseline_version`` and the candidate arm
    version ``candidate_version``. L0 is aggregated over the PROMOTE
    (objective-met + correctness-held) tasks — matching the artifact's
    ``realized_*`` headline — and readiness is computed from the real facts
    (organic trace count, frozen-suite presence) via
    :func:`ail.readiness.compute_readiness`.
    """
    stamp = generated_at or artifact.generated_at or datetime.now(UTC).isoformat()
    promote = _promote_outcomes(artifact)
    n_total = artifact.n_tasks

    baseline_agg = _aggregate_version(
        promote,
        candidate=False,
        agent_name=agent_name,
        agent_version=baseline_version,
        experiment_id=experiment_id,
        n_traces_total=n_total,
        generated_at=stamp,
    )
    candidate_agg = _aggregate_version(
        promote,
        candidate=True,
        agent_name=agent_name,
        agent_version=candidate_version,
        experiment_id=experiment_id,
        n_traces_total=n_total,
        generated_at=stamp,
    )

    deltas = [
        _metric_delta(
            metric,
            unit,
            _version_metric_value(baseline_agg, metric),
            _version_metric_value(candidate_agg, metric),
        )
        for metric, unit in _COMPARISON_METRICS
    ]
    objective_metric = artifact.objective_metric or "total_tokens"
    headline = next(d for d in deltas if d.metric == objective_metric)

    # Real readiness: organic trace count + frozen-suite presence. The Phase-2 run
    # used the FROZEN Task Suite (a non-empty content hash), so that universal gate
    # passes; the trace-count gates reflect the actual organic volume (collecting
    # until the floor is reached). This is what keeps a tiny seed from reading as
    # "ready to prove" — the wall, honestly applied.
    goal = _DeterministicGoal(objective_metric=objective_metric)
    facts = ReadinessFacts(
        trace_count=n_total,
        frozen_suite_present=bool(artifact.suite_content_hash),
    )
    cohort = Cohort.from_tags(
        candidate_version,
        {TAG_AGENT: agent_name, TAG_AGENT_VERSION: candidate_version},
    )
    readiness = compute_readiness(cohort, goal, facts, thresholds=thresholds, generated_at=stamp)

    any_regressed = any(o.l1_outcome is L1Outcome.REGRESSED for o in artifact.outcomes)
    correctness_held = artifact.n_promote > 0 and not any_regressed
    status = _decide_status(
        n_promote=artifact.n_promote,
        correctness_held=correctness_held,
        headline_improved=headline.improved,
        headline_worse=_metric_worse(headline),
        any_regressed=any_regressed,
        can_prove_improvement=readiness.can_prove_improvement,
    )

    notes = [
        "L0 aggregated over PROMOTE (correctness-held) tasks only; blocked tasks "
        "never counted as a win.",
        "Delta is a measured controlled-comparison result on the FROZEN suite "
        "(L1-gated, NO_LLM_JUDGE). Trust treatment is gated by the readiness wall.",
    ]
    if status is VersionComparisonStatus.CONTROLLED_PROOF_COLLECTING:
        notes.append(
            "Controlled proof holds; organic readiness still collecting "
            f"(tier={readiness.tier.value}) — not shown as a cleared/green improvement."
        )

    comparison = VersionComparison(
        agent_name=agent_name,
        baseline_version=baseline_version,
        candidate_version=candidate_version,
        objective_metric=objective_metric,
        status=status,
        n_promote=artifact.n_promote,
        n_block=artifact.n_block,
        n_errored=artifact.n_errored,
        correctness_held=correctness_held,
        frozen_suite_present=facts.frozen_suite_present,
        headline_metric=objective_metric,
        headline_baseline=headline.baseline,
        headline_candidate=headline.candidate,
        headline_delta_absolute=headline.delta_absolute,
        headline_delta_pct=headline.delta_pct,
        headline_improved=headline.improved,
        readiness=readiness,
        deltas=deltas,
        generated_at=stamp,
        notes=notes,
    )
    return VersionPublishBundle(aggregates=[baseline_agg, candidate_agg], comparison=comparison)


# ---------------------------------------------------------------------------
# Flat rows (column orders declared once; reused by DDL + INSERTs, as in publish.py)
# ---------------------------------------------------------------------------

REGISTRY_COLUMNS: list[str] = [
    "agent_name",
    "experiment_id",
    "description",
    "judge_config_json",
    "tag_filter_json",
    "goal_config_json",
    "annotations_table",
    "target_workspace",
    "generated_at",
]

VERSION_L0_COLUMNS: list[str] = [
    "agent_name",
    "agent_version",
    "experiment_id",
    "source",
    "basis",
    "n_traces",
    "n_traces_total",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cache_total_tokens",
    "tokens_per_trace",
    "total_tool_calls",
    "redundant_tool_calls",
    "redundancy_rate",
    "total_cost_usd",
    "cost_priced",
    "generated_at",
]

VERSION_COMPARISON_COLUMNS: list[str] = [
    "agent_name",
    "baseline_version",
    "candidate_version",
    "metric_tier",
    "metric",
    "unit",
    "lower_is_better",
    "baseline_value",
    "candidate_value",
    "delta_absolute",
    "delta_pct",
    "improved",
    "generated_at",
]

VERSION_READINESS_COLUMNS: list[str] = [
    "agent_name",
    "baseline_version",
    "candidate_version",
    "objective_metric",
    "status",
    "readiness_tier",
    "can_prove_improvement",
    "trace_count",
    "frozen_suite_present",
    "n_promote",
    "n_block",
    "n_errored",
    "correctness_held",
    "proof_source",
    "headline_metric",
    "headline_baseline",
    "headline_candidate",
    "headline_delta_pct",
    "headline_improved",
    "reasons",
    "generated_at",
]


def _registry_row(agent: Agent, *, generated_at: str | None) -> list[Any]:
    return [
        agent.agent_name,
        agent.experiment_id,
        agent.description,
        json.dumps(agent.judge_config) if agent.judge_config is not None else None,
        json.dumps(agent.tag_filter) if agent.tag_filter is not None else None,
        json.dumps(agent.goal_config) if agent.goal_config is not None else None,
        agent.annotations_table,
        agent.target_workspace,
        generated_at,
    ]


def _version_l0_row(agg: AgentVersionAggregate) -> list[Any]:
    return [
        agg.agent_name,
        agg.agent_version,
        agg.experiment_id,
        agg.source,
        agg.basis,
        agg.n_traces,
        agg.n_traces_total,
        agg.input_tokens,
        agg.output_tokens,
        agg.total_tokens,
        agg.cache_total_tokens,
        agg.tokens_per_trace,
        agg.total_tool_calls,
        agg.redundant_tool_calls,
        agg.redundancy_rate,
        agg.total_cost_usd,
        agg.cost_priced,
        agg.generated_at,
    ]


def _comparison_rows(comparison: VersionComparison) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for d in comparison.deltas:
        rows.append(
            [
                comparison.agent_name,
                comparison.baseline_version,
                comparison.candidate_version,
                "L0",
                d.metric,
                d.unit,
                d.lower_is_better,
                d.baseline,
                d.candidate,
                d.delta_absolute,
                d.delta_pct,
                d.improved,
                comparison.generated_at,
            ]
        )
    return rows


def _readiness_row(comparison: VersionComparison) -> list[Any]:
    r = comparison.readiness
    return [
        comparison.agent_name,
        comparison.baseline_version,
        comparison.candidate_version,
        comparison.objective_metric,
        comparison.status.value,
        r.tier.value,
        r.can_prove_improvement,
        r.trace_count,
        comparison.frozen_suite_present,
        comparison.n_promote,
        comparison.n_block,
        comparison.n_errored,
        comparison.correctness_held,
        comparison.proof_source,
        comparison.headline_metric,
        comparison.headline_baseline,
        comparison.headline_candidate,
        comparison.headline_delta_pct,
        comparison.headline_improved,
        " | ".join(r.reasons),
        comparison.generated_at,
    ]


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------


def _ddl(catalog: str, schema: str) -> list[str]:
    fqn = f"`{catalog}`.`{schema}`"
    return [
        f"CREATE SCHEMA IF NOT EXISTS {fqn} "
        "COMMENT 'Agent self-optimization loop: L0 deterministic metrics (Tier A).'",
        # MIGRATION NOTE (existing deployments): this is CREATE TABLE IF NOT EXISTS, so
        # the goal_config_json / annotations_table / target_workspace columns are NOT
        # added to a pre-existing agent_registry by this CREATE. The bootstrap's additive
        # reconcile (ail.jobs.bootstrap_tables.reconcile_app_table_columns) diffs this
        # declared schema against the live table and emits the ALTER ... ADD COLUMNS
        # before anything reads them:
        #     ALTER TABLE <catalog>.<schema>.agent_registry
        #         ADD COLUMNS (goal_config_json STRING, annotations_table STRING,
        #                      target_workspace STRING);
        f"""CREATE TABLE IF NOT EXISTS {fqn}.{REGISTRY_TABLE} (
            agent_name STRING,
            experiment_id STRING,
            description STRING,
            judge_config_json STRING,
            tag_filter_json STRING,
            goal_config_json STRING,
            annotations_table STRING,
            target_workspace STRING,
            generated_at STRING
        ) USING DELTA
        COMMENT 'Agent registry: agent_name -> dedicated MLflow experiment (+ optional config).'""",
        f"""CREATE TABLE IF NOT EXISTS {fqn}.{VERSION_L0_TABLE} (
            agent_name STRING,
            agent_version STRING,
            experiment_id STRING,
            source STRING,
            basis STRING,
            n_traces INT,
            n_traces_total INT,
            input_tokens DOUBLE,
            output_tokens DOUBLE,
            total_tokens DOUBLE,
            cache_total_tokens DOUBLE,
            tokens_per_trace DOUBLE,
            total_tool_calls DOUBLE,
            redundant_tool_calls DOUBLE,
            redundancy_rate DOUBLE,
            total_cost_usd DOUBLE,
            cost_priced BOOLEAN,
            generated_at STRING
        ) USING DELTA
        COMMENT 'Per (agent, version) L0 aggregate over counted PROMOTE traces; cost ESTIMATE.'""",
        f"""CREATE TABLE IF NOT EXISTS {fqn}.{VERSION_COMPARISON_TABLE} (
            agent_name STRING,
            baseline_version STRING,
            candidate_version STRING,
            metric_tier STRING,
            metric STRING,
            unit STRING,
            lower_is_better BOOLEAN,
            baseline_value DOUBLE,
            candidate_value DOUBLE,
            delta_absolute DOUBLE,
            delta_pct DOUBLE,
            improved BOOLEAN,
            generated_at STRING
        ) USING DELTA
        COMMENT 'Per (agent, baseline_version, candidate_version, metric) version delta.'""",
        f"""CREATE TABLE IF NOT EXISTS {fqn}.{VERSION_READINESS_TABLE} (
            agent_name STRING,
            baseline_version STRING,
            candidate_version STRING,
            objective_metric STRING,
            status STRING,
            readiness_tier STRING,
            can_prove_improvement BOOLEAN,
            trace_count INT,
            frozen_suite_present BOOLEAN,
            n_promote INT,
            n_block INT,
            n_errored INT,
            correctness_held BOOLEAN,
            proof_source STRING,
            headline_metric STRING,
            headline_baseline DOUBLE,
            headline_candidate DOUBLE,
            headline_delta_pct DOUBLE,
            headline_improved BOOLEAN,
            reasons STRING,
            generated_at STRING
        ) USING DELTA
        COMMENT 'Per comparison: readiness tier + trust-gated display status + proof header.'""",
    ]


# ---------------------------------------------------------------------------
# Publish orchestration
# ---------------------------------------------------------------------------


def publish_registry(
    registry: AgentRegistry,
    *,
    client: Any,
    warehouse_id: str,
    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
    generated_at: str | None = None,
) -> int:
    """Write each registered agent to ``agent_registry`` (idempotent per agent).

    Returns the number of agents written. Each agent's row is swapped in by a
    composite ``agent_name`` predicate, so re-publishing one agent never disturbs
    another.
    """
    stamp = generated_at or datetime.now(UTC).isoformat()
    fqn = f"`{catalog}`.`{schema}`"
    for ddl in _ddl(catalog, schema):
        _execute(client, warehouse_id, ddl)
    for agent in registry.agents:
        _atomic_replace_table(
            client,
            warehouse_id,
            fqn,
            REGISTRY_TABLE,
            REGISTRY_COLUMNS,
            [_registry_row(agent, generated_at=stamp)],
            f"agent_name = {_lit(agent.agent_name)}",
        )
    return len(registry.agents)


# ---------------------------------------------------------------------------
# Registry read-back — the SINGLE source of truth, as fully-typed Agent objects
# ---------------------------------------------------------------------------


class _RegistryTableMissing(RuntimeError):
    """The ``agent_registry`` table does not exist yet (fresh workspace)."""


def load_registered_agents_full(
    *,
    client: Any,
    warehouse_id: str,
    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
) -> list[Agent]:
    """Read every registered agent back from ``agent_registry`` as a typed :class:`Agent`.

    The symmetric read of :func:`publish_registry`: it ``SELECT *``s the registry
    table and reconstructs fully-typed :class:`~ail.registry.Agent` objects carrying
    every field a per-agent job or the local companion needs — the goal config, the
    annotations table, the target workspace, the judge config, and the tag filter.
    Both the job entrypoints and the companion read the single source of truth
    through this one helper (it lives here, not in the heavier onboarding service, so
    either side can import it without the wizard's dependency chain).

    Robust to a real, evolving deployment, and it never fabricates a value:

    * A **not-yet-created** table (fresh workspace, no publish yet) is an empty
      registry (``[]``), not an error. Any *other* read failure (permission,
      warehouse down) propagates so a caller fails closed rather than treating
      "cannot read" as "no agents".
    * An **old row** written before the ``goal_config_json`` / ``annotations_table``
      / ``target_workspace`` columns existed — or a migrated table whose old rows
      carry ``NULL`` there — reconstructs cleanly: the absent value is ``None`` on
      the ``Agent``, never a guessed default. ``SELECT *`` returns only the columns
      that physically exist, so a pre-migration table simply yields no such keys and
      ``dict.get`` returns ``None``.
    * A row missing the primary ``agent_name`` / ``experiment_id`` cannot form a
      valid ``Agent`` and is skipped rather than fabricated.
    """
    fqn = f"`{catalog}`.`{schema}`.{REGISTRY_TABLE}"
    try:
        rows = _query_registry_rows(client, warehouse_id, f"SELECT * FROM {fqn}")
    except _RegistryTableMissing:
        return []
    agents: list[Agent] = []
    for row in rows:
        name = row.get("agent_name")
        exp = row.get("experiment_id")
        if not name or not exp:
            continue  # unreconstructable row — never fabricate a primary key
        description = row.get("description")
        agents.append(
            Agent(
                agent_name=str(name),
                experiment_id=str(exp),
                description="" if description is None else str(description),
                judge_config=_json_or_none(row.get("judge_config_json")),
                goal_config=_json_or_none(row.get("goal_config_json")),
                annotations_table=_str_or_none(row.get("annotations_table")),
                tag_filter=_json_or_none(row.get("tag_filter_json")),
                target_workspace=_str_or_none(row.get("target_workspace")),
            )
        )
    return agents


def _str_or_none(value: Any) -> str | None:
    """A live cell as a plain string, or ``None`` when absent — never fabricated."""
    return None if value is None else str(value)


def _json_or_none(value: Any) -> Any:
    """Parse a JSON string cell back to its structure; ``None``/empty stays ``None``.

    The symmetric inverse of the ``json.dumps(...)`` write in :func:`_registry_row`.
    A ``NULL`` / absent / empty cell is "not configured" (``None``); a genuinely
    malformed non-empty JSON value raises loud rather than fabricating a structure.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return json.loads(text)


def _query_registry_rows(client: Any, warehouse_id: str, statement: str) -> list[dict[str, Any]]:
    """Run a SELECT and return rows as ``{column: value}`` dicts.

    Mirrors the statement-execution wait loop used across the framework
    (:func:`ail.publish._execute`), but reads the result set. A "table/view not
    found" failure is raised as :class:`_RegistryTableMissing` (a fresh workspace,
    not a real error); any other non-success is a hard :class:`RuntimeError`.
    """
    import time

    from databricks.sdk.service.sql import StatementState

    resp = client.statement_execution.execute_statement(
        warehouse_id=warehouse_id, statement=statement, wait_timeout="50s"
    )
    statement_id = resp.statement_id
    state = resp.status.state if resp.status else None
    while state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(1.0)
        resp = client.statement_execution.get_statement(statement_id)
        state = resp.status.state if resp.status else None
    if state != StatementState.SUCCEEDED:
        detail = ""
        if resp.status and resp.status.error:
            detail = resp.status.error.message or ""
        low = detail.lower()
        if "table_or_view_not_found" in low or "does not exist" in low or "cannot be found" in low:
            raise _RegistryTableMissing(detail)
        raise RuntimeError(f"statement {state}: {detail}\nSQL head: {statement[:200]}")

    manifest = resp.manifest
    columns = [c.name for c in manifest.schema.columns] if manifest and manifest.schema else []
    data = resp.result.data_array if resp.result and resp.result.data_array else []
    return [dict(zip(columns, row, strict=False)) for row in data]


def publish_version_bundle(
    bundle: VersionPublishBundle,
    *,
    client: Any,
    warehouse_id: str,
    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
) -> None:
    """Write a :class:`VersionPublishBundle` to the unified per-version tables.

    Each slice is swapped atomically by a composite predicate: the L0 rows per
    (agent, version), the comparison + readiness rows per (agent, baseline,
    candidate). A re-publish replaces exactly that slice and nothing else.
    """
    fqn = f"`{catalog}`.`{schema}`"
    for ddl in _ddl(catalog, schema):
        _execute(client, warehouse_id, ddl)

    cmp = bundle.comparison
    for agg in bundle.aggregates:
        _atomic_replace_table(
            client,
            warehouse_id,
            fqn,
            VERSION_L0_TABLE,
            VERSION_L0_COLUMNS,
            [_version_l0_row(agg)],
            f"agent_name = {_lit(agg.agent_name)} AND agent_version = {_lit(agg.agent_version)}",
        )

    comparison_predicate = (
        f"agent_name = {_lit(cmp.agent_name)} "
        f"AND baseline_version = {_lit(cmp.baseline_version)} "
        f"AND candidate_version = {_lit(cmp.candidate_version)}"
    )
    _atomic_replace_table(
        client,
        warehouse_id,
        fqn,
        VERSION_COMPARISON_TABLE,
        VERSION_COMPARISON_COLUMNS,
        _comparison_rows(cmp),
        comparison_predicate,
    )
    _atomic_replace_table(
        client,
        warehouse_id,
        fqn,
        VERSION_READINESS_TABLE,
        VERSION_READINESS_COLUMNS,
        [_readiness_row(cmp)],
        comparison_predicate,
    )


def publish_phase2_seed(
    *,
    artifact_path: str | Path,
    agent_name: str,
    baseline_version: str,
    candidate_version: str,
    warehouse_id: str,
    profile: str | None = None,
    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
    registry_path: str | Path | None = None,
) -> VersionPublishBundle:
    """Seed the unified tables from a committed Phase-2 artifact (no live MLflow).

    Reads the committed :class:`~ail.optimize.phase2.Phase2Artifact` JSON, looks the
    agent's experiment up in the registry (for provenance), builds the per-version
    bundle, and writes the registry + the unified per-version tables. This is the
    committed-artifact fallback that lets the comparison view render real data even
    when live trace auth is unavailable. Returns the published bundle.
    """
    artifact = Phase2Artifact.model_validate_json(Path(artifact_path).read_text(encoding="utf-8"))
    registry = load_registry(registry_path)
    experiment_id: str | None = None
    try:
        experiment_id = registry.get(agent_name).experiment_id
    except KeyError:
        experiment_id = None

    bundle = build_phase2_version_bundle(
        artifact,
        agent_name=agent_name,
        baseline_version=baseline_version,
        candidate_version=candidate_version,
        experiment_id=experiment_id,
    )

    client = _build_workspace_client(profile)
    publish_registry(
        registry, client=client, warehouse_id=warehouse_id, catalog=catalog, schema=schema
    )
    publish_version_bundle(
        bundle, client=client, warehouse_id=warehouse_id, catalog=catalog, schema=schema
    )

    cmp = bundle.comparison
    print(
        f"published agent={agent_name} {baseline_version} -> {candidate_version}: "
        f"{cmp.headline_metric} {cmp.headline_baseline:,.0f} -> {cmp.headline_candidate:,.0f} "
        f"({cmp.headline_delta_pct:+.2f}% ); status={cmp.status.value} "
        f"readiness={cmp.readiness.tier.value} promote={cmp.n_promote} block={cmp.n_block}"
    )
    return bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Publish the unified per-agent-version comparison tables (Tier A)."
    )
    parser.add_argument(
        "--artifact",
        default="artifacts/phase2_token_lever.json",
        help="Committed Phase-2 comparison artifact to seed from.",
    )
    parser.add_argument("--agent", default="claude_code")
    parser.add_argument("--baseline-version", default="v0-baseline-no-skill")
    parser.add_argument("--candidate-version", default="v1-token-efficiency-skill")
    parser.add_argument(
        "--registry",
        default="config/agents.yaml",
        help="Agent registry YAML (default: config/agents.yaml).",
    )
    parser.add_argument(
        "--warehouse-id",
        default=os.environ.get("AIL_WAREHOUSE_ID"),
        help="SQL warehouse id used to create and populate the Delta tables.",
    )
    parser.add_argument(
        "--profile",
        default=os.environ.get("AIL_DATABRICKS_PROFILE", "dais-demo"),
        help="Databricks CLI profile (ignored if DATABRICKS_HOST/DATABRICKS_TOKEN are set).",
    )
    parser.add_argument("--catalog", default=DEFAULT_CATALOG)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    args = parser.parse_args(argv)

    if not args.warehouse_id:
        parser.error("--warehouse-id is required (or set AIL_WAREHOUSE_ID)")

    registry_path = args.registry if Path(args.registry).exists() else None
    publish_phase2_seed(
        artifact_path=args.artifact,
        agent_name=args.agent,
        baseline_version=args.baseline_version,
        candidate_version=args.candidate_version,
        warehouse_id=args.warehouse_id,
        profile=args.profile,
        catalog=args.catalog,
        schema=args.schema,
        registry_path=registry_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
