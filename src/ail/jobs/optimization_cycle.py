"""``ail-optimization-cycle`` — the unified, scheduled optimization cycle.

This is the single scheduled runner that unifies **L3/RLM review** and the
**layered A+B loop controller** onto **one cadence over one trace set**
(``docs/LOOP_CONTROLLER.md`` lane 2). On each firing it:

1. **reviews** the cycle's sampled/recent trace set with the *existing*
   :mod:`ail.l3.continuous` RLM reviewer — reusing its sampling knobs
   (``max-results`` / ``sample-rate`` / ``min-tokens``), its idempotency (skip
   traces already carrying ``rlm_*``), and its fail-closed failed-marker. This
   is the same reviewer the retired standalone job ran; here it runs *in-cycle*
   so review and planning always see the same window. A review failure is
   recorded and **never blocks the cycle** (per-trace faults are isolated by the
   reviewer itself; a total review failure is caught here);
2. **plans** over the *now-fresh* feedback: the deterministic **Lane A** rules
   (:func:`ail.loop.decision_rules.decide`) **and** the LLM-agent **Lane B**
   planner (:func:`ail.loop.planner.agent_planner`), de-duped into one union
   (:func:`ail.loop.planner.combined_decisions`);
3. **proves + gates + proposes** by driving that union through the unchanged
   controller pipeline (:func:`ail.loop.planner.run_cycle_with_planner` →
   :func:`ail.loop.controller.run_cycle`) with the **real** prover
   (:func:`ail.optimize.phase2.run_phase2_comparison`) and gate
   (:func:`ail.readiness.compute_readiness`); and
4. **publishes** the resulting PENDING proposals to the unified
   ``agent_proposed_actions`` table (:func:`ail.loop.publish_proposals.publish_agent_proposals`),
   atomically replacing this agent's slice — so a superseded pending proposal
   disappears and the write is idempotent.

**Nothing here applies a change.** The controller emits only PENDING proposals;
a human approves the live apply in the app (lane 3). This module wires seams and
publishes; it registers no version, sets no alias, and runs no ``CREATE``.

**Injectable seams = testable.** :func:`run_optimization_cycle` takes the RLM
step, feedback source, candidate builder, prover, gate, planner, and publish
function as parameters (the controller's own philosophy), so a whole cycle runs
in tests with fakes — no live MLflow / agent / warehouse. :func:`main` wires the
**real** defaults.

Auth mirrors :mod:`ail.jobs.publish_job` / :mod:`ail.jobs.continuous_rlm`
(:func:`resolve_job_auth`); no hardcoded host.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ail.goals.compiler import CompiledGoal, GoalTarget, Guardrail
from ail.jobs.publish_job import resolve_job_auth
from ail.l3.cohort_review import aggregate_assets
from ail.l3.continuous import ContinuousRlmRunReport, run_continuous_rlm
from ail.l3.contract import HaloReviewVerdict, RankedAsset
from ail.l3.reviewer import OVERALL_FEEDBACK_NAME
from ail.loop.candidate_builders import token_efficiency_candidate_builder
from ail.loop.controller import Candidate, CycleResult
from ail.loop.decision_rules import (
    DecisionThresholds,
    FeedbackBundle,
    RedundantReadSignal,
    RlmAssetSignal,
)
from ail.loop.planner import (
    CombinedDecisions,
    Planner,
    agent_planner,
    run_cycle_with_planner,
)
from ail.loop.proposals import ProposalStatus, ProposedAction
from ail.loop.publish_proposals import PROPOSALS_TABLE, publish_agent_proposals
from ail.metrics.contract import L0MetricsReport
from ail.publish import DEFAULT_CATALOG, DEFAULT_SCHEMA
from ail.readiness import ReadinessStatus, compute_readiness
from ail.registry import Agent

__all__ = [
    "OptimizationCycleReport",
    "verdict_from_trace",
    "ranked_assets_from_traces",
    "rlm_asset_signals",
    "redundant_reads_from_l0",
    "build_feedback_bundle",
    "run_optimization_cycle",
    "main",
]


@dataclass(slots=True)
class OptimizationCycleReport:
    """Summary of one unified optimization-cycle run (for logging / auditing)."""

    agent_name: str
    rlm: ContinuousRlmRunReport | None
    rlm_error: str | None
    cycle: CycleResult
    plan: CombinedDecisions
    n_published: int


# ---------------------------------------------------------------------------
# Feedback assembly (pure, testable): read the now-fresh cohort feedback.
# ---------------------------------------------------------------------------


def verdict_from_trace(trace: Any) -> HaloReviewVerdict | None:
    """Reconstruct the attached HALO verdict from a trace's ``rlm_review`` feedback.

    The in-cycle RLM reviewer writes the full verdict as ``verdict_json`` in the
    metadata of the overall ``rlm_review`` assessment (see
    :func:`ail.l3.reviewer._overall_metadata`). This reads it back so the feedback
    source consumes the *now-fresh* review without re-running HALO. Shape-tolerant
    (mirrors :func:`ail.l3.continuous.has_rlm_assessment`): a trace with no parseable
    ``rlm_review`` verdict returns ``None`` (it simply contributes no assets).
    """
    info = getattr(getattr(trace, "raw", None), "info", None)
    assessments = getattr(info, "assessments", None) if info is not None else None
    for assessment in list(assessments or []):
        if str(getattr(assessment, "name", "") or "") != OVERALL_FEEDBACK_NAME:
            continue
        metadata = getattr(assessment, "metadata", None)
        if not isinstance(metadata, dict):
            continue
        verdict_json = metadata.get("verdict_json")
        if not verdict_json:
            continue
        try:
            return HaloReviewVerdict.model_validate_json(str(verdict_json))
        except Exception:  # noqa: BLE001 - a malformed verdict is skipped, never fabricated
            return None
    return None


def ranked_assets_from_traces(traces: Sequence[Any]) -> list[RankedAsset]:
    """Recurrence-rank the RLM-recommended assets attached across ``traces``.

    Reuses :func:`ail.l3.cohort_review.aggregate_assets` (no re-implementation of
    the roll-up) over the verdicts reconstructed from each trace's attached
    ``rlm_review`` feedback. Traces without a parseable verdict contribute nothing.
    """
    verdicts = [v for t in traces if (v := verdict_from_trace(t)) is not None]
    return aggregate_assets(verdicts)


def rlm_asset_signals(ranked: Sequence[RankedAsset]) -> tuple[RlmAssetSignal, ...]:
    """Map recurrence-ranked assets onto the decision-rule signal type."""
    return tuple(
        RlmAssetSignal(
            asset_type=a.asset_type,
            title=a.title,
            n_traces=a.n_traces,
            rank=a.rank,
            trace_ids=tuple(a.trace_ids),
        )
        for a in ranked
    )


def redundant_reads_from_l0(
    report: L0MetricsReport,
    *,
    min_occurrences: int = 2,
    dominant_top_n: int = 1,
) -> tuple[RedundantReadSignal, ...]:
    """Map the L0 repeated-call diagnosis onto the redundant-read signal type.

    Reuses the deterministic L0 redundancy diagnosis
    (:attr:`ail.metrics.contract.ToolRedundancy.repeated_calls`) — the byte-exact /
    path / prologue repeats the un-gameable L0 metric already surfaces. The
    ``dominant_top_n`` most-repeated identities are flagged ``dominant`` (the
    redundant-read decision rule fires only on a dominant, recurring pattern); the
    rest travel as context. Identities repeated fewer than ``min_occurrences`` times
    are dropped (not a pattern worth a skill update).
    """
    repeats = sorted(
        (r for r in report.aggregate.redundancy.repeated_calls if r.count >= min_occurrences),
        key=lambda r: (-r.count, r.tool, r.identity),
    )
    return tuple(
        RedundantReadSignal(
            tool=r.tool,
            repeated_target=r.identity,
            occurrences=r.count,
            dominant=i < dominant_top_n,
        )
        for i, r in enumerate(repeats)
    )


def build_feedback_bundle(
    traces: Sequence[Any],
    *,
    objective_metric: str,
    objective_baseline: float | None = None,
    min_redundant_occurrences: int = 2,
    dominant_top_n: int = 1,
) -> FeedbackBundle:
    """Assemble a :class:`FeedbackBundle` from the fresh cohort traces (real feedback).

    Composed entirely from existing, un-gameable sources: the objective value and
    redundant-read patterns from the deterministic L0 metrics
    (:func:`ail.metrics.l0_deterministic.compute_l0`), and the RLM-recommended
    assets from the in-cycle review just attached to these traces
    (:func:`ranked_assets_from_traces`). ``objective_metric_value`` is wired for the
    deterministic headline ``total_tokens`` (the frozen-suite objective); other
    objectives are left ``None`` (conservatively treated as "not yet met") rather
    than guessing a value. ``judge_dimensions`` and ``post_apply_regressions`` are
    left empty here — they require the judge-agreement cadence and post-apply
    version comparison, which are owned by other lanes; leaving them empty is
    fail-closed (no fabricated judge / regression signal), and Lane A/B act on the
    signals that are present.
    """
    from ail.metrics.l0_deterministic import compute_l0

    report = compute_l0(list(traces))
    objective_value: float | None = None
    if objective_metric == "total_tokens":
        objective_value = float(report.aggregate.tokens.total_tokens)

    return FeedbackBundle(
        objective_metric_value=objective_value,
        objective_baseline_value=objective_baseline,
        rlm_assets=rlm_asset_signals(ranked_assets_from_traces(traces)),
        redundant_reads=redundant_reads_from_l0(
            report,
            min_occurrences=min_redundant_occurrences,
            dominant_top_n=dominant_top_n,
        ),
    )


# ---------------------------------------------------------------------------
# Seam types + the injectable orchestration.
# ---------------------------------------------------------------------------

#: The in-cycle RLM step: reviews the sampled set and returns the run report.
RlmStep = Callable[[], ContinuousRlmRunReport]
#: Publishes the cycle's PENDING proposals; returns the number of rows written.
PublishFn = Callable[[list[ProposedAction]], int]
#: The controller seams (re-exported for the signature below).
FeedbackSource = Callable[[], FeedbackBundle]
CandidateBuilder = Callable[..., Candidate | None]
Prover = Callable[..., Any]
Gate = Callable[..., ReadinessStatus]


def run_optimization_cycle(
    agent: Agent,
    goal: CompiledGoal,
    *,
    rlm_step: RlmStep,
    feedback_source: FeedbackSource,
    candidate_builder: CandidateBuilder,
    prover: Prover,
    gate: Gate,
    planner: Planner = agent_planner,
    publish_fn: PublishFn,
    decision_thresholds: DecisionThresholds | None = None,
    now: str | None = None,
) -> OptimizationCycleReport:
    """Run one unified cycle: in-cycle RLM review → layered A+B propose → publish.

    The RLM step runs first (fail-closed and **non-blocking**: a total review
    failure is caught and recorded, the cycle still runs — per-trace review faults
    are already isolated inside the reviewer). The layered A+B controller then runs
    over the now-fresh feedback via :func:`ail.loop.planner.run_cycle_with_planner`
    (the unchanged prove → gate → propose pipeline), and the resulting PENDING
    proposals are published. Applies nothing.
    """
    rlm_report: ContinuousRlmRunReport | None = None
    rlm_error: str | None = None
    try:
        rlm_report = rlm_step()
    except Exception as exc:  # noqa: BLE001 - a review failure must NOT block the cycle
        rlm_error = f"{type(exc).__name__}: {exc}"

    pc = run_cycle_with_planner(
        agent,
        goal,
        feedback_source=feedback_source,
        candidate_builder=candidate_builder,
        prover=prover,
        gate=gate,
        planner=planner,
        decision_thresholds=decision_thresholds,
        now=now,
    )

    n_published = publish_fn(list(pc.result.proposals))

    return OptimizationCycleReport(
        agent_name=agent.agent_name,
        rlm=rlm_report,
        rlm_error=rlm_error,
        cycle=pc.result,
        plan=pc.plan,
        n_published=n_published,
    )


# ---------------------------------------------------------------------------
# Real default seams (production wiring; each reuses an existing module).
# ---------------------------------------------------------------------------


def _default_rlm_step(args: argparse.Namespace) -> RlmStep:
    """Real RLM step: the existing continuous reviewer over the sampled set."""

    def _step() -> ContinuousRlmRunReport:
        return run_continuous_rlm(
            args.experiment,
            judge_model=args.judge_model,
            sql_warehouse_id=args.warehouse_id,
            max_results=args.max_results,
            max_reviews=args.max_reviews,
            sample_rate=args.sample_rate,
            min_tokens=args.min_tokens,
            reviewer_experiment_id=args.reviewer_experiment or None,
            max_turns=args.max_turns,
            temperature=args.temperature,
        )

    return _step


def _default_feedback_source(agent: Agent, args: argparse.Namespace) -> FeedbackSource:
    """Real feedback source: read the fresh cohort feedback over the same window."""

    def _source() -> FeedbackBundle:
        from ail.ingest.mlflow_source import MLflowTraceSource

        src = MLflowTraceSource(profile=args.profile or None)
        traces = src.fetch_cohort_traces(
            agent.cohort(),
            experiment_id=args.experiment,
            max_results=args.max_results,
            order_by=["timestamp_ms DESC"],
        )
        return build_feedback_bundle(
            traces,
            objective_metric=args.objective_metric,
            objective_baseline=args.objective_baseline,
        )

    return _source


def _default_gate(args: argparse.Namespace) -> Gate:
    """Real gate: readiness for the agent's cohort (reuse of the preflight facts path)."""

    def _gate(*, goal: CompiledGoal, agent: Agent) -> ReadinessStatus:
        from ail.jobs.readiness_preflight import gather_facts

        cohort = agent.cohort()
        facts = gather_facts(
            args.experiment,
            cohort,
            profile=args.profile or None,
            warehouse_id=args.warehouse_id,
        )
        return compute_readiness(cohort, goal, facts)

    return _gate


def _fetch_pending_proposal_ids(agent: Agent, args: argparse.Namespace) -> frozenset[str] | None:
    """Read this agent's currently-PENDING proposal ids from ``agent_proposed_actions``.

    The cost guard's input (:func:`ail.loop.candidate_builders.token_efficiency_candidate_builder`):
    the builder skips (re-)building a candidate whose proposal id is already pending,
    so an expensive frozen-suite proof runs at most once per open proposal instead of
    every hourly firing. Reuses the same SELECT-side helpers lane 3 reads with
    (:func:`ail.loop.apply_service._query_rows`, :func:`ail.publish._build_workspace_client` /
    :func:`ail.publish._lit`) — no new read path.

    Returns ``None`` — the fail-closed "unavailable, do not spend" sentinel — on ANY
    read failure (auth, network, or a missing table on the very first run before
    bootstrap/publish creates it); the builder then proposes nothing this cycle rather
    than re-proving a known result blindly. An empty set (table present, no pending
    rows) is distinct from ``None`` and lets the first proposal be built.
    """
    from ail.loop.apply_service import _query_rows
    from ail.publish import _build_workspace_client, _lit

    fqn = f"`{args.catalog}`.`{args.schema}`.{PROPOSALS_TABLE}"
    sql = (
        f"SELECT proposal_id FROM {fqn} WHERE agent_name = {_lit(agent.agent_name)} "
        f"AND status = {_lit(ProposalStatus.PENDING.value)}"
    )
    try:
        client = _build_workspace_client(args.profile or None)
        rows = _query_rows(client, args.warehouse_id, sql)
    except Exception:  # noqa: BLE001 - fail-closed toward NOT spending on any read failure
        return None
    return frozenset(str(r["proposal_id"]) for r in rows if r.get("proposal_id"))


def _default_candidate_builder(agent: Agent, args: argparse.Namespace) -> CandidateBuilder:
    """Real candidate builder: the first provable candidate→prove path, cost-guarded.

    Delegates to :func:`ail.loop.candidate_builders.token_efficiency_candidate_builder`,
    which maps a ``SKILL_UPDATE`` decision for a token-reduction goal to a
    :class:`~ail.loop.controller.Candidate` carrying the *proven*
    :func:`ail.optimize.lever.token_efficiency_intervention` (flowed through
    :func:`_default_prover` unchanged), and returns ``None`` for every other action
    kind / goal — the controller's first-class fail-closed "no candidate ⇒ no
    proposal" outcome (additive ``metric_view`` has no frozen-suite intervention,
    ``gepa_prompt`` bodies come from the separate heavy GEPA run, ``instruction_update``
    / ``revert`` are unwired). Returning ``None`` keeps the cycle honest — it proposes
    nothing it cannot prove and fabricates no proof.

    Fetches this agent's pending proposals **once** here and closes over them so the
    expensive real-agent frozen-suite proof runs at most once per open proposal (and
    not at all if the pending check could not run — fail-closed toward not spending).
    """
    pending = _fetch_pending_proposal_ids(agent, args)
    return token_efficiency_candidate_builder(pending_proposal_ids=pending)


def _default_prover(args: argparse.Namespace) -> Prover:
    """Real prover: the frozen-suite comparison harness (unchanged).

    Delegates to :func:`ail.optimize.phase2.run_phase2_comparison` — the same
    WITH/WITHOUT, correctness-held proof the rest of the framework uses; the
    controller reads its :class:`~ail.optimize.phase2.Phase2Artifact` verbatim and
    fabricates nothing. It adapts the controller's :class:`~ail.loop.controller.Candidate`
    into the harness by reading the candidate's ``prover_input`` as the
    :class:`~ail.optimize.lever.SkillInjectionIntervention` to prove; the heavy agent
    adapter + frozen suite are constructed lazily on first use. A candidate that
    carries no runnable intervention raises :class:`NotImplementedError` — caught by
    :func:`ail.loop.controller.run_cycle` as a fail-closed skip *with* the reason,
    never a fabricated proof (an additive ``metric_view`` has no frozen-suite
    intervention; that proof path is upstream-incomplete, see ``docs/DEPLOY.md``).
    """

    def _prove(candidate: Candidate, *, goal: CompiledGoal, agent: Agent) -> Any:
        from ail.ingest.adapters.claude_code import ClaudeCodeAdapter
        from ail.optimize.lever import LeverConfig, SkillInjectionIntervention
        from ail.optimize.phase2 import run_phase2_comparison
        from ail.task_suite import load_task_suite

        intervention = candidate.prover_input
        if not isinstance(intervention, SkillInjectionIntervention):
            raise NotImplementedError(
                "the default frozen-suite prover proves a SkillInjectionIntervention supplied as "
                "the candidate's prover_input; this candidate carries "
                f"{type(intervention).__name__} (change kind {candidate.change.kind.value}), for "
                "which no frozen-suite proof path is wired yet — inject a prover for it."
            )
        suite = load_task_suite()
        adapter = ClaudeCodeAdapter(mlflow_experiment=agent.experiment_id)
        candidate_lever = LeverConfig(name="candidate", intervention=intervention)
        return run_phase2_comparison(
            suite=suite,
            adapter=adapter,
            candidate=candidate_lever,
            experiment=args.experiment,
            profile=args.profile or None,
            warehouse_id=args.warehouse_id,
        )

    return _prove


def _default_publish(agent: Agent, args: argparse.Namespace) -> PublishFn:
    """Real publish: atomically replace this agent's slice of the proposals table."""

    def _publish(proposals: list[ProposedAction]) -> int:
        from ail.publish import _build_workspace_client

        client = _build_workspace_client(args.profile or None)
        return publish_agent_proposals(
            proposals,
            agent_name=agent.agent_name,
            client=client,
            warehouse_id=args.warehouse_id,
            catalog=args.catalog,
            schema=args.schema,
        )

    return _publish


# ---------------------------------------------------------------------------
# Config resolution + CLI entrypoint.
# ---------------------------------------------------------------------------


def _opt_float(value: str) -> float | None:
    """Parse an optional float CLI arg; an empty string yields ``None``.

    The bundle wires this from a DAB variable whose default is empty (so the goal
    keeps its fail-closed "no baseline ⇒ objective treated as not-yet-met" default
    rather than a fabricated one). ``type=float`` would raise on the empty string a
    named-parameter substitution passes, so this maps blank ⇒ ``None`` and defers to
    ``float`` otherwise.
    """
    return float(value) if value.strip() else None


def _build_goal(args: argparse.Namespace) -> CompiledGoal:
    """Build the operator-configured goal; require an explicit confirmation flag.

    A scheduled job optimizes a goal the operator configured, so the deployment's
    ``--confirm-goal`` flag (or ``AIL_CONFIRM_GOAL=1``) *is* the human confirmation
    the controller requires. Without it the goal stays unconfirmed and
    :func:`ail.loop.controller.run_cycle` refuses to run (fail-loud) — a
    misconfigured job never optimizes an unreviewed goal.
    """
    guardrails: list[Guardrail] = []
    for spec in args.guardrail_judge or []:
        if not spec.strip():
            continue  # blank spec (e.g. the empty DAB var default) => no guardrail
        name, _, threshold = spec.partition(":")
        guardrails.append(
            Guardrail(
                name=name.strip(),
                kind="judge",
                threshold=float(threshold) if threshold.strip() else None,
            )
        )
    goal = CompiledGoal(
        objective_metric=args.objective_metric,
        direction=args.goal_direction,
        target=GoalTarget(value=args.goal_target, kind=args.goal_target_kind),
        guardrails=tuple(guardrails),
        cohort=args.agent,
    )
    confirmed = str(args.goal_confirmed).strip().lower() in {"1", "true", "yes"}
    return goal.confirm() if confirmed else goal


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one unified optimization cycle: in-cycle RLM review over the sampled "
            "trace set, then the layered A+B loop controller (propose-only), then "
            "publish PENDING proposals. Applies nothing."
        )
    )
    parser.add_argument("--agent", default="claude_code", help="Agent name (proposal scope).")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--warehouse-id", default=os.environ.get("AIL_WAREHOUSE_ID"))
    parser.add_argument("--profile", default=os.environ.get("DATABRICKS_CONFIG_PROFILE"))
    parser.add_argument("--catalog", default=DEFAULT_CATALOG)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    # in-cycle RLM sampling knobs (the existing ail.l3.continuous scheme; no new one)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--max-results", type=int, default=100)
    parser.add_argument("--max-reviews", type=int, default=2)
    parser.add_argument("--sample-rate", type=float, default=0.10)
    parser.add_argument("--min-tokens", type=int, default=50_000)
    parser.add_argument("--reviewer-experiment", default="")
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=None)
    # goal (operator-configured; confirmed by --confirm-goal)
    parser.add_argument("--objective-metric", default="total_tokens")
    parser.add_argument("--goal-direction", default="minimize", choices=["minimize", "maximize"])
    parser.add_argument("--goal-target", type=float, default=-0.30)
    parser.add_argument("--goal-target-kind", default="relative", choices=["relative", "absolute"])
    parser.add_argument(
        "--guardrail-judge",
        action="append",
        default=None,
        help="Judge guardrail as 'name:threshold' (repeatable).",
    )
    parser.add_argument(
        "--objective-baseline",
        type=_opt_float,
        default=None,
        help="Baseline value a relative objective target is measured against "
        "(empty => treated as not-yet-met, no fabricated baseline).",
    )
    parser.add_argument(
        "--goal-confirmed",
        default=os.environ.get("AIL_CONFIRM_GOAL", "false"),
        help="'true' to mark the operator-configured goal human-confirmed (required to run; "
        "a scheduled job's bundle-authored goal is confirmed by the operator who deploys it). "
        "Anything else leaves it unconfirmed and run_cycle refuses to run (fail-loud).",
    )
    parser.add_argument("--planner-model", default=None)
    parser.add_argument(
        "--token-secret-scope",
        default=os.environ.get("AIL_TOKEN_SECRET_SCOPE", ""),
    )
    parser.add_argument(
        "--token-secret-key",
        default=os.environ.get("AIL_TOKEN_SECRET_KEY", ""),
    )
    args = parser.parse_args(argv)
    if not args.warehouse_id:
        parser.error("--warehouse-id is required (or set AIL_WAREHOUSE_ID)")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    auth_path = resolve_job_auth(
        token_secret_scope=args.token_secret_scope or None,
        token_secret_key=args.token_secret_key or None,
    )
    agent = Agent(agent_name=args.agent, experiment_id=args.experiment)
    goal = _build_goal(args)

    planner: Planner
    if args.planner_model:
        from functools import partial

        planner = partial(agent_planner, model=args.planner_model)
    else:
        planner = agent_planner

    print(
        f"[ail.jobs.optimization_cycle] auth={auth_path} "
        f"host={os.environ.get('DATABRICKS_HOST')} agent={agent.agent_name} "
        f"experiment={args.experiment} objective={goal.objective_metric} "
        f"sample_rate={args.sample_rate} max_reviews={args.max_reviews}"
    )

    report = run_optimization_cycle(
        agent,
        goal,
        rlm_step=_default_rlm_step(args),
        feedback_source=_default_feedback_source(agent, args),
        candidate_builder=_default_candidate_builder(agent, args),
        prover=_default_prover(args),
        gate=_default_gate(args),
        planner=planner,
        publish_fn=_default_publish(agent, args),
    )

    rlm = report.rlm
    print(
        "[ail.jobs.optimization_cycle] "
        f"rlm_reviewed={rlm.n_reviewed if rlm else 0} "
        f"rlm_failed={rlm.n_failed if rlm else 0} rlm_error={report.rlm_error} "
        f"decisions_a={report.plan.n_from_a} decisions_b={report.plan.n_from_b} "
        f"deduped={report.plan.n_deduped} planner_error={report.plan.planner_error} "
        f"proposals={len(report.cycle.proposals)} skipped={len(report.cycle.skipped)} "
        f"published={report.n_published}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
