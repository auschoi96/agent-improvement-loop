"""``ail-local-cycle`` — run the full optimization cycle **locally**, surfacing every step.

This is the local-machine sibling of the scheduled serverless job
(:mod:`ail.jobs.optimization_cycle`). Its reason to exist: the cycle's proof step
(:func:`ail.optimize.phase2.run_phase2_comparison`) drives the **Claude Agent SDK**
through :class:`~ail.ingest.adapters.claude_code.ClaudeCodeAdapter`, which needs a
local Claude auth + a local filesystem for its per-arm git worktrees. Serverless
compute cannot run that prover; a laptop with ``pip install claude-agent-sdk`` (a
self-contained bundle — no Node/CLI to install) and local Claude auth can. So the
local runner is where the **real** prover actually executes end-to-end — the opt-in
Tier-2 verification of ``docs/PRODUCT_ARCHITECTURE.md`` §3, run to completion.

**It reuses the exact spine, unmodified.** The whole detect → decide → prove → gate →
propose → publish pipeline is :func:`ail.jobs.optimization_cycle.run_optimization_cycle`
(→ :func:`ail.loop.planner.run_cycle_with_planner` → :func:`ail.loop.controller.run_cycle`),
wired with the **same real seams** the serverless ``main`` uses: the in-cycle RLM
reviewer (:func:`ail.l3.continuous.run_continuous_rlm`), the real feedback source, the
real cost-guarded candidate builder
(:func:`ail.loop.candidate_builders.token_efficiency_candidate_builder`), the **real
prover**, the real readiness gate (:func:`ail.readiness.compute_readiness`), and the
real UC publish path (:func:`ail.loop.publish_proposals.publish_agent_proposals`, the
same ``agent_proposed_actions`` table the deployed approval-queue app reads). This
module imports those default seam factories rather than re-deriving any of them.

**What is net-new here — and only this:**

1. **Env, static-token auth** (:func:`resolve_local_auth`): ``DATABRICKS_HOST`` +
   ``DATABRICKS_TOKEN`` required, fail-loud if missing, and any ambient
   ``DATABRICKS_CONFIG_PROFILE`` is dropped so a *long* prover run cannot fall onto
   OAuth (which expires mid-run). Deliberately stricter than
   :func:`ail.jobs.publish_job.resolve_job_auth` (which is allowed to mint/refresh).
2. **A reporting layer** (:class:`LocalCycleReporter` + the ``_report_*`` seam
   wrappers): the core requirement — as the cycle runs it prints, structured and
   human-readable, the RLM findings per trace, the Lane A/B plan and *why*, the
   candidate, the baseline-vs-candidate proof (token + tool-call deltas, correctness
   held), the readiness/judge gate, and the proposals written (or the explicit
   fail-closed reason none were). The wrappers only observe and print; they return
   each seam's value verbatim.
3. **A gateway thread**: the RLM/HALO LLM base_url + token
   (:func:`resolve_llm_gateway`), passed to :func:`run_continuous_rlm` so the reviewer
   uses the static bearer, not OAuth.

**Fail-closed is inherited, never weakened.** A failed / errored / timed-out prover
run yields **no** proposal: the controller records it as a fail-closed skip (a crashed
prover raises → the ``_report_prover`` wrapper prints the error and re-raises so
:func:`ail.loop.controller.run_cycle` records the skip; a timed-out / non-improving run
BLOCKs on the frozen suite and never PROMOTEs). This module fabricates no proof and
adds no gate bypass. Proposals are only ever written with a real passing proof and a
cleared gate — the human then approves the apply in the app.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from collections.abc import MutableMapping
from typing import TextIO

from ail.goals.compiler import CompiledGoal
from ail.jobs.optimization_cycle import (
    CandidateBuilder,
    FeedbackSource,
    Gate,
    OptimizationCycleReport,
    Prover,
    PublishFn,
    RlmStep,
    _build_goal,
    _default_candidate_builder,
    _default_feedback_source,
    _default_gate,
    _default_prover,
    _default_publish,
    _opt_float,
    run_optimization_cycle,
)
from ail.l3.continuous import ContinuousRlmRunReport, run_continuous_rlm
from ail.loop.controller import Candidate
from ail.loop.decision_rules import Decision, FeedbackBundle
from ail.loop.planner import Planner, agent_planner
from ail.loop.proposals import ProofSummary, ProposedAction, TriggerKind, TriggerSignal
from ail.loop.publish_proposals import PROPOSALS_TABLE
from ail.optimize.phase2 import Phase2Artifact
from ail.publish import DEFAULT_CATALOG, DEFAULT_SCHEMA
from ail.readiness.contract import ReadinessStatus
from ail.registry import Agent

__all__ = [
    "LocalCycleReporter",
    "resolve_local_auth",
    "resolve_llm_gateway",
    "run_local_cycle",
    "main",
]

#: The tool-call L0 metric the comparison harness emits (``ail.compare.harness``),
#: surfaced alongside the token delta in the proof so the user sees the behavioural
#: change, not just the cost change.
_TOOL_CALLS_METRIC = "total_tool_calls"

_RULE = "─" * 78


# ---------------------------------------------------------------------------
# Auth (net-new: env, static-token; fail-loud; no OAuth for long prover runs)
# ---------------------------------------------------------------------------


def resolve_local_auth(env: MutableMapping[str, str] | None = None) -> tuple[str, str]:
    """Require an explicit static ``DATABRICKS_HOST`` + ``DATABRICKS_TOKEN``; fail loud.

    A local prover run proves a whole frozen suite through the Claude Agent SDK and
    routinely outlives the ~1h life of an OAuth access token. So — unlike
    :func:`ail.jobs.publish_job.resolve_job_auth`, which is allowed to mint/refresh a
    bearer from ambient auth — the local runner insists on a **static** token supplied
    in the environment and refuses to start without it. Any ambient
    ``DATABRICKS_CONFIG_PROFILE`` is removed so no downstream MLflow/SDK call can fall
    back to per-request OAuth resolution (which would expire mid-run).

    Returns ``(host, token)``. Raises :class:`SystemExit` with a clear message when
    either variable is missing — the "fail loud if a required var is missing" contract.
    """
    environ = os.environ if env is None else env
    host = (environ.get("DATABRICKS_HOST") or "").strip()
    token = (environ.get("DATABRICKS_TOKEN") or "").strip()
    missing = [n for n, v in (("DATABRICKS_HOST", host), ("DATABRICKS_TOKEN", token)) if not v]
    if missing:
        raise SystemExit(
            "ail-local-cycle: missing required auth env var(s): "
            f"{', '.join(missing)}.\n"
            "Set a STATIC token matched to the experiment's workspace, e.g.\n"
            "  export DATABRICKS_HOST=https://<workspace-host>\n"
            "  export DATABRICKS_TOKEN=dapi...\n"
            "Use a static PAT (not a --profile OAuth login): a full local prover run "
            "outlives the ~1h OAuth token lifetime."
        )
    # Drop the profile so MLflow's per-request credential resolution cannot fall back
    # to OAuth for some spans while using the explicit bearer for others (the same
    # guard resolve_job_auth applies on the minted path).
    environ.pop("DATABRICKS_CONFIG_PROFILE", None)
    environ["DATABRICKS_HOST"] = host
    environ["DATABRICKS_TOKEN"] = token
    return host, token


def resolve_llm_gateway(env: MutableMapping[str, str], host: str, token: str) -> tuple[str, str]:
    """Resolve the base_url + token for the RLM/HALO LLM gateway (static, no OAuth).

    The in-cycle RLM reviewer calls an OpenAI-compatible chat endpoint. Rather than let
    :func:`ail.l3.reviewer.review_trace` resolve it from a CLI profile (OAuth, expires
    mid-run), the local runner passes an explicit ``base_url`` + ``api_key``:

    * ``AIL_LLM_BASE_URL`` / ``AIL_LLM_API_KEY`` if the operator set them (e.g. a
      dedicated AI-gateway in front of the models); otherwise
    * the workspace's own Foundation Model serving endpoints —
      ``<host>/serving-endpoints`` authenticated by the same static ``DATABRICKS_TOKEN``
      — mirroring :func:`ail.l3.reviewer._resolve_databricks_openai` but with the
      static bearer instead of a minted one.

    The Lane B planner uses MLflow's Databricks deploy client, which reads the same
    ``DATABRICKS_HOST`` / ``DATABRICKS_TOKEN`` — so both model callers ride the one
    static token. Returns ``(base_url, api_key)``.
    """
    default_base_url = f"{host.rstrip('/')}/serving-endpoints"
    base_url = (env.get("AIL_LLM_BASE_URL") or "").strip() or default_base_url
    api_key = (env.get("AIL_LLM_API_KEY") or "").strip() or token
    return base_url, api_key


# ---------------------------------------------------------------------------
# Reporting (net-new: the "surface the feedback + plan to the user" requirement)
# ---------------------------------------------------------------------------


def _lane_label(trigger: TriggerSignal) -> str:
    """Attribute a decision to its lane + rule, straight off the trigger kind."""
    lane_a_rules = {
        TriggerKind.RLM_RECOMMENDED_ASSET: "RLM-recommended-asset rule",
        TriggerKind.REDUNDANT_READ_PATTERN: "redundant-read-pattern rule",
        TriggerKind.JUDGE_DIMENSION_BELOW_THRESHOLD: "judge-dimension rule",
        TriggerKind.POST_APPLY_REGRESSION: "post-apply-regression rule",
    }
    if trigger.kind is TriggerKind.AGENT_PLANNER:
        return "Lane B · LLM planner"
    return f"Lane A · {lane_a_rules.get(trigger.kind, trigger.kind.value)}"


def _fmt_pct(pct: float | None) -> str:
    return "n/a" if pct is None else f"{pct:+.1f}%"


class LocalCycleReporter:
    """Prints each stage of a local cycle in a structured, human-readable form.

    Purely an observer: every method reads already-computed data and writes to a
    stream (``sys.stdout`` by default; resolved at write time so pytest's ``capsys``
    capture is honoured). It never mutates the cycle or influences a decision — the
    ``_report_*`` seam wrappers call these around the unchanged seams.
    """

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream
        self._n_decisions = 0

    # -- low-level -------------------------------------------------------

    def _emit(self, line: str = "") -> None:
        print(line, file=self._stream or sys.stdout)

    def _step(self, title: str) -> None:
        self._emit()
        self._emit("╔" + "═" * 78)
        self._emit(f"║ {title}")
        self._emit("╚" + "═" * 78)

    def warn(self, message: str) -> None:
        self._emit(f"⚠  {message}")

    # -- header ----------------------------------------------------------

    def header(self, *, host: str, agent: Agent, goal: CompiledGoal, base_url: str) -> None:
        self._emit(_RULE)
        self._emit("ail-local-cycle · local self-improvement optimization cycle")
        self._emit(_RULE)
        self._emit(f"  host           : {host}")
        self._emit(f"  agent          : {agent.agent_name}")
        self._emit(f"  experiment     : {agent.experiment_id}")
        self._emit(
            f"  goal           : {goal.direction} {goal.objective_metric} "
            f"(target {goal.target.kind} {goal.target.value}; confirmed={goal.human_confirmed})"
        )
        if goal.guardrails:
            rails = ", ".join(f"{g.name} ({g.kind})" for g in goal.guardrails)
            self._emit(f"  guardrails     : {rails}")
        self._emit(f"  llm gateway    : {base_url}")
        self._emit("  prover         : real Claude Agent SDK on the frozen suite (fail-closed)")

    # -- (a) RLM findings ------------------------------------------------

    def rlm_start(self) -> None:
        self._step("STEP 1/5 · IN-CYCLE RLM REVIEW (per-trace findings)")

    def rlm_failed(self, exc: BaseException) -> None:
        self.warn(
            f"RLM review failed (non-blocking): {type(exc).__name__}: {exc} — "
            "the cycle continues over already-attached feedback; no verdict is fabricated."
        )

    def rlm_findings(self, report: ContinuousRlmRunReport) -> None:
        self._emit(
            f"  scanned={report.n_scanned} already-reviewed={report.n_already_reviewed} "
            f"sampled-out={report.n_sampled_out} selected={report.n_selected} "
            f"reviewed={report.n_reviewed} failed={report.n_failed} "
            f"(judge={report.judge_model}, sample_rate={report.sample_rate})"
        )
        if not report.outcomes:
            self._emit("  (no traces reviewed this cycle)")
            return
        for o in report.outcomes:
            if o.status == "reviewed":
                self._emit(
                    f"  ✓ {o.trace_id}: efficiency={o.token_efficiency} "
                    f"waste_score={o.token_waste_score} "
                    f"recommended_assets={o.n_recommended_assets} "
                    f"(tokens={o.total_tokens})"
                )
            else:
                self._emit(f"  ✗ {o.trace_id}: review_failed — {o.error}")

    # -- (b-context) feedback --------------------------------------------

    def feedback(self, bundle: FeedbackBundle) -> None:
        self._step("STEP 2/5 · FEEDBACK SIGNALS (what the planners act on)")
        self._emit(
            f"  objective value: {bundle.objective_metric_value}  "
            f"(baseline {bundle.objective_baseline_value})"
        )

        self._emit(f"  RLM-recommended assets ({len(bundle.rlm_assets)}):")
        for a in bundle.rlm_assets:
            self._emit(
                f"    - [{a.asset_type}] {a.title!r}: recurred across {a.n_traces} trace(s), "
                f"rank {a.rank}"
            )
        if not bundle.rlm_assets:
            self._emit("    (none)")

        self._emit(f"  L0 redundant-read patterns ({len(bundle.redundant_reads)}):")
        for r in bundle.redundant_reads:
            target = r.repeated_target or r.tool or "repeated target"
            self._emit(f"    - {target!r}: repeated {r.occurrences}x; dominant={r.dominant}")
        if not bundle.redundant_reads:
            self._emit("    (none)")

        self._emit(
            f"  judge dimensions below par: {len(bundle.judge_dimensions)}   "
            f"post-apply regressions: {len(bundle.post_apply_regressions)}"
        )

    # -- (e) gate / readiness --------------------------------------------

    def readiness(self, status: ReadinessStatus) -> None:
        self._step("STEP 3/5 · READINESS GATE (data sufficiency + judge trust)")
        self._emit(
            f"  tier={status.tier.value}  can_prove_improvement={status.can_prove_improvement}  "
            f"scored_coverage={status.eval_health.scored_coverage:.2f}  "
            f"distrusted_judges={status.eval_health.n_distrusted_judges}"
        )
        if status.reasons:
            for reason in status.reasons:
                self._emit(f"    - unmet: {reason}")
        for jh in status.eval_health.judges:
            trust = "distrusted" if jh.distrusted else "trusted"
            self._emit(
                f"    - judge {jh.judge_name!r}: {trust} "
                f"(measured={jh.measured}, agreement={jh.agreement_rate})"
            )

    # -- (b) decision + (c) candidate ------------------------------------

    def decision(self, decision: Decision) -> None:
        if self._n_decisions == 0:
            self._step("STEP 4/5 · PLAN → PROVE (per decision; fail-closed)")
        self._n_decisions += 1
        t = decision.trigger
        self._emit()
        self._emit(f"  ── Decision #{self._n_decisions}: {decision.action_kind.value} ──")
        self._emit(f"     lane   : {_lane_label(t)}")
        self._emit(f"     why    : {t.summary}")
        if t.trace_refs:
            self._emit(f"     traces : {list(t.trace_refs)}")

    def candidate(self, candidate: Candidate | None) -> None:
        if candidate is None:
            self._emit(
                "     candidate: NONE — fail-closed (no frozen-suite-provable change for this "
                "decision); no proposal."
            )
            return
        change = candidate.change
        self._emit(f"     candidate: [{change.kind.value}] {change.summary}")
        diff = change.diff or change.sql or change.evolved_body_ref or change.revert_target or ""
        head = [ln for ln in diff.splitlines() if ln.strip()][:6]
        for ln in head:
            self._emit(f"       | {ln}")

    # -- (d) proof -------------------------------------------------------

    def proving(self) -> None:
        self._emit("     proving  : running baseline vs candidate on the frozen suite …")

    def prove_failed(self, exc: BaseException) -> None:
        self._emit(
            f"     PROOF FAILED (fail-closed): {type(exc).__name__}: {exc} — "
            "no proposal for this decision."
        )

    def proof(self, artifact: Phase2Artifact) -> None:
        summary = ProofSummary.from_phase2_artifact(artifact)
        self._emit(
            f"     proof    : {artifact.n_promote} PROMOTE / {artifact.n_block} BLOCK / "
            f"{artifact.n_errored} ERRORED over {artifact.n_tasks} task(s) "
            f"[suite {artifact.suite_version}]"
        )
        self._emit(
            f"       realized token savings (PROMOTE only): "
            f"{artifact.realized_token_savings_absolute:,.0f} "
            f"({_fmt_pct(artifact.realized_token_savings_pct)})  "
            f"[{artifact.realized_baseline_tokens:,.0f} → "
            f"{artifact.realized_candidate_tokens:,.0f}]"
        )
        self._emit(
            f"       proved_improvement={summary.proved_improvement}  "
            f"correctness_held={summary.correctness_held}"
        )
        for o in artifact.outcomes:
            tool_line = ""
            if o.comparison is not None:
                td = o.comparison.delta_for(_TOOL_CALLS_METRIC)
                if td is not None:
                    tool_line = (
                        f"  tools {td.baseline:,.0f}→{td.candidate:,.0f} "
                        f"(Δ{td.delta_absolute:+,.0f})"
                    )
            self._emit(
                f"       · {o.task_id} [{o.category}/{o.difficulty}]: {o.recommendation.value}  "
                f"tokens {o.baseline_total_tokens:,.0f}→{o.candidate_total_tokens:,.0f} "
                f"(Δ{o.token_delta_absolute:+,.0f}, {_fmt_pct(o.token_delta_pct)})"
                f"{tool_line}  correctness={o.l1_outcome.value}"
            )

    # -- (f) publish -----------------------------------------------------

    def published(self, proposals: list[ProposedAction], n_written: int) -> None:
        self._step("STEP 5/5 · PROPOSALS WRITTEN TO agent_proposed_actions")
        if not proposals:
            self._emit(
                "  0 proposals — nothing cleared proof + gate this cycle (fail-closed). "
                f"The agent's slice of `{PROPOSALS_TABLE}` was replaced with the empty set "
                "(any superseded pending proposal is cleared)."
            )
            return
        self._emit(f"  {n_written} proposal(s) written for approval in the app:")
        for p in proposals:
            self._emit(
                f"  • {p.proposal_id} [{p.action_kind.value}] status={p.status.value}\n"
                f"      proof: saved {p.proof.realized_savings_absolute:,.0f} tokens "
                f"({_fmt_pct(p.proof.realized_savings_pct)}), "
                f"correctness_held={p.proof.correctness_held}\n"
                f"      gate : tier={p.gate_status.readiness_tier} gated={p.gate_status.gated}"
            )

    # -- summary ---------------------------------------------------------

    def summary(self, report: OptimizationCycleReport) -> None:
        self._step("SUMMARY")
        plan = report.plan
        self._emit(
            f"  plan: Lane A={plan.n_from_a}  Lane B={plan.n_from_b}  "
            f"deduped={plan.n_deduped}  planner_error={plan.planner_error}"
        )
        if report.rlm_error:
            self._emit(f"  rlm : failed (non-blocking) — {report.rlm_error}")
        self._emit(
            f"  result: {len(report.cycle.proposals)} proposal(s), "
            f"{len(report.cycle.skipped)} skipped, {report.n_published} published."
        )
        if report.cycle.skipped:
            self._emit("  fail-closed skips (considered but not proposed):")
            for s in report.cycle.skipped:
                self._emit(f"    - {s.action_kind}/{s.trigger_kind}: {s.reason}")


# ---------------------------------------------------------------------------
# Seam wrappers: observe + print, return the seam's value verbatim.
# ---------------------------------------------------------------------------


def _report_rlm(inner: RlmStep, rep: LocalCycleReporter) -> RlmStep:
    def step() -> ContinuousRlmRunReport:
        rep.rlm_start()
        try:
            report = inner()
        except Exception as exc:  # noqa: BLE001 - print live, then re-raise so the
            rep.rlm_failed(exc)  # cycle's own non-blocking catch records rlm_error
            raise
        rep.rlm_findings(report)
        return report

    return step


def _report_feedback(inner: FeedbackSource, rep: LocalCycleReporter) -> FeedbackSource:
    def source() -> FeedbackBundle:
        bundle = inner()
        rep.feedback(bundle)
        return bundle

    return source


def _report_gate(inner: Gate, rep: LocalCycleReporter) -> Gate:
    def gate(*, goal: CompiledGoal, agent: Agent) -> ReadinessStatus:
        status = inner(goal=goal, agent=agent)
        rep.readiness(status)
        return status

    return gate


def _report_candidate(inner: CandidateBuilder, rep: LocalCycleReporter) -> CandidateBuilder:
    def build(decision: Decision, *, goal: CompiledGoal, agent: Agent) -> Candidate | None:
        rep.decision(decision)
        candidate = inner(decision, goal=goal, agent=agent)
        rep.candidate(candidate)
        return candidate

    return build


def _report_prover(inner: Prover, rep: LocalCycleReporter) -> Prover:
    def prove(candidate: Candidate, *, goal: CompiledGoal, agent: Agent) -> Phase2Artifact:
        rep.proving()
        try:
            artifact = inner(candidate, goal=goal, agent=agent)
        except Exception as exc:  # noqa: BLE001 - print the honest error, then re-raise
            rep.prove_failed(exc)  # so run_cycle records a fail-closed skip (no proposal)
            raise
        rep.proof(artifact)
        return artifact

    return prove


def _report_publish(inner: PublishFn, rep: LocalCycleReporter) -> PublishFn:
    def publish(proposals: list[ProposedAction]) -> int:
        n = inner(proposals)
        rep.published(proposals, n)
        return n

    return publish


# ---------------------------------------------------------------------------
# The injectable local cycle (testable with fakes) + real-seam entrypoint.
# ---------------------------------------------------------------------------


def run_local_cycle(
    agent: Agent,
    goal: CompiledGoal,
    *,
    rlm_step: RlmStep,
    feedback_source: FeedbackSource,
    candidate_builder: CandidateBuilder,
    prover: Prover,
    gate: Gate,
    publish_fn: PublishFn,
    planner: Planner = agent_planner,
    reporter: LocalCycleReporter | None = None,
    now: str | None = None,
) -> OptimizationCycleReport:
    """Run one cycle through the **unchanged** spine, wrapping each seam with reporting.

    This is a thin, injectable wrapper over
    :func:`ail.jobs.optimization_cycle.run_optimization_cycle`: it decorates the RLM,
    feedback, candidate-builder, prover, gate, and publish seams with the
    :class:`LocalCycleReporter` observers and delegates the *entire* orchestration
    (in-cycle RLM → layered A+B plan → prove → gate → propose → publish, fail-closed)
    to that function. The reporting wrappers return every value verbatim, so the proof,
    gate, and fail-closed behaviour are byte-for-byte the serverless ones. A prover that
    raises still surfaces as a fail-closed skip (never a proposal). Returns the same
    :class:`~ail.jobs.optimization_cycle.OptimizationCycleReport` for programmatic use.
    """
    rep = reporter or LocalCycleReporter()
    report = run_optimization_cycle(
        agent,
        goal,
        rlm_step=_report_rlm(rlm_step, rep),
        feedback_source=_report_feedback(feedback_source, rep),
        candidate_builder=_report_candidate(candidate_builder, rep),
        prover=_report_prover(prover, rep),
        gate=_report_gate(gate, rep),
        planner=planner,
        publish_fn=_report_publish(publish_fn, rep),
        now=now,
    )
    rep.summary(report)
    return report


def _local_rlm_step(args: argparse.Namespace, *, base_url: str, api_key: str) -> RlmStep:
    """Real in-cycle RLM step, threading the explicit static-token LLM gateway.

    Mirrors :func:`ail.jobs.optimization_cycle._default_rlm_step` (the same reviewer,
    the same sampling knobs) but passes ``base_url`` / ``api_key`` so HALO uses the
    static bearer rather than resolving an OAuth endpoint from a CLI profile.
    """

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
            base_url=base_url,
            api_key=api_key,
        )

    return _step


def _preflight_claude_sdk(rep: LocalCycleReporter) -> bool:
    """Warn (loudly, but non-fatally) if the Claude Agent SDK is not importable.

    The prover arms need ``claude-agent-sdk`` (self-contained; ``pip install
    claude-agent-sdk``). If it is absent, every candidate's proof will raise and be
    recorded as a fail-closed skip — no proposal, no fabricated proof. We surface that
    up front so the operator understands *why* no proposals will be written, rather
    than leave it buried per-decision. The RLM review + planning still run and are
    reported. (Local Claude auth itself is verified by the SDK at run time.)
    """
    if importlib.util.find_spec("claude_agent_sdk") is not None:
        return True
    rep.warn(
        "claude-agent-sdk is not importable — the prover arms cannot run, so every "
        "candidate will fail closed (no proposals). Install it and authenticate Claude "
        'locally: pip install -e ".[claude,align,l3,agents]" and log in to Claude.'
    )
    return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ail-local-cycle",
        description=(
            "Run one full optimization cycle LOCALLY (where the Claude Agent SDK prover "
            "can execute): in-cycle RLM review, layered A+B planning, REAL frozen-suite "
            "proof, readiness+judge gate, and PENDING-proposal publish — surfacing every "
            "step. Propose-only + fail-closed; a human approves the apply in the app."
        ),
    )
    parser.add_argument("--agent", default="claude_code", help="Agent name (proposal scope).")
    parser.add_argument("--experiment", required=True, help="MLflow experiment id (the agent's).")
    parser.add_argument("--warehouse-id", default=os.environ.get("AIL_WAREHOUSE_ID"))
    parser.add_argument("--catalog", default=DEFAULT_CATALOG)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    # in-cycle RLM sampling knobs (the existing ail.l3.continuous scheme; no new one)
    parser.add_argument("--judge-model", required=True, help="HALO judge serving-endpoint name.")
    parser.add_argument("--max-results", type=int, default=100)
    parser.add_argument("--max-reviews", type=int, default=2)
    parser.add_argument("--sample-rate", type=float, default=0.10)
    parser.add_argument("--min-tokens", type=int, default=50_000)
    parser.add_argument("--reviewer-experiment", default="")
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=None)
    # goal (operator-configured; confirmed by --confirm-goal / AIL_CONFIRM_GOAL)
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
        help="Baseline a relative objective target is measured against "
        "(empty => treated as not-yet-met, no fabricated baseline).",
    )
    parser.add_argument(
        "--confirm-goal",
        dest="goal_confirmed",
        action="store_const",
        const="true",
        default=os.environ.get("AIL_CONFIRM_GOAL", "false"),
        help="Mark the operator-configured goal human-confirmed (required to run). "
        "Without it (or AIL_CONFIRM_GOAL=1) the runner refuses to optimize an unreviewed goal.",
    )
    parser.add_argument("--planner-model", default=None, help="Lane B planner endpoint (optional).")
    parser.add_argument(
        "--llm-base-url",
        default=None,
        help="LLM gateway base_url for the RLM/HALO review (defaults to "
        "AIL_LLM_BASE_URL, then <host>/serving-endpoints).",
    )
    parser.add_argument(
        "--llm-api-key",
        default=None,
        help="LLM gateway token (defaults to AIL_LLM_API_KEY, then DATABRICKS_TOKEN).",
    )
    # The local runner uses the static env token, never a CLI profile (long prover runs
    # outlive OAuth). The _default_* seam factories read args.profile, so pin it to None.
    parser.set_defaults(profile=None)
    args = parser.parse_args(argv)
    if not args.warehouse_id:
        parser.error("--warehouse-id is required (or set AIL_WAREHOUSE_ID)")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # 1. Auth: static env token, fail-loud (net-new; stricter than resolve_job_auth).
    host, token = resolve_local_auth()
    if args.llm_base_url:
        os.environ["AIL_LLM_BASE_URL"] = args.llm_base_url
    if args.llm_api_key:
        os.environ["AIL_LLM_API_KEY"] = args.llm_api_key
    base_url, api_key = resolve_llm_gateway(os.environ, host, token)

    # 2. Goal: refuse to run an unconfirmed goal (fail-loud before any live work).
    confirmed = str(args.goal_confirmed).strip().lower() in {"1", "true", "yes"}
    if not confirmed:
        raise SystemExit(
            "ail-local-cycle: refusing to run on an unconfirmed goal. Review the goal, then "
            "pass --confirm-goal (or set AIL_CONFIRM_GOAL=1)."
        )
    goal = _build_goal(args)
    agent = Agent(agent_name=args.agent, experiment_id=args.experiment)

    planner: Planner = agent_planner
    if args.planner_model:
        from functools import partial

        planner = partial(agent_planner, model=args.planner_model)

    reporter = LocalCycleReporter()
    reporter.header(host=host, agent=agent, goal=goal, base_url=base_url)
    _preflight_claude_sdk(reporter)

    run_local_cycle(
        agent,
        goal,
        rlm_step=_local_rlm_step(args, base_url=base_url, api_key=api_key),
        feedback_source=_default_feedback_source(agent, args),
        candidate_builder=_default_candidate_builder(agent, args),
        prover=_default_prover(args),
        gate=_default_gate(args),
        planner=planner,
        publish_fn=_default_publish(agent, args),
        reporter=reporter,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
