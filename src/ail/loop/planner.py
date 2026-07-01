"""Lane B: the LLM-agent planner — a *second* decision source for the loop.

``docs/LOOP_CONTROLLER.md`` step 2 (decide) has one deterministic implementation
today: :func:`ail.loop.decision_rules.decide`, the pure rule engine (**Lane A**).
This module adds **Lane B**: an LLM agent that reads the *same*
:class:`~ail.loop.decision_rules.FeedbackBundle` and proposes *what to try* as the
**same** :class:`~ail.loop.decision_rules.Decision` type Lane A returns — so a B
decision flows through the **exact same** candidate_builder → prover → gate →
propose pipeline as an A decision (:func:`ail.loop.controller.run_cycle`). Nothing
about proving, gating, or applying changes here.

Three hard lines, all enforced below and echoed by the tests:

* **B proposes; it never builds, proves, applies, aliases, or ``CREATE``s.** This
  module produces :class:`Decision` objects and nothing else — there is no apply
  seam, registry call, or SQL here (grep-provable).
* **Fail-closed parsing (mirrors the HALO parser,** :mod:`ail.l3.parser`\\ **).**
  Malformed, empty, or all-low-confidence agent output yields **zero** decisions
  via a typed :class:`PlanParseError` the caller records as a skip — never a
  fabricated decision. Individual bad plan entries degrade (dropped with a
  warning), exactly as HALO drops an unscorable guideline; only when *nothing*
  usable survives does the whole parse fail loud.
* **A-vs-B attributable.** Every B decision carries a
  :class:`~ail.loop.proposals.TriggerSignal` with
  :attr:`~ail.loop.proposals.TriggerKind.AGENT_PLANNER` as its ``kind``, so a
  proposal's origin is readable straight off ``trigger.kind`` while the target
  detail (judge / metric / asset type) still travels for the gates.

:func:`combined_decisions` runs both lanes and returns the de-duped **A ∪ B**
union; :func:`run_cycle_with_planner` is the thin wrapper that feeds that union
through :func:`~ail.loop.controller.run_cycle`'s unchanged pipeline (via its
``decisions`` injection seam) — it does **not** fork the proven cycle.

**The LLM is an injectable seam.** :class:`PlannerLLM` mirrors
:class:`ail.goals.compiler.GoalProposerLLM`: a ``(system, user) -> str`` callable.
Production resolves a Databricks chat serving endpoint via MLflow's Databricks
deploy client (the same path :func:`ail.goals.compiler._default_databricks_proposer`
uses — no hardcoded host); every test injects a canned callable so no live model
call is ever made.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from ail.goals.compiler import CompiledGoal
from ail.loop.controller import (
    CandidateBuilder,
    CycleResult,
    FeedbackSource,
    Gate,
    Prover,
    run_cycle,
)
from ail.loop.decision_rules import (
    Decision,
    DecisionThresholds,
    FeedbackBundle,
    decide,
    objective_target_met,
)
from ail.loop.proposals import (
    ActionKind,
    RiskClass,
    TriggerKind,
    TriggerSignal,
    default_risk_class,
)
from ail.registry import Agent

__all__ = [
    "DEFAULT_PLANNER_MODEL",
    "DEFAULT_CONFIDENCE_FLOOR",
    "PlanParseError",
    "PlannerLLM",
    "Planner",
    "CombinedDecisions",
    "PlannerCycleResult",
    "build_planner_prompt",
    "parse_plan",
    "agent_planner",
    "combined_decisions",
    "run_cycle_with_planner",
]

#: Default planner model URI (MLflow ``provider:/endpoint`` form). Matches the
#: repo's existing reflection/judge default
#: (:data:`ail.optimize.gepa_runner.DEFAULT_REFLECTION_LM`), so Lane B, GEPA, and
#: the L3 judge all default to the same Databricks endpoint. No hardcoded host —
#: the deploy client resolves the workspace from ambient auth.
DEFAULT_PLANNER_MODEL = "databricks:/databricks-claude-sonnet-4-6"

#: Below this confidence a proposed plan entry is dropped (fail-closed: a
#: low-confidence guess is not surfaced as a decision). A caller may raise or lower
#: it; a value of ``0.0`` keeps every well-formed entry.
DEFAULT_CONFIDENCE_FLOOR = 0.5

#: The action kinds Lane B is allowed to propose — exactly the controller's
#: vocabulary (:class:`ail.loop.proposals.ActionKind`). A plan entry naming
#: anything else is dropped, never coerced.
_ALLOWED_ACTION_KINDS: frozenset[str] = frozenset(k.value for k in ActionKind)


class PlanParseError(ValueError):
    """Raised when the agent planner produced no usable, sufficiently-confident plan.

    The Lane-B analogue of :class:`ail.l3.parser.HaloReportParseError`: a degenerate
    planner response — no parseable JSON object, no ``plan`` array, or an array
    from which **no** entry survives validation and the confidence floor — **must
    fail loudly**. Silently returning ``[]`` would erase the distinction between
    "the planner proposed nothing this cycle" and "the planner is broken / the
    model hallucinated", and could let a malformed response masquerade as a clean
    "no action needed". The caller (:func:`combined_decisions`) catches it and
    records a skip, so Lane A still runs and the failure stays visible/auditable —
    but a fabricated decision is never returned in its place.
    """


class PlannerLLM(Protocol):
    """The injectable LLM seam: maps a ``(system, user)`` prompt to raw model text.

    Identical in shape to :class:`ail.goals.compiler.GoalProposerLLM`, so the same
    kind of callable satisfies both. Production supplies a Databricks chat endpoint
    (see :func:`_default_planner_llm`); tests inject a canned-string callable so no
    live model call is made.
    """

    def __call__(self, *, system: str, user: str) -> str: ...


class Planner(Protocol):
    """A Lane-B decision source: feedback + goal + agent → proposed decisions.

    Returns the **same** :class:`~ail.loop.decision_rules.Decision` type Lane A's
    :func:`ail.loop.decision_rules.decide` returns, so its output flows through the
    controller's build → prove → gate → propose pipeline unchanged. It **proposes
    only** — it must never build a candidate, prove, gate, or apply anything.

    May raise :class:`PlanParseError` when it produced nothing usable (fail-closed);
    callers that combine lanes catch it so Lane A is unaffected.
    """

    def __call__(
        self, feedback: FeedbackBundle, goal: CompiledGoal, agent: Agent
    ) -> list[Decision]: ...


@dataclass(frozen=True, slots=True)
class CombinedDecisions:
    """The de-duped **A ∪ B** decision union plus its provenance counts.

    ``decisions`` is what the controller iterates (Lane A first, then the Lane-B
    decisions that were not already covered by A). The counts and
    ``planner_error`` make a cycle auditable: a reviewer can see how many decisions
    each lane contributed, how many B duplicates were dropped, and — if Lane B
    failed closed — the recorded reason (never a fabricated decision).
    """

    decisions: list[Decision]
    n_from_a: int = 0
    n_from_b: int = 0
    n_deduped: int = 0
    planner_error: str | None = None


@dataclass(frozen=True, slots=True)
class PlannerCycleResult:
    """The result of a layered A+B cycle: the controller's output + the plan record."""

    result: CycleResult
    plan: CombinedDecisions


# ---------------------------------------------------------------------------
# Prompt composition
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an optimization planner for an autonomous agent-improvement loop. You are \
given structured feedback about one agent's recent execution traces and a \
confirmed optimization goal. Propose which improvement ACTIONS the loop should \
try — you decide *what to try*, nothing more. You never build, prove, or apply \
anything; a separate, deterministic pipeline proves each proposed action on a \
frozen task suite and a human approves any live change. So propose freely but \
honestly: ground every proposal in the feedback you were given.

Return ONLY a single JSON object, with no prose before or after it, of the form:

{
  "plan": [
    {
      "action_kind": "<one of: <<ACTION_KINDS>>>",
      "rationale": "<one or two sentences, grounded in the feedback above>",
      "confidence": <number between 0 and 1>,
      "metric": "<objective metric or judged dimension this targets, or null>",
      "judge_name": "<judge whose dimension this targets, or null>",
      "asset_type": "<recommended asset type for a metric_view, or null>",
      "trace_refs": ["<trace id from the feedback>", "..."]
    }
  ]
}

Rules:
- Use ONLY the listed action_kind values. Omit any action you cannot ground in \
the feedback rather than inventing one.
- For an action that targets a judged quality dimension (typically gepa_prompt), \
set "judge_name" to that judge so the loop can gate on its trust.
- "confidence" is your own estimate that the action would help; be calibrated. \
Low-confidence guesses will be discarded.
- If the feedback motivates no action, return {"plan": []}.\
"""


def build_planner_prompt(feedback: FeedbackBundle, goal: CompiledGoal) -> tuple[str, str]:
    """Render the ``(system, user)`` prompt for the agent planner.

    The system prompt fixes the role, the allowed action vocabulary, and the strict
    JSON schema (so :func:`parse_plan` has something to validate against). The user
    prompt is a faithful, compact rendering of the goal and the four
    :class:`FeedbackBundle` signal families — no signal is embellished or invented.
    """
    system = _SYSTEM_PROMPT.replace("<<ACTION_KINDS>>", " | ".join(sorted(_ALLOWED_ACTION_KINDS)))

    met = objective_target_met(
        goal,
        observed=feedback.objective_metric_value,
        baseline=feedback.objective_baseline_value,
    )
    lines: list[str] = []
    lines.append("GOAL")
    lines.append(f"- objective_metric: {goal.objective_metric}")
    lines.append(f"- direction: {goal.direction}")
    lines.append(f"- target: {goal.target.kind} {goal.target.value}")
    if goal.guardrails:
        rails = ", ".join(f"{g.name} ({g.kind})" for g in goal.guardrails)
        lines.append(f"- guardrails: {rails}")
    lines.append(
        f"- current objective value: {feedback.objective_metric_value} "
        f"(baseline {feedback.objective_baseline_value}); "
        f"target already met: {met}"
    )
    lines.append("")

    lines.append("RLM-RECOMMENDED ASSETS (recurrence-ranked across the cohort)")
    if feedback.rlm_assets:
        for a in feedback.rlm_assets:
            lines.append(
                f"- [{a.asset_type}] {a.title!r}: recurred across {a.n_traces} trace(s), "
                f"rank {a.rank}; traces {list(a.trace_ids)}"
            )
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("L0 WASTE / REDUNDANT-READ PATTERNS")
    if feedback.redundant_reads:
        for r in feedback.redundant_reads:
            target = r.repeated_target or r.tool or "repeated target"
            waste = (
                f", ~{r.estimated_wasted_tokens} wasted tokens"
                if r.estimated_wasted_tokens is not None
                else ""
            )
            lines.append(
                f"- {target!r}: repeated {r.occurrences} time(s){waste}; "
                f"dominant={r.dominant}; traces {list(r.trace_ids)}"
            )
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("JUDGE DIMENSIONS BELOW PAR")
    if feedback.judge_dimensions:
        for j in feedback.judge_dimensions:
            lines.append(
                f"- judge {j.judge_name!r} dimension {j.dimension!r}: score {j.score}, "
                f"trusted={j.trusted}; traces {list(j.trace_ids)}"
            )
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("POST-APPLY REGRESSIONS")
    if feedback.post_apply_regressions:
        for p in feedback.post_apply_regressions:
            lines.append(
                f"- version {p.agent_version!r} vs {p.predecessor_version!r} on "
                f"{p.objective_metric!r}: regressed={p.regressed}; traces {list(p.trace_ids)}"
            )
    else:
        lines.append("- (none)")

    return system, "\n".join(lines)


# ---------------------------------------------------------------------------
# Fail-closed parsing (mirrors ail.l3.parser)
# ---------------------------------------------------------------------------


def _strip_code_fences(text: str) -> str:
    """Strip a surrounding ``` / ```json markdown fence, if the model added one."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def _extract_json_object(raw: str) -> dict[str, Any] | None:
    """Decode the first JSON object in ``raw`` (fenced, or the outermost braces).

    Mirrors :func:`ail.goals.compiler._parse_proposal_json` / the HALO parser's
    tolerance: try the fence-stripped body first, then fall back to the outermost
    ``{...}`` span in case the model wrapped the object in stray prose. Returns
    ``None`` when nothing parses to a JSON object — the caller fails closed on that.
    """
    text = _strip_code_fences(raw)
    for candidate in (text, _outermost_braces(text)):
        if candidate is None:
            continue
        try:
            decoded = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(decoded, dict):
            return decoded
    return None


def _outermost_braces(text: str) -> str | None:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    return text[start : end + 1]


def _coerce_confidence(value: Any) -> float | None:
    """Coerce a confidence to a float in ``0..1``, or ``None`` if unusable.

    A boolean is rejected (``bool`` is an ``int`` subclass, so ``float(True)`` would
    otherwise sneak ``1.0`` past the check — the same guard the HALO parser applies
    to scores). A non-numeric or out-of-``[0, 1]`` value returns ``None`` so the
    entry is dropped rather than silently kept at a fabricated confidence.
    """
    if isinstance(value, bool):
        return None
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return None
    if conf < 0.0 or conf > 1.0:
        return None
    return conf


def _str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if value in (None, ""):
        return []
    return [str(value)]


def _opt_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _decision_from_entry(
    entry: Any, *, confidence_floor: float, warnings: list[str]
) -> Decision | None:
    """Build one Lane-B :class:`Decision` from a plan entry, or ``None`` to drop it.

    Fail-closed and defensive (mirrors the HALO parser's per-item coercion): an
    entry that is not an object, names an unknown/absent ``action_kind``, carries no
    non-empty ``rationale``, or whose ``confidence`` is unparseable or below
    ``confidence_floor`` is dropped with a recorded warning — never coerced into a
    fabricated decision. The trigger is stamped
    :attr:`~ail.loop.proposals.TriggerKind.AGENT_PLANNER` (the A-vs-B marker) and
    carries the target detail (metric / judge / asset type / trace refs) so the
    controller's proof and certifying-judge gates apply to it unchanged.
    """
    if not isinstance(entry, dict):
        warnings.append("dropped a non-object plan entry")
        return None

    raw_kind = str(entry.get("action_kind", "")).strip()
    if raw_kind not in _ALLOWED_ACTION_KINDS:
        warnings.append(f"dropped plan entry with unknown action_kind {raw_kind!r}")
        return None
    action_kind = ActionKind(raw_kind)

    rationale = str(entry.get("rationale", "")).strip()
    if not rationale:
        warnings.append(f"dropped {raw_kind!r} plan entry with no rationale")
        return None

    confidence = _coerce_confidence(entry.get("confidence"))
    if confidence is None:
        warnings.append(
            f"dropped {raw_kind!r} plan entry: missing or invalid confidence "
            f"{entry.get('confidence')!r}"
        )
        return None
    if confidence < confidence_floor:
        warnings.append(
            f"dropped {raw_kind!r} plan entry below confidence floor "
            f"({confidence} < {confidence_floor})"
        )
        return None

    judge_name = _opt_str(entry.get("judge_name"))
    metric = _opt_str(entry.get("metric"))
    asset_type = _opt_str(entry.get("asset_type"))
    trace_refs = _str_list(entry.get("trace_refs"))

    trigger = TriggerSignal(
        kind=TriggerKind.AGENT_PLANNER,
        summary=(
            f"agent planner proposed {action_kind.value} (confidence {confidence}): {rationale}"
        ),
        metric=metric,
        observed_value=confidence,
        n_traces=len(trace_refs),
        trace_refs=trace_refs,
        judge_name=judge_name,
        asset_type=asset_type,
    )
    return Decision(action_kind, _risk_class_for(action_kind), trigger)


def _risk_class_for(action_kind: ActionKind) -> RiskClass:
    """The (informational) risk class for a B decision — the controller's default."""
    return default_risk_class(action_kind)


def parse_plan(
    raw: str,
    *,
    confidence_floor: float = DEFAULT_CONFIDENCE_FLOOR,
) -> list[Decision]:
    """Parse the planner's raw text into Lane-B decisions, **fail-closed**.

    Extracts the JSON object, then validates each ``plan`` entry (see
    :func:`_decision_from_entry`), dropping bad/low-confidence entries with a
    warning. Raises :class:`PlanParseError` when the response has no parseable JSON
    object, no ``plan`` array, or **no** entry survives — mirroring
    :func:`ail.l3.parser.parse_halo_report`, where individual optional items degrade
    but a wholly-unusable response fails loud rather than returning a fabricated
    default. Never returns a fabricated decision.
    """
    payload = _extract_json_object(raw)
    if payload is None:
        raise PlanParseError(
            "agent planner returned no parseable JSON object (the response likely was not a plan)"
        )
    plan = payload.get("plan")
    if not isinstance(plan, list):
        raise PlanParseError(
            f"agent planner response has no 'plan' array (got {type(plan).__name__})"
        )

    warnings: list[str] = []
    decisions: list[Decision] = []
    for entry in plan:
        decision = _decision_from_entry(entry, confidence_floor=confidence_floor, warnings=warnings)
        if decision is not None:
            decisions.append(decision)

    if not decisions:
        detail = "; ".join(warnings) if warnings else "the plan was empty"
        raise PlanParseError(
            f"agent planner produced no usable, sufficiently-confident decisions ({detail})"
        )
    return decisions


# ---------------------------------------------------------------------------
# The planner + the layered A∪B combination
# ---------------------------------------------------------------------------


def _default_planner_llm(model: str) -> PlannerLLM:
    """Build the production planner LLM: a Databricks chat serving endpoint.

    Reuses the model-call path :func:`ail.goals.compiler._default_databricks_proposer`
    uses — MLflow's Databricks deploy client at temperature 0 — rather than adding a
    new client. ``model`` is an MLflow ``databricks:/<endpoint>`` URI; the leading
    ``databricks:/`` is stripped to the bare endpoint the deploy client wants. No
    hardcoded host: the client resolves the workspace from ambient Databricks auth.
    """
    endpoint = model.split(":/", 1)[1] if model.startswith("databricks:/") else model
    if not endpoint:
        raise ValueError(f"planner model {model!r} does not name an endpoint")

    def _call(*, system: str, user: str) -> str:
        from mlflow.deployments import get_deploy_client

        client = get_deploy_client("databricks")
        response = client.predict(
            endpoint=endpoint,
            inputs={
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0,
            },
        )
        return str(response["choices"][0]["message"]["content"])

    return _call


def agent_planner(
    feedback: FeedbackBundle,
    goal: CompiledGoal,
    agent: Agent,
    *,
    llm: PlannerLLM | None = None,
    model: str = DEFAULT_PLANNER_MODEL,
    confidence_floor: float = DEFAULT_CONFIDENCE_FLOOR,
) -> list[Decision]:
    """Lane B: compose the feedback into a prompt, call the LLM, parse fail-closed.

    A :class:`Planner`. Renders the ``(system, user)`` prompt from ``feedback`` +
    ``goal`` (:func:`build_planner_prompt`), calls the injected ``llm`` (or a default
    Databricks endpoint resolved from ``model``), and parses the response into
    decisions (:func:`parse_plan`). Raises :class:`PlanParseError` on a
    wholly-unusable response — the fail-closed contract. Builds, proves, and applies
    **nothing**: it returns :class:`Decision` objects only.

    ``agent`` is part of the :class:`Planner` contract (the planner is agent-scoped)
    and is available for future agent-specific prompting; the current prompt is
    driven entirely by the goal and feedback.
    """
    _ = agent  # agent-scoped by contract; not needed by the current prompt
    call = llm if llm is not None else _default_planner_llm(model)
    system, user = build_planner_prompt(feedback, goal)
    raw = call(system=system, user=user)
    return parse_plan(raw, confidence_floor=confidence_floor)


def _dedup_key(decision: Decision) -> tuple[str, str, str, str]:
    """Identity of a decision for A-vs-B de-duplication.

    At the decision stage there is no built change yet, so identity is the *target*
    the decision addresses: its action kind plus the trigger's judge / metric /
    asset-type detail. Two decisions with the same key (e.g. Lane A and Lane B both
    proposing a ``gepa_prompt`` for the same judge) are duplicates.
    """
    t = decision.trigger
    return (
        decision.action_kind.value,
        t.judge_name or "",
        t.metric or "",
        t.asset_type or "",
    )


def combined_decisions(
    feedback: FeedbackBundle,
    goal: CompiledGoal,
    agent: Agent,
    *,
    planner: Planner = agent_planner,
    thresholds: DecisionThresholds | None = None,
) -> CombinedDecisions:
    """Run both lanes and return the de-duped **A ∪ B** union.

    Lane A is the deterministic rule engine (:func:`ail.loop.decision_rules.decide`);
    Lane B is ``planner`` (defaults to :func:`agent_planner`). The union lists Lane A
    first, then each Lane-B decision whose :func:`_dedup_key` was not already
    contributed by A — so on a collision A (the evidence-grounded, deterministic
    lane) wins and B never displaces it. A :class:`PlanParseError` from Lane B is
    **caught** and recorded in :attr:`CombinedDecisions.planner_error` (fail-closed:
    Lane A is unaffected and no fabricated B decision is added).
    """
    a_decisions = decide(feedback, goal, thresholds=thresholds)

    planner_error: str | None = None
    b_decisions: list[Decision] = []
    try:
        b_decisions = planner(feedback, goal, agent)
    except PlanParseError as exc:
        planner_error = str(exc)

    seen: set[tuple[str, str, str, str]] = {_dedup_key(d) for d in a_decisions}
    union: list[Decision] = list(a_decisions)
    n_deduped = 0
    for d in b_decisions:
        key = _dedup_key(d)
        if key in seen:
            n_deduped += 1
            continue
        seen.add(key)
        union.append(d)

    return CombinedDecisions(
        decisions=union,
        n_from_a=len(a_decisions),
        n_from_b=len(b_decisions),
        n_deduped=n_deduped,
        planner_error=planner_error,
    )


def run_cycle_with_planner(
    agent: Agent,
    goal: CompiledGoal,
    *,
    feedback_source: FeedbackSource,
    candidate_builder: CandidateBuilder,
    prover: Prover,
    gate: Gate,
    planner: Planner = agent_planner,
    decision_thresholds: DecisionThresholds | None = None,
    now: str | None = None,
) -> PlannerCycleResult:
    """Run one layered A+B cycle: combine both lanes, then drive the proven pipeline.

    The thin wrapper the unified job uses. It gathers the feedback **once**, forms
    the de-duped A ∪ B union (:func:`combined_decisions`), and feeds that union to
    :func:`ail.loop.controller.run_cycle` via its ``decisions`` injection seam — so
    every decision, A or B, goes through the **exact same** build → prove → gate →
    propose pipeline. It does **not** reimplement or fork that pipeline; the
    controller still owns fail-closed proving, gating, and propose-only emission.
    ``feedback_source`` is passed through unchanged (run_cycle re-invokes it, per its
    contract) so the controller sees the same feedback the planner did.
    """
    feedback = feedback_source()
    plan = combined_decisions(
        feedback, goal, agent, planner=planner, thresholds=decision_thresholds
    )
    result = run_cycle(
        agent,
        goal,
        feedback_source=lambda: feedback,
        candidate_builder=candidate_builder,
        prover=prover,
        gate=gate,
        decision_thresholds=decision_thresholds,
        now=now,
        decisions=plan.decisions,
    )
    return PlannerCycleResult(result=result, plan=plan)
