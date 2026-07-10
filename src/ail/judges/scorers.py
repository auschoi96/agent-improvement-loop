"""L2 judged-metric scorers: thin, configurable wrappers over ``make_judge``.

This is the L2 tier of the layered metrics design (``docs/ARCHITECTURE.md`` §3):
LLM-as-judge scorers built with the public MLflow GenAI API
:func:`mlflow.genai.judges.make_judge`. The factory ships four scorers,
chosen for the Milestone-1 token-reduction lever:

* **correctness** — the key guardrail Phase 2 needs. A token-reduction
  intervention is only allowed to ship if correctness does not regress, so this
  scorer is categorical (``yes``/``no``): a clean pass/fail to threshold on.
* **modularity** — graded code-structure quality (1–5), where gradations matter.
* **groundedness** — whether the response is supported by the provided context
  (``yes``/``no``), the anti-hallucination check.
* **token_efficiency** — graded (1–5) judgement of whether the token spend was
  *justified* for the work accomplished. This is a **``{{ trace }}``-based,
  MemAlign-alignable** judge: it reads the run's trace directly (the tool-call
  sequence, redundant/repeated reads, boilerplate re-runs, output verbosity,
  spend vs. work) and — like every other judge — the scheduled auto-align
  cadence (:func:`ail.judges.auto_align.auto_align_scorers`) re-aligns it from
  human trace labels. It **complements**, and does not replace, the
  deterministic L0 layer (:mod:`ail.metrics`): L0 owns the un-gameable
  token/cost *count*; the judge adds the subjective call L0 cannot make — was
  that spend justified, was the redundancy avoidable, is quality-per-token
  good — conditioned on success read from the trace. As a ``{{ trace }}`` judge
  it is context-bound: it aligns and scores on judge-ingestible traces, and the
  heavy tail (~900K-token traces) needs the not-yet-built digest-fed-judge seam
  (see :func:`make_token_efficiency_judge`). It is the Phase-2 partner of
  ``correctness``: tokens may fall only if quality does not.

Each scorer is a :class:`ScorerSpec` (name + instructions/rubric + output type)
that the factory turns into an MLflow ``Judge``. Instructions and the feedback
value type are overridable per call, so a deployer can tune a rubric without
touching this module. The returned judge is a standard MLflow ``Judge``: call it
with ``inputs``/``outputs``/``expectations`` to get a structured score, and pass
it to :func:`ail.judges.alignment.align_judge` to align it.

Design notes:

* **No model is called here.** ``make_judge`` only *constructs* a judge; the
  judge calls its model lazily on ``__call__``. So building a scorer is offline
  and free, and tests construct real judges with no network.
* **MLflow imported lazily**, matching :mod:`ail.ingest.mlflow_source`, so
  importing this package never pulls the judge runtime until a scorer is built.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from mlflow.genai.judges import Judge

    from ail.metrics.contract import TraceMetrics

__all__ = [
    "ScorerSpec",
    "CORRECTNESS",
    "MODULARITY",
    "GROUNDEDNESS",
    "TOKEN_EFFICIENCY",
    "DEFAULT_SCORERS",
    "make_scorer",
    "make_correctness_judge",
    "make_modularity_judge",
    "make_groundedness_judge",
    "make_token_efficiency_judge",
    "build_token_efficiency_inputs",
    "with_rubric",
]


@dataclass(frozen=True, slots=True)
class ScorerSpec:
    """A reusable definition of one L2 scorer.

    ``instructions`` is the judge rubric and **must** reference at least one
    MLflow template variable (``{{ inputs }}``, ``{{ outputs }}``,
    ``{{ expectations }}``, or ``{{ trace }}``) — ``make_judge`` rejects a rubric
    that references none. ``feedback_value_type`` fixes the **constrained**
    structured output the judge returns: a ``Literal[...]`` of the allowed labels
    (categorical) or a ``Literal[...]`` of the allowed integers (a bounded graded
    scale) so the judge can never emit an out-of-domain value.

    ``aggregations`` overrides MLflow's default cross-trace aggregation for the
    registered scorer. ``make_judge`` only auto-aggregates a bare ``int``/
    ``float``/``bool`` (``["mean"]``); a bounded ``Literal[...]`` scale gets none
    by default, so a graded spec restores meaningful aggregations explicitly.
    ``None`` keeps MLflow's default.

    ``auto_alignable`` marks whether the scheduled MemAlign auto-align cadence
    (:func:`ail.judges.auto_align.auto_align_scorers`) may align this judge from
    human trace labels. Every built-in judge is (the default ``True``): each is a
    ``{{ trace }}`` judge that grades a real run, and a human can label that run.
    Setting this ``False`` is the generic, name-free opt-out the cadence honours —
    for a judge that cannot be aligned from trace labels (e.g. one fed only
    app-computed inputs and ``{{ expectations }}`` the trace-label align path
    cannot supply) — so it skips such a judge cleanly instead of recording a
    spurious failure every run. No built-in scorer currently sets it ``False``.
    """

    name: str
    instructions: str
    feedback_value_type: Any
    description: str
    aggregations: tuple[str, ...] | None = None
    auto_alignable: bool = True


# --- default rubrics -------------------------------------------------------
#
# Each rubric is written from scratch for this project (see PROVENANCE.md). The
# wording describes the dimension and the decision boundary; the {{ ... }}
# tokens are MLflow's template variables, filled from a judge call's
# inputs/outputs/expectations.

CORRECTNESS = ScorerSpec(
    name="correctness",
    description="Does the response correctly accomplish the task, per the expected result?",
    feedback_value_type=Literal["yes", "no"],  # constrained categorical guardrail
    instructions=(
        "You are judging whether an agent's response is CORRECT.\n\n"
        "Task / request:\n{{ inputs }}\n\n"
        "Agent response to judge:\n{{ outputs }}\n\n"
        "Expected result (ground truth):\n{{ expectations }}\n\n"
        "A response is correct when it accomplishes the task and is consistent "
        "with the expected result: the substantive facts, conclusions, and any "
        "code or commands match what is expected. Ignore differences in wording, "
        "formatting, or ordering that do not change the outcome. If the expected "
        "result is empty, judge whether the response correctly and completely "
        "satisfies the task on its own terms.\n\n"
        "Answer 'yes' if the response is correct, or 'no' if it is wrong, "
        "incomplete, or contradicts the expected result. Briefly justify the call."
    ),
)

MODULARITY = ScorerSpec(
    name="modularity",
    description="How modular and well-structured is the produced code (1=poor, 5=excellent)?",
    feedback_value_type=Literal[1, 2, 3, 4, 5],  # bounded graded scale
    # A bounded Literal scale loses make_judge's default mean aggregation, so
    # restore aggregations meaningful for a graded metric (Phase-4 leaderboard).
    aggregations=("mean", "median", "p90"),
    instructions=(
        "You are rating the MODULARITY of the code in an agent's response on a "
        "1-to-5 scale.\n\n"
        "Task / request:\n{{ inputs }}\n\n"
        "Agent response to judge:\n{{ outputs }}\n\n"
        "Modular code has clear separation of concerns, small focused functions "
        "with single responsibilities, low coupling and high cohesion, reuse of "
        "existing helpers instead of duplication, and sensible names. Penalize "
        "god-functions, copy-pasted logic, leaked abstractions, and tangled "
        "dependencies.\n\n"
        "Scoring guide:\n"
        "  1 - monolithic / heavily duplicated, no separation of concerns\n"
        "  2 - mostly tangled with isolated good parts\n"
        "  3 - workable structure with notable coupling or duplication\n"
        "  4 - clean separation with minor issues\n"
        "  5 - exemplary modularity, cohesive and reusable\n\n"
        "Return the single integer (1-5) that best fits, and briefly justify it. "
        "If the response contains no code to assess, return 3 and say so."
    ),
)

GROUNDEDNESS = ScorerSpec(
    name="groundedness",
    description="Is the response supported by the provided context, with no fabrication?",
    feedback_value_type=Literal["yes", "no"],  # constrained categorical
    instructions=(
        "You are judging whether an agent's response is GROUNDED in the context "
        "it was given.\n\n"
        "Provided context / request (the only support the response may rely on):\n"
        "{{ inputs }}\n\n"
        "Agent response to judge:\n{{ outputs }}\n\n"
        "A response is grounded when every substantive claim, file path, symbol, "
        "API, figure, or quotation in it is supported by the provided context. A "
        "response is NOT grounded if it invents facts, cites sources or paths not "
        "present in the context, or states specifics that the context does not "
        "support — even if they sound plausible. General knowledge that does not "
        "contradict the context is acceptable.\n\n"
        "Answer 'yes' if the response is fully grounded, or 'no' if any part is "
        "unsupported or fabricated. Briefly justify the call, naming the "
        "ungrounded claim if there is one."
    ),
)

TOKEN_EFFICIENCY = ScorerSpec(
    name="token_efficiency",
    description=(
        "Was the token spend justified for the task, conditioned on success "
        "(1=wasteful, 5=tightly efficient)?"
    ),
    feedback_value_type=Literal[1, 2, 3, 4, 5],  # bounded graded scale
    # As with modularity, a bounded Literal loses make_judge's default mean
    # aggregation; restore aggregations meaningful for a graded metric.
    aggregations=("mean", "median", "p90"),
    # token_efficiency is a {{ trace }}-based, MemAlign-alignable judge (auto_alignable
    # defaults True): the ail-auto-align cadence includes it and re-aligns it from human
    # trace labels like every other judge. It complements the deterministic L0 layer —
    # ail.metrics owns the un-gameable token/cost count; the judge learns the subjective
    # "was that spend justified / was the redundancy avoidable" call from those labels.
    instructions=(
        "You are rating the TOKEN EFFICIENCY of an agent run on a 1-to-5 scale, "
        "judging it directly FROM THE TRACE of the run.\n\n"
        "Full execution trace to judge:\n{{ trace }}\n\n"
        "Read the trace end to end — the sequence of tool calls, the model turns, "
        "and the final response — and decide whether the tokens the run spent were "
        "justified by the work it actually accomplished. Judge whether the run "
        "SUCCEEDED from the trace itself: does it reach a complete, coherent "
        "outcome for the task it was given, or does it stall, loop, or give up? Do "
        "not assume an external ground-truth answer — read success or failure off "
        "the trace.\n\n"
        "Weigh the concrete efficiency signals the trace exposes:\n"
        "  - REDUNDANT WORK — the same file read again and again, identical "
        "searches or shell setup re-run, the same context re-fetched when nothing "
        "changed to justify it. Re-checking something that legitimately changed is "
        "not waste; re-deriving an unchanged fact is.\n"
        "  - BOILERPLATE / RE-RUNS — repeated scaffolding, re-issuing a command "
        "that already succeeded, re-loading state the run already had.\n"
        "  - OUTPUT VERBOSITY — long, padded, or repetitive model output that "
        "spends tokens without adding information the task needed.\n"
        "  - SPEND vs. WORK — a large spend on a genuinely large task can still be "
        "efficient; a large spend that bought only a small or incomplete result is "
        "not. Judge QUALITY-PER-TOKEN: did the spend buy a correspondingly "
        "complete outcome?\n\n"
        "CRITICAL — efficiency is conditioned on SUCCESS, read from the trace. "
        "Spending few tokens by doing less, stopping early, or leaving the task "
        "unfinished or wrong is NOT efficient: score it LOW, not high. Reward "
        "fewer tokens only when the trace shows the task was still accomplished. "
        "This scorer pairs with the correctness guardrail: tokens may fall only "
        "when quality does not.\n\n"
        "Scoring guide:\n"
        "  1 - large avoidable waste (e.g. the same target hit many times for no "
        "gain), or tokens burned without accomplishing the task\n"
        "  2 - clear avoidable waste with some useful work\n"
        "  3 - acceptable; spend roughly fits the task, minor avoidable overhead\n"
        "  4 - efficient; little avoidable redundancy, spend tracks the work\n"
        "  5 - tightly efficient; spend is well justified by the task with no "
        "meaningful avoidable waste\n\n"
        "Return the single integer (1-5) that best fits. In the rationale, NAME "
        "the specific waste you saw in the trace (which repeated read, which "
        "re-run command, which verbose passage — or say there was none) so the "
        "call is actionable — do not just restate token counts."
    ),
)

#: The built-in scorer set, keyed by name. ``make_scorer`` and the loop
#: controller look scorers up here; a deployer extends the set by adding a
#: :class:`ScorerSpec`.
DEFAULT_SCORERS: dict[str, ScorerSpec] = {
    CORRECTNESS.name: CORRECTNESS,
    MODULARITY.name: MODULARITY,
    GROUNDEDNESS.name: GROUNDEDNESS,
    TOKEN_EFFICIENCY.name: TOKEN_EFFICIENCY,
}

# Sentinel distinguishing "feedback_value_type not overridden" from an explicit
# ``None`` (which make_judge treats as "let the judge decide its own type").
_UNSET: Any = object()


def make_scorer(
    spec: ScorerSpec,
    *,
    model: str | None = None,
    instructions: str | None = None,
    feedback_value_type: Any = _UNSET,
    name: str | None = None,
    inference_params: dict[str, Any] | None = None,
) -> Judge:
    """Build an MLflow ``Judge`` from a :class:`ScorerSpec`.

    A thin wrapper over :func:`mlflow.genai.judges.make_judge`: it applies the
    spec's defaults and lets a caller override the rubric, output type, name, or
    model without redefining the spec.

    Args:
        spec: The scorer definition (one of :data:`DEFAULT_SCORERS` or a custom
            one).
        model: Judge model URI (e.g. ``"databricks:/..."`` or
            ``"openai:/gpt-4.1-mini"``). ``None`` uses MLflow's default judge
            model for the active tracking backend (Databricks-managed by default
            for this project).
        instructions: Override the spec's rubric. Must still reference at least
            one ``{{ ... }}`` template variable.
        feedback_value_type: Override the spec's structured output type. Left
            unset, the spec's type is used.
        name: Override the judge name (defaults to the spec name).
        inference_params: Optional model inference params (e.g.
            ``{"temperature": 0.0}`` for reproducible scoring).

    Returns:
        A configured MLflow ``Judge`` ready to call or to align.
    """
    from mlflow.genai.judges import make_judge

    resolved_type = (
        spec.feedback_value_type if feedback_value_type is _UNSET else feedback_value_type
    )
    judge = make_judge(
        name=name or spec.name,
        instructions=instructions or spec.instructions,
        model=model,
        description=spec.description,
        feedback_value_type=resolved_type,
        inference_params=inference_params,
    )
    # ``make_judge`` only auto-aggregates a bare numeric type; a bounded Literal
    # scale gets none. Restore the spec's aggregations so a graded scorer still
    # rolls up across traces (the value type stays the constrained Literal).
    if spec.aggregations is not None:
        judge = judge.model_copy(update={"aggregations": list(spec.aggregations)})
    return judge


def make_correctness_judge(
    *,
    model: str | None = None,
    instructions: str | None = None,
    inference_params: dict[str, Any] | None = None,
) -> Judge:
    """Build the **correctness** guardrail judge (categorical ``yes``/``no``)."""
    return make_scorer(
        CORRECTNESS, model=model, instructions=instructions, inference_params=inference_params
    )


def make_modularity_judge(
    *,
    model: str | None = None,
    instructions: str | None = None,
    inference_params: dict[str, Any] | None = None,
) -> Judge:
    """Build the **modularity** judge (graded 1–5)."""
    return make_scorer(
        MODULARITY, model=model, instructions=instructions, inference_params=inference_params
    )


def make_groundedness_judge(
    *,
    model: str | None = None,
    instructions: str | None = None,
    inference_params: dict[str, Any] | None = None,
) -> Judge:
    """Build the **groundedness** judge (categorical ``yes``/``no``)."""
    return make_scorer(
        GROUNDEDNESS, model=model, instructions=instructions, inference_params=inference_params
    )


def make_token_efficiency_judge(
    *,
    model: str | None = None,
    instructions: str | None = None,
    inference_params: dict[str, Any] | None = None,
) -> Judge:
    """Build the **token-efficiency** judge (graded 1–5).

    A ``{{ trace }}``-based, MemAlign-alignable judge: it reads the run's trace
    directly and grades quality-per-token — was the spend justified, was the
    redundancy avoidable, did the run actually finish the job — judging success
    *from the trace*, not from a ground-truth expectation. It **complements** the
    deterministic L0 layer (:mod:`ail.metrics`), which owns the un-gameable
    token/cost count; the judge adds the subjective call L0 cannot make, and the
    scheduled auto-align cadence re-aligns it from human labels like every other
    judge.

    Large-trace caveat (honest scope): as a ``{{ trace }}`` judge it is
    context-bound — it can align and score only on **judge-ingestible** traces.
    The corpus's heavy tail (~900K-token traces, ``docs/ARCHITECTURE.md`` §8)
    exceeds a judge's context window; those are **not** covered here and need the
    not-yet-built digest-fed-judge seam (a digest supplied at the trace-feeding
    boundary, not a rubric change). This does not fake coverage of the big
    traces. The retained :func:`build_token_efficiency_inputs` is an independent
    L0-summary helper (used by the labeling workflow and tests), not this judge's
    input path.
    """
    return make_scorer(
        TOKEN_EFFICIENCY, model=model, instructions=instructions, inference_params=inference_params
    )


#: Cap on how many named repeated-target identities flow into the judge input.
#: The full L0 ``repeated_calls`` list can be long; the judge only needs the
#: worst offenders to name the waste, and a compact input keeps the judge call
#: well inside its context window (the large-trace-safety contract).
_TOP_REPEATS_FOR_JUDGE = 8


def build_token_efficiency_inputs(
    metrics: TraceMetrics,
    *,
    task: Any = None,
) -> dict[str, Any]:
    """Build the token-efficiency judge's ``inputs`` from an L0 record.

    This is an L0→dict bridge: it consumes the **already-computed**
    :class:`ail.metrics.contract.TraceMetrics` (the deterministic L0 metrics for
    one trace) and packs those signals into a small dict. Nothing here re-derives
    L0 — every number is copied straight from ``metrics`` — so any consumer of
    this **summary** reads facts rather than recounting them from the raw
    (possibly 900K-token) trace. NOTE: the default ``token_efficiency`` judge is
    now ``{{ trace }}``-based and does **not** read this; the helper is retained
    for the labeling workflow and tests.

    Args:
        metrics: The per-trace L0 metrics (from
            :func:`ail.metrics.l0_deterministic.compute_trace_metrics`).
        task: Optional task/request description for the run, so the judge can
            decide whether the spend was justified *for that task*. Passed
            through verbatim under ``"task"``.

    Returns:
        A JSON-serializable ``inputs`` dict with a ``"task"`` field and an
        ``"l0_signals"`` block (tokens, tool calls, redundancy with the top
        named repeated targets, cost, model, duration).
    """
    redundancy = metrics.redundancy
    repeated = [
        {
            "tool": r.tool,
            "identity": r.identity,
            "count": r.count,
            "kind": r.signature_kind,
        }
        for r in redundancy.repeated_calls[:_TOP_REPEATS_FOR_JUDGE]
    ]
    l0_signals: dict[str, Any] = {
        "model": metrics.model,
        "total_tokens": metrics.tokens.total_tokens,
        "input_tokens": metrics.tokens.input_tokens,
        "output_tokens": metrics.tokens.output_tokens,
        "cache_total_tokens": metrics.tokens.cache_total_tokens,
        "total_tool_calls": metrics.total_tool_calls,
        "redundancy_rate": redundancy.redundancy_rate,
        "redundant_tool_calls": redundancy.redundant_tool_calls,
        "repeated_calls": repeated,
        "duration_seconds": metrics.duration_seconds,
        "cost_usd": metrics.cost.total_usd if metrics.cost.priced else None,
        "cost_priced": metrics.cost.priced,
    }
    return {"task": task, "l0_signals": l0_signals}


def with_rubric(spec: ScorerSpec, instructions: str) -> ScorerSpec:
    """Return a copy of ``spec`` with a replaced rubric (convenience for tuning)."""
    return replace(spec, instructions=instructions)
