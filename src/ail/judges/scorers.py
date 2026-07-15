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
  *justified*, conditioned on task success. The judge reads the full MLflow trace,
  including its recorded token usage and tool-call history, so the same assessment
  can be human-labelled and MemAlign-aligned. Deterministic L0 token/cost metrics
  remain the authoritative optimization objective; this judge adds the qualitative
  question of whether the spend was necessary.

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
    human trace labels. Every built-in judge is trace-native and alignable.
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
    description="Does the traced agent run correctly and completely accomplish the user's task?",
    feedback_value_type=Literal["yes", "no"],  # constrained categorical guardrail
    instructions=(
        "You are judging whether a complete agent run is CORRECT.\n\n"
        "Inspect the request, intermediate reasoning and tool calls, tool results, "
        "errors, validations, and final response in this MLflow trace:\n{{ trace }}\n\n"
        "A run is correct only when it actually accomplishes the user's task and its "
        "claims are supported by the observed tool results. Penalize incomplete work, "
        "ignored failures, fabricated success, incorrect code or commands, and a final "
        "answer that overstates what was verified. Ignore harmless wording differences.\n\n"
        "Answer 'yes' only when the traced evidence supports a correct and complete "
        "outcome; otherwise answer 'no'. Briefly cite the decisive trace evidence."
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
        "You are rating the MODULARITY of code produced or modified by an agent run "
        "on a 1-to-5 scale. Inspect the complete MLflow trace, including the request, "
        "file edits, tool results, tests, and final response:\n{{ trace }}\n\n"
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
        "If the run contains no code change to assess, return 3 and say so."
    ),
)

GROUNDEDNESS = ScorerSpec(
    name="groundedness",
    description="Is the response supported by the provided context, with no fabrication?",
    feedback_value_type=Literal["yes", "no"],  # constrained categorical
    instructions=(
        "You are judging whether an agent run is GROUNDED in the evidence it actually "
        "observed. Inspect the complete MLflow trace:\n{{ trace }}\n\n"
        "Every substantive claim in the final response about files, APIs, commands, "
        "tests, metrics, or outcomes must be supported by the request or by a tool "
        "result visible in the trace. A run is not grounded when it invents a path or "
        "fact, claims a command/test succeeded without evidence, or presents an "
        "assumption as an observed result.\n\n"
        "Answer 'yes' only when the final response is fully supported by traced "
        "evidence; otherwise answer 'no' and name the unsupported claim."
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
    auto_alignable=True,
    instructions=(
        "You are rating the TOKEN EFFICIENCY of an agent run on a 1-to-5 scale. "
        "Inspect the complete MLflow trace, including its request, recorded token "
        "usage, tool calls and results, errors, and final response:\n{{ trace }}\n\n"
        "Use the trace's recorded usage as the measurement; do not invent a token "
        "count. Your job is the judgement the raw number cannot make on its own:\n"
        "  - Was the token spend JUSTIFIED by what the task actually required? A "
        "large spend on a genuinely large task can still be efficient.\n"
        "  - Was the redundancy AVOIDABLE or NECESSARY? Re-reading the same file "
        "many times or re-running identical shell setup is usually avoidable "
        "waste; re-checking a file that legitimately changed is not. Use the "
        "named 'repeated_calls' targets to decide which.\n"
        "  - Is QUALITY-PER-TOKEN good — did the spend buy a correspondingly "
        "complete, correct outcome?\n\n"
        "CRITICAL — efficiency is conditioned on SUCCESS. Spending few tokens by "
        "doing less, stopping early, or producing a wrong/incomplete result is "
        "NOT efficient: judge it harshly. If the response did not accomplish the "
        "task (per the expected result / success criteria), a low token count "
        "earns a LOW score, not a high one. Reward fewer tokens only when the "
        "task was still accomplished. This scorer pairs with the correctness "
        "guardrail: tokens may fall only when quality does not.\n\n"
        "Scoring guide:\n"
        "  1 - large avoidable waste (e.g. the same target hit many times for no "
        "gain), or tokens burned without accomplishing the task\n"
        "  2 - clear avoidable waste with some useful work\n"
        "  3 - acceptable; spend roughly fits the task, minor avoidable overhead\n"
        "  4 - efficient; little avoidable redundancy, spend tracks the work\n"
        "  5 - tightly efficient; spend is well justified by the task with no "
        "meaningful avoidable waste\n\n"
        "Return the single integer (1-5) that best fits. In the rationale, NAME "
        "the specific waste you saw (which repeated target, which boilerplate, or "
        "say there was none) so the call is actionable — do not just restate the "
        "numbers."
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

    The hybrid scorer. It judges efficiency from the **L0 summary**, never the
    raw trace: feed its ``inputs`` with :func:`build_token_efficiency_inputs`,
    which packs the already-computed deterministic signals (tokens, tool-call
    count, redundancy, named repeated targets, cost, model) into a compact dict.
    The judge adds only the verdict — was the spend justified, was redundancy
    avoidable, is quality-per-token good — conditioned on task success. It is
    deliberately **not** given ``{{ trace }}``: this corpus has 900K-token
    traces that exceed a judge's context window, and the L0 facts the judge
    needs are already summarized.
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

    This is the L0→L2 bridge: it consumes the **already-computed**
    :class:`ail.metrics.contract.TraceMetrics` (the deterministic L0 metrics for
    one trace) and packs the signals the judge reasons over into a small dict.
    Nothing here re-derives L0 — every number is copied straight from ``metrics``
    — so the judge never recounts tokens or recomputes redundancy, and it scores
    off this **summary**, not the raw (possibly 900K-token) trace.

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
