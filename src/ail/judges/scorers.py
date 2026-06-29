"""L2 judged-metric scorers: thin, configurable wrappers over ``make_judge``.

This is the L2 tier of the layered metrics design (``docs/ARCHITECTURE.md`` §3):
LLM-as-judge scorers built with the public MLflow GenAI API
:func:`mlflow.genai.judges.make_judge`. The factory ships three scorers,
chosen for the Milestone-1 token-reduction lever:

* **correctness** — the key guardrail Phase 2 needs. A token-reduction
  intervention is only allowed to ship if correctness does not regress, so this
  scorer is categorical (``yes``/``no``): a clean pass/fail to threshold on.
* **modularity** — graded code-structure quality (1–5), where gradations matter.
* **groundedness** — whether the response is supported by the provided context
  (``yes``/``no``), the anti-hallucination check.

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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mlflow.genai.judges import Judge

__all__ = [
    "ScorerSpec",
    "CORRECTNESS",
    "MODULARITY",
    "GROUNDEDNESS",
    "DEFAULT_SCORERS",
    "make_scorer",
    "make_correctness_judge",
    "make_modularity_judge",
    "make_groundedness_judge",
    "with_rubric",
]


@dataclass(frozen=True, slots=True)
class ScorerSpec:
    """A reusable definition of one L2 scorer.

    ``instructions`` is the judge rubric and **must** reference at least one
    MLflow template variable (``{{ inputs }}``, ``{{ outputs }}``,
    ``{{ expectations }}``, or ``{{ trace }}``) — ``make_judge`` rejects a rubric
    that references none. ``feedback_value_type`` fixes the structured output
    type the judge returns (``bool`` / a ``Literal[...]`` of allowed labels /
    ``int`` for a graded scale).
    """

    name: str
    instructions: str
    feedback_value_type: Any
    description: str


# --- default rubrics -------------------------------------------------------
#
# Each rubric is written from scratch for this project (see PROVENANCE.md). The
# wording describes the dimension and the decision boundary; the {{ ... }}
# tokens are MLflow's template variables, filled from a judge call's
# inputs/outputs/expectations.

CORRECTNESS = ScorerSpec(
    name="correctness",
    description="Does the response correctly accomplish the task, per the expected result?",
    feedback_value_type=str,  # categorical "yes"/"no"
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
    feedback_value_type=int,  # graded 1..5
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
    feedback_value_type=str,  # categorical "yes"/"no"
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

#: The built-in scorer set, keyed by name. ``make_scorer`` and the loop
#: controller look scorers up here; a deployer extends the set by adding a
#: :class:`ScorerSpec`.
DEFAULT_SCORERS: dict[str, ScorerSpec] = {
    CORRECTNESS.name: CORRECTNESS,
    MODULARITY.name: MODULARITY,
    GROUNDEDNESS.name: GROUNDEDNESS,
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
    return make_judge(
        name=name or spec.name,
        instructions=instructions or spec.instructions,
        model=model,
        description=spec.description,
        feedback_value_type=resolved_type,
        inference_params=inference_params,
    )


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


def with_rubric(spec: ScorerSpec, instructions: str) -> ScorerSpec:
    """Return a copy of ``spec`` with a replaced rubric (convenience for tuning)."""
    return replace(spec, instructions=instructions)
