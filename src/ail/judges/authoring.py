"""Author a MemAlign-alignable judge from a natural-language quality description.

This is the **judge-authoring** capability: a human describes, in plain language,
a quality dimension they care about ("is the final answer actually helpful?",
"did the agent follow the user's instructions?") and gets back a registered,
MemAlign-alignable LLM judge *plus* the matching label schema a human labels
against. It is the "human explains what they're looking for → we make a judge for
it" front door to the L2 layer (``docs/L2_JUDGES_CONTRACT.md``).

It is **additive**: it does not rebuild the judge scaffolding, it composes it. An
authored judge is an ordinary :class:`~ail.judges.scorers.ScorerSpec` built with
:func:`ail.judges.scorers.make_scorer` and registered through the existing
:func:`ail.judges.registration.create_aligned_scorer` path, so everything the L2
layer already does — MemAlign alignment, agreement auditing, scheduled scoring —
works on it unchanged.

Two conventions are load-bearing and are guaranteed here so the user never has to
know them:

1. **The judge is a ``{{ trace }}``-template judge.** The rubric embeds the
   ``{{ trace }}`` MLflow template variable, so MemAlign can align it from human
   feedback on real traces (:func:`ail.judges.alignment.align_judge`). Authoring
   a judge on app-computed/derived inputs instead would yield *zero* MemAlign
   training examples — the documented mistake this capability exists to avoid.
   (The one deliberate exception in this codebase is ``token_efficiency``, which
   is computed-inputs on purpose; see the module note below.)
2. **The label schema's name EXACTLY matches the judge name.**
   :func:`ail.judges.alignment.align_judge` / MemAlign pair a human's feedback to
   a judge's scores by matching the label-schema ``name`` to the judge ``name`` —
   a mismatch silently breaks alignment (the judge learns from nothing). Here both
   names derive from a single canonical :func:`normalize_judge_name`, so the
   pairing holds *by construction*.

Turning the description into a gradeable rubric is a **deterministic template**
(:func:`build_instructions`): it structures the user's criteria into a clear,
bounded rubric (a 1–5 scale or pass/fail) with a required one-line rationale.
An **optional** single LLM refinement pass (:func:`refine_criteria`, behind
``refine=True``) can sharpen vague criteria first; it is deliberately minimal.

> **Seam — adversarial refinement.** A future lane will add a Designer/Critic
> refinement loop that iteratively hardens the rubric. It plugs in *here*, by
> supplying a richer :class:`CriteriaRefiner` (or replacing
> :func:`refine_criteria`). Do not build it now — the single-pass seam is enough.

> **Seam — large traces.** A ``{{ trace }}`` judge is context-bound: it must fit
> the whole trace in the judge model's window. v1 scopes to **judge-ingestible**
> traces. For very large traces a future lane will feed the judge an RLM/HALO
> **digest** in place of the raw trace — the substitution happens at the
> trace-*feeding* boundary (whatever supplies the ``{{ trace }}`` value at score
> time), not in the rubric, so the authored judge does not need to change. The
> digest wiring is intentionally **not** built here.

MLflow is imported lazily (matching the rest of the package), so importing this
module never requires a tracking backend; only :func:`create_matching_label_schema`
and registration touch one, and only :func:`refine_criteria` (with the default
refiner) calls a model.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

from ail.judges.registration import (
    DEFAULT_SAMPLING_RATE,
    _configure_databricks,
    create_aligned_scorer,
)
from ail.judges.scorers import ScorerSpec, make_scorer

if TYPE_CHECKING:
    from mlflow.genai.judges import Judge
    from mlflow.genai.judges.base import AlignmentOptimizer
    from mlflow.genai.label_schemas import LabelSchema

    from ail.judges.registration import ScorerRegistration
    from ail.pools import AlignmentSet

__all__ = [
    "JudgeScale",
    "DEFAULT_SCALE",
    "TRACE_TEMPLATE_VAR",
    "CriteriaRefiner",
    "AuthoredJudge",
    "normalize_judge_name",
    "build_instructions",
    "build_judge_spec",
    "refine_criteria",
    "create_matching_label_schema",
    "author_judge",
]

#: The two output shapes an authored judge can take. ``"1-5"`` is a bounded graded
#: scale (gradations matter — "how helpful", "how complete"); ``"pass_fail"`` is a
#: clean categorical guardrail (a hard yes/no). Both are **constrained** structured
#: outputs, so a judge can never emit an out-of-domain value.
JudgeScale = Literal["1-5", "pass_fail"]

#: Default scale for an authored judge: a bounded 1–5 graded scale.
DEFAULT_SCALE: JudgeScale = "1-5"

#: The MLflow template variable the authored rubric embeds. Kept as a named
#: constant because it is the load-bearing token: its presence is what makes the
#: judge MemAlign-alignable (align learns from human feedback on the traces the
#: ``{{ trace }}`` judge reads). It is also the seam for the large-trace variant —
#: a future lane feeds a digest into this same slot upstream of the judge.
TRACE_TEMPLATE_VAR = "{{ trace }}"

#: Aggregations restored for a graded (1–5) authored judge. A bounded ``Literal``
#: scale loses ``make_judge``'s default mean aggregation (see
#: :mod:`ail.judges.scorers`), so a graded scorer restores meaningful roll-ups.
_GRADED_AGGREGATIONS = ("mean", "median", "p90")

#: Env var naming the Databricks chat serving endpoint used by the *default*
#: criteria refiner. No fabricated default — refinement is opt-in, and the default
#: refiner fails loud with guidance when it is unset (mirrors ``ail.goals``).
_DEFAULT_REFINER_ENDPOINT_ENV = "AIL_JUDGE_AUTHOR_LLM_ENDPOINT"


class CriteriaRefiner(Protocol):
    """The injectable LLM seam for the optional criteria-refinement pass.

    Any callable taking ``system``/``user`` strings and returning the model's raw
    text satisfies it — a Databricks chat endpoint in production
    (:func:`_default_databricks_refiner`), a canned-string lambda in tests. This is
    the single place authoring touches a model, and it is only reached when
    ``refine=True``. Mirrors ``ail.goals.compiler.GoalProposerLLM`` so the two
    NL→structure front doors share one seam shape.
    """

    def __call__(self, *, system: str, user: str) -> str: ...


@dataclass(frozen=True, slots=True)
class AuthoredJudge:
    """The result of authoring one judge: the spec, the judge, and its label schema.

    ``spec`` is the reusable :class:`~ail.judges.scorers.ScorerSpec` (the authored
    rubric + output type) — the structural artifact the rest of the L2 layer
    consumes. ``judge`` is a live, MemAlign-alignable MLflow ``Judge``: the
    **registered** judge when ``author_judge`` registered it (``registration`` set),
    otherwise the base judge built from ``spec``. ``label_schema`` is the created
    MLflow ``LabelSchema`` whose ``name`` equals ``spec.name`` (the pairing that
    lets a human's labels align this judge). ``registration`` is the
    :class:`~ail.judges.registration.ScorerRegistration` provenance when the judge
    was registered, else ``None``.
    """

    spec: ScorerSpec
    judge: Judge
    label_schema: LabelSchema
    registration: ScorerRegistration | None = None


# --- name canonicalization -------------------------------------------------

_NAME_SEP_RE = re.compile(r"[\s\-.]+")
_NAME_STRIP_RE = re.compile(r"[^a-z0-9_]+")
_NAME_COLLAPSE_RE = re.compile(r"_+")


def normalize_judge_name(name: str) -> str:
    """Canonicalize a user-supplied dimension name to a judge/label-schema name.

    The judge name and the label-schema name **must** be identical for MemAlign to
    pair human feedback to judge scores. Deriving both from this one function is
    how that pairing is guaranteed — the caller never sees, or has to match, two
    names. The canonical form is lowercase ``snake_case`` (matching the built-in
    scorers ``correctness`` / ``token_efficiency`` / …): spaces, hyphens and dots
    become underscores, other punctuation is dropped, repeats collapse, and a
    leading digit is rejected (an identifier must start with a letter).

    Raises:
        ValueError: If ``name`` is blank or normalizes to something that is not a
            valid identifier (empty, or starting with a digit).
    """
    lowered = name.strip().lower()
    if not lowered:
        raise ValueError("judge name must not be blank")
    underscored = _NAME_SEP_RE.sub("_", lowered)
    cleaned = _NAME_STRIP_RE.sub("", underscored)
    canonical = _NAME_COLLAPSE_RE.sub("_", cleaned).strip("_")
    if not canonical:
        raise ValueError(f"judge name {name!r} has no usable identifier characters")
    if canonical[0].isdigit():
        raise ValueError(
            f"judge name {name!r} normalizes to {canonical!r}, which starts with a digit; "
            "start the name with a letter (e.g. 'q1_helpfulness')."
        )
    return canonical


# --- deterministic rubric template -----------------------------------------


def _display_name(judge_name: str) -> str:
    """A human-readable rendering of a snake_case judge name (for the rubric prose)."""
    return judge_name.replace("_", " ").strip()


def build_instructions(judge_name: str, criteria: str, *, scale: JudgeScale = DEFAULT_SCALE) -> str:
    """Structure the user's ``criteria`` into a gradeable ``{{ trace }}`` rubric.

    The deterministic v1 template. It embeds :data:`TRACE_TEMPLATE_VAR` (so the
    judge is MemAlign-alignable), states the dimension and the user's criteria,
    tells the judge to read the trace and judge *only* this dimension, and appends
    a **bounded** output rubric with a **required one-line rationale** that must
    name the specific trace evidence — so a verdict is actionable, not a bare
    number. ``scale`` selects a 1–5 graded rubric or a pass/fail one.

    Raises:
        ValueError: If ``criteria`` is blank or ``scale`` is unknown.
    """
    criteria_text = criteria.strip()
    if not criteria_text:
        raise ValueError("criteria (the natural-language description) must not be blank")
    display = _display_name(judge_name)

    # NB: TRACE_TEMPLATE_VAR is concatenated as a literal, never interpolated, so
    # the ``{{ trace }}`` braces survive intact (an f-string would eat them).
    head = (
        f"You are evaluating a single agent run for one quality dimension: {display}.\n\n"
        "What this dimension means — the criteria to judge against:\n"
        f"{criteria_text}\n\n"
        "Inspect the complete agent run using its execution trace:\n" + TRACE_TEMPLATE_VAR + "\n\n"
        "Read the trace's request, intermediate steps, tool calls, and final "
        f"output, then judge how well the run satisfies the criteria above. Judge "
        f"ONLY {display}; do not reward or penalize unrelated qualities.\n\n"
    )

    if scale == "1-5":
        return head + (
            "Scoring guide (1 = worst, 5 = best):\n"
            f"  1 - does not meet the criteria; no meaningful evidence of {display}\n"
            "  2 - largely fails the criteria, with only isolated positive evidence\n"
            "  3 - partially meets the criteria; notable gaps remain\n"
            "  4 - meets the criteria with only minor gaps\n"
            f"  5 - fully and clearly meets the criteria for {display}\n\n"
            "Return the single integer (1-5) that best fits, then a one-line "
            "rationale naming the specific evidence in the trace that determined "
            "the score."
        )
    if scale == "pass_fail":
        return head + (
            f"Answer 'pass' if the run meets the criteria for {display}, or 'fail' "
            "if it does not. Then give a one-line rationale naming the specific "
            "evidence in the trace that determined the verdict."
        )
    raise ValueError(f"unknown scale {scale!r}; expected '1-5' or 'pass_fail'")


def _feedback_value_type(scale: JudgeScale) -> Any:
    """The constrained structured-output type for ``scale`` (a bounded ``Literal``)."""
    if scale == "1-5":
        return Literal[1, 2, 3, 4, 5]
    if scale == "pass_fail":
        return Literal["pass", "fail"]
    raise ValueError(f"unknown scale {scale!r}; expected '1-5' or 'pass_fail'")


def build_judge_spec(
    name: str,
    description: str,
    *,
    scale: JudgeScale = DEFAULT_SCALE,
    refine: bool = False,
    refiner: CriteriaRefiner | None = None,
    refine_model: str | None = None,
) -> ScorerSpec:
    """Build the authored :class:`~ail.judges.scorers.ScorerSpec` (offline).

    Canonicalizes the name (:func:`normalize_judge_name`), optionally sharpens the
    ``description`` with one LLM pass (:func:`refine_criteria`, only when
    ``refine=True``), then renders the deterministic ``{{ trace }}`` rubric
    (:func:`build_instructions`) into a spec with the scale's constrained output
    type and (for the graded scale) restored aggregations.

    No model is called unless ``refine=True``; building the spec is otherwise
    fully offline. The returned spec is exactly what
    :func:`ail.judges.scorers.make_scorer` and
    :func:`ail.judges.registration.create_aligned_scorer` already consume, so an
    authored judge is a first-class member of the built-in scorer set.
    """
    judge_name = normalize_judge_name(name)
    criteria = description
    if refine:
        criteria = refine_criteria(description, refiner=refiner, model=refine_model)
    instructions = build_instructions(judge_name, criteria, scale=scale)
    aggregations = _GRADED_AGGREGATIONS if scale == "1-5" else None
    return ScorerSpec(
        name=judge_name,
        description=description.strip(),
        feedback_value_type=_feedback_value_type(scale),
        instructions=instructions,
        aggregations=aggregations,
    )


# --- optional LLM refinement (single pass, behind a flag) ------------------

_REFINER_SYSTEM_PROMPT = (
    "You sharpen a vague description of an LLM-judge quality dimension into crisp, "
    "concrete, gradeable criteria. Return ONLY the improved criteria as short prose "
    "or a few bullet points describing what a good vs. poor run looks like for this "
    "dimension. Do NOT add a numeric scale, a pass/fail verdict, or any mention of "
    "the trace — those are added separately. Keep it faithful to the original intent; "
    "do not invent unrelated requirements."
)


def refine_criteria(
    description: str,
    *,
    refiner: CriteriaRefiner | None = None,
    model: str | None = None,
) -> str:
    """Sharpen a vague ``description`` into gradeable criteria with one LLM pass.

    The optional refinement step. Calls ``refiner`` (or the default Databricks
    endpoint refiner when ``refiner`` is ``None``) exactly once and returns its
    text, falling back to the original ``description`` if the model returns empty.
    This is a single, simple pass on purpose — the adversarial Designer/Critic loop
    is a future lane that plugs into this same seam.

    Args:
        description: The user's natural-language criteria to sharpen.
        refiner: The injectable LLM seam. ``None`` builds the default Databricks
            endpoint refiner (which requires ``model`` or the
            ``AIL_JUDGE_AUTHOR_LLM_ENDPOINT`` env var).
        model: Databricks chat serving **endpoint name** for the default refiner
            (ignored when ``refiner`` is supplied). ``None`` falls back to the env
            var.

    Raises:
        ValueError: If ``refiner`` is ``None`` and no endpoint is configured.
    """
    call = refiner if refiner is not None else _default_databricks_refiner(model)
    raw = call(system=_REFINER_SYSTEM_PROMPT, user=description.strip())
    refined = raw.strip()
    # A model that returns nothing usable must not silently blank the criteria; keep
    # the human's original description rather than author an empty rubric.
    return refined or description.strip()


def _default_databricks_refiner(model: str | None) -> CriteriaRefiner:
    """Build the default refiner: a Databricks chat serving endpoint at temperature 0.

    The single live seam for refinement, mirroring
    ``ail.goals.compiler._default_databricks_proposer``. Resolves the endpoint from
    ``model`` or the ``AIL_JUDGE_AUTHOR_LLM_ENDPOINT`` env var (no fabricated
    default). Tests never reach this — they inject a mock ``refiner``.
    """
    endpoint = model or os.environ.get(_DEFAULT_REFINER_ENDPOINT_ENV)
    if not endpoint:
        raise ValueError(
            "refine=True but no refiner and no endpoint configured. Pass "
            "author_judge(..., refiner=<callable>) / refine_model=<endpoint>, or set "
            f"the {_DEFAULT_REFINER_ENDPOINT_ENV} env var to a Databricks chat serving "
            "endpoint."
        )

    def _refiner(*, system: str, user: str) -> str:
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

    return _refiner


# --- the matching label schema (name == judge name) ------------------------


def _label_input(scale: JudgeScale) -> Any:
    """The MLflow label-schema input control matching the judge's output scale."""
    from mlflow.genai.label_schemas import InputNumeric, InputPassFail

    if scale == "1-5":
        return InputNumeric(min_value=1, max_value=5)
    if scale == "pass_fail":
        return InputPassFail(positive_label="pass", negative_label="fail")
    raise ValueError(f"unknown scale {scale!r}; expected '1-5' or 'pass_fail'")


def _default_label_instruction(display: str, scale: JudgeScale) -> str:
    """The prompt a human sees when labeling a trace for this judge."""
    verdict = "a score from 1 (worst) to 5 (best)" if scale == "1-5" else "'pass' or 'fail'"
    return (
        f"Judge this run for {display}. Give {verdict}, and use the comment box for a "
        "one-line rationale naming the specific evidence in the trace."
    )


def create_matching_label_schema(
    judge_name: str,
    *,
    scale: JudgeScale = DEFAULT_SCALE,
    instruction: str | None = None,
    title: str | None = None,
    enable_comment: bool = True,
    overwrite: bool = False,
    experiment_id: str | None = None,
) -> LabelSchema:
    """Create the ``feedback`` label schema whose ``name`` **equals** ``judge_name``.

    This is the load-bearing pairing: ``mlflow.genai.label_schemas.create_label_schema``
    is called with ``name=judge_name`` and ``type='feedback'`` so a human's labels
    recorded under this schema align the judge of the same name (MemAlign pairs by
    name; a mismatch silently breaks alignment). The input control matches the
    judge's output — :class:`~mlflow.genai.label_schemas.InputNumeric` ``1..5`` for
    the graded scale, :class:`~mlflow.genai.label_schemas.InputPassFail` for
    pass/fail — and ``enable_comment`` is on so the human can record the same
    one-line rationale the judge is asked to produce.

    Assumes MLflow is pointed at the tracking backend (a Databricks notebook, or a
    configured workspace). Pass ``overwrite=True`` to replace an existing schema of
    this name.
    """
    import mlflow
    from mlflow.genai.label_schemas import InputCategorical, create_label_schema
    from mlflow.utils.databricks_utils import is_databricks_uri

    display = _display_name(judge_name)
    label_input = _label_input(scale)
    schema_experiment_id = experiment_id
    if is_databricks_uri(mlflow.get_tracking_uri()):
        schema_experiment_id = None
        if scale == "pass_fail":
            label_input = InputCategorical(options=["pass", "fail"])
    return create_label_schema(
        name=judge_name,
        type="feedback",
        input=label_input,
        instruction=instruction or _default_label_instruction(display, scale),
        title=title or display,
        enable_comment=enable_comment,
        overwrite=overwrite,
        experiment_id=schema_experiment_id,
    )


# --- the front door --------------------------------------------------------


def author_judge(
    name: str,
    description: str,
    *,
    experiment_id: str,
    scale: JudgeScale = DEFAULT_SCALE,
    register: bool = True,
    model: str | None = None,
    sampling_rate: float = DEFAULT_SAMPLING_RATE,
    alignment_set: AlignmentSet | None = None,
    optimizer: AlignmentOptimizer | None = None,
    filter_string: str | None = None,
    refine: bool = False,
    refiner: CriteriaRefiner | None = None,
    refine_model: str | None = None,
    label_instruction: str | None = None,
    overwrite_label_schema: bool = False,
    profile: str | None = None,
) -> AuthoredJudge:
    """Turn a natural-language quality description into a registered, alignable judge.

    The front door of the capability. It:

    1. Canonicalizes ``name`` and builds the ``{{ trace }}``-template
       :class:`~ail.judges.scorers.ScorerSpec` (:func:`build_judge_spec`),
       optionally sharpening ``description`` with one LLM pass (``refine=True``).
    2. Creates the **matching** label schema (:func:`create_matching_label_schema`)
       whose name equals the judge name — the MemAlign pairing, guaranteed here.
    3. Registers the judge through the existing MemAlign-aware path
       (:func:`ail.judges.registration.create_aligned_scorer`) when ``register`` is
       set: with a non-empty ``alignment_set`` it is aligned before registration,
       otherwise the base judge is registered and flagged ``aligned=false`` (not yet
       trusted — the standard L2 state until labels exist). With ``register=False``
       it just builds the base judge (no ``databricks-agents`` needed), so a caller
       can preview the authored judge and its schema before scheduling scoring.

    The label schema is always created (it is the human-labeling target that makes
    the judge alignable), independent of ``register``.

    Args:
        name: The quality dimension's name (canonicalized to snake_case).
        description: The natural-language description of what to judge.
        experiment_id: Target MLflow experiment (label schema + registration).
        scale: ``"1-5"`` (graded, default) or ``"pass_fail"`` (categorical).
        register: Register the judge as a scheduled scorer (default ``True``).
        model: Judge model URI (``None`` → MLflow's default judge model).
        sampling_rate / alignment_set / optimizer / filter_string / profile:
            Forwarded to :func:`~ail.judges.registration.create_aligned_scorer`.
        refine / refiner / refine_model: The optional single LLM refinement pass.
        label_instruction: Override the human-labeling prompt on the schema.
        overwrite_label_schema: Replace an existing label schema of this name.

    Returns:
        An :class:`AuthoredJudge` with the spec, the (registered or base) judge, the
        matching label schema, and the registration provenance (``None`` when not
        registered).
    """
    spec = build_judge_spec(
        name,
        description,
        scale=scale,
        refine=refine,
        refiner=refiner,
        refine_model=refine_model,
    )
    # Point MLflow at the Databricks-managed backend (mirroring the registration
    # seam) BEFORE creating the label schema, so the schema write lands on the
    # right workspace regardless of caller (CLI, library) — not only inside a
    # notebook where MLflow is preconfigured. Idempotent with create_aligned_scorer,
    # which applies the same config; needs only mlflow, never databricks-agents.
    _configure_databricks(profile=profile, tracking_uri="databricks", registry_uri="databricks-uc")
    label_schema = create_matching_label_schema(
        spec.name,
        scale=scale,
        instruction=label_instruction,
        overwrite=overwrite_label_schema,
        experiment_id=experiment_id,
    )
    registration: ScorerRegistration | None = None
    if register:
        registration = create_aligned_scorer(
            spec,
            experiment_id=experiment_id,
            alignment_set=alignment_set,
            optimizer=optimizer,
            model=model,
            sampling_rate=sampling_rate,
            filter_string=filter_string,
            profile=profile,
        )
        judge = registration.judge
    else:
        judge = make_scorer(spec, model=model)
    return AuthoredJudge(
        spec=spec, judge=judge, label_schema=label_schema, registration=registration
    )
