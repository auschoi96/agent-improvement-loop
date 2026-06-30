"""Compile a natural-language optimization goal into a validated ``CompiledGoal``.

``docs/ARCHITECTURE.md`` §4: the loop starts from a natural-language goal that
``goals/compiler.py`` turns into *objective + target metric(s) + guardrails*.
This module is that step. An injectable LLM proposes the structured mapping; the
proposal is **strictly** validated against the schema and the
:mod:`ail.goals.allowlist` — a goal that names a metric the framework does not
produce fails loud (:class:`UnmappedMetricError`), never silently mis-mapped.

**Human-in-the-loop.** :func:`compile_goal` always returns
``human_confirmed=False``. An LLM-compiled goal is a *proposal*: it must be
reviewed and confirmed (:meth:`CompiledGoal.confirm`) before it drives
optimization. Nothing here promotes or runs anything — the loop controller is
responsible for refusing to act on an unconfirmed goal.

**Readiness contract.** :class:`CompiledGoal` satisfies the
:class:`ail.readiness.GoalView` Protocol (``objective_metric``,
``guardrail_names``, ``requires_quality``) so the merged readiness module can
gate it without importing this lane. The mapping onto that Protocol is the
subtle part and is enforced here, not left to convention:

* :attr:`~CompiledGoal.requires_quality` is ``True`` iff a *judge* is involved —
  the objective is itself a judged metric, or any guardrail is a judge.
* :attr:`~CompiledGoal.guardrail_names` exposes exactly the **judge** guardrail
  names, because readiness's ``judge_trusted`` gate treats every name there as a
  judge that must be measured and trusted. Deterministic *metric* guardrails are
  enforced by the comparison harness, not by readiness, so they are tracked in
  :attr:`~CompiledGoal.guardrails` but never leak into ``guardrail_names``.
* If the objective is itself a judged metric (e.g. ``correctness``), it **must**
  also appear as a judge guardrail — readiness does no objective-as-judge
  auto-detection, so a judged objective is required here to be listed among the
  guardrails or compilation fails loud (:class:`GoalContractError`).
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, model_validator

from ail.cohorts import Cohort
from ail.goals.allowlist import (
    is_judge,
    is_known_metric,
    is_l0_metric,
)

__all__ = [
    "GoalCompileError",
    "UnmappedMetricError",
    "GoalContractError",
    "GoalDirection",
    "GoalTarget",
    "Guardrail",
    "CompiledGoal",
    "GoalProposerLLM",
    "compile_goal",
]


# --- errors ---------------------------------------------------------------
#
# All three derive from Exception (NOT ValueError) so that, when raised inside a
# pydantic ``model_validator``, pydantic lets them propagate **unwrapped** rather
# than folding them into a generic ValidationError. Direct construction of a
# CompiledGoal and ``compile_goal`` therefore both surface the same typed error.
# Genuine *schema* problems (unknown field, wrong type) still raise pydantic's
# ValidationError — they are not allowlist/contract failures.


class GoalCompileError(Exception):
    """A natural-language goal could not be compiled into a valid ``CompiledGoal``."""


class UnmappedMetricError(GoalCompileError):
    """A goal references a metric/judge name that is not in the allowlist.

    The fail-loud guarantee: an NL goal mapping to an unknown metric (or a
    guardrail whose name is inconsistent with its ``kind``) is refused outright
    rather than silently invented or mis-mapped onto a real metric.
    """


class GoalContractError(GoalCompileError):
    """A goal is internally inconsistent or violates the readiness contract.

    E.g. a judged objective not also listed as a guardrail, a guardrail that
    constrains nothing, or a relative target whose sign disagrees with the
    optimization direction.
    """


GoalDirection = Literal["minimize", "maximize"]


class _GoalModel(BaseModel):
    """Base for the goal schema models: forbid unknown fields so drift is loud."""

    model_config = ConfigDict(extra="forbid")


class GoalTarget(_GoalModel):
    """The goal's target movement on the objective metric.

    Args:
        value: The target magnitude. For a ``relative`` target it is a signed
            fraction (e.g. ``-0.30`` = a 30% reduction; ``0.10`` = a 10% lift).
            For an ``absolute`` target it is the metric value to reach (e.g.
            ``0.95`` correctness, ``0.50`` USD).
        kind: Whether ``value`` is relative to the current baseline or an
            absolute level.
    """

    value: float
    kind: Literal["relative", "absolute"] = "relative"


class Guardrail(_GoalModel):
    """A check the goal must clear alongside its objective.

    Args:
        name: The metric/judge the guardrail watches. Must be in the allowlist
            and consistent with ``kind`` (an L0 metric for ``metric``, a
            registered judge for ``judge``).
        kind: ``"metric"`` for a deterministic L0 guardrail (enforced by the
            comparison harness) or ``"judge"`` for an L2 judged guardrail (whose
            trust readiness gates on).
        must_not_regress: The guardrail passes as long as the metric does not get
            worse than the baseline.
        threshold: An absolute bar the metric must satisfy (e.g. a correctness
            judge scoring ``>= 4``). ``None`` when only ``must_not_regress``
            applies.
    """

    name: str
    kind: Literal["metric", "judge"]
    must_not_regress: bool = False
    threshold: float | None = None

    @model_validator(mode="after")
    def _validate(self) -> Guardrail:
        if self.kind == "judge" and not is_judge(self.name):
            raise UnmappedMetricError(
                f"guardrail {self.name!r} has kind 'judge' but is not a registered "
                "judge; expected one of the judge names in ail.goals.allowlist."
            )
        if self.kind == "metric" and not is_l0_metric(self.name):
            raise UnmappedMetricError(
                f"guardrail {self.name!r} has kind 'metric' but is not a known L0 "
                "metric; expected one of the L0 names in ail.goals.allowlist."
            )
        if not self.must_not_regress and self.threshold is None:
            raise GoalContractError(
                f"guardrail {self.name!r} constrains nothing: set must_not_regress=True "
                "and/or a threshold."
            )
        return self


class CompiledGoal(_GoalModel):
    """A validated, structured optimization goal — the compiler's output contract.

    Construction is fully validated (schema + allowlist + the readiness contract),
    so any ``CompiledGoal`` instance — whether built by :func:`compile_goal` or
    directly — is guaranteed well-formed and accepted by
    :func:`ail.readiness.compute_readiness` as a :class:`ail.readiness.GoalView`.

    Frozen: a compiled goal is an immutable record. :meth:`confirm` returns a new
    confirmed copy rather than mutating in place, keeping the human gate explicit.

    Args:
        objective_metric: The metric to optimize. Must be in the allowlist (an L0
            metric or a judge).
        direction: ``"minimize"`` or ``"maximize"`` the objective.
        target: The target movement (relative or absolute).
        guardrails: Checks that must also hold (judge and/or deterministic-metric).
        cohort: The :class:`ail.cohorts.Cohort` this goal is scoped to, or its
            name. A goal is always bound to one cohort.
        human_confirmed: Whether a human has reviewed and confirmed this goal. The
            compiler always returns ``False``; flip it with :meth:`confirm`.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True, frozen=True)

    objective_metric: str
    direction: GoalDirection
    target: GoalTarget
    guardrails: tuple[Guardrail, ...] = ()
    cohort: Cohort | str
    human_confirmed: bool = False

    @model_validator(mode="after")
    def _validate(self) -> CompiledGoal:
        if not is_known_metric(self.objective_metric):
            raise UnmappedMetricError(
                f"objective_metric {self.objective_metric!r} is not a known metric; "
                "expected an L0 metric or a registered judge (see ail.goals.allowlist). "
                "The goal could not be mapped — refusing rather than inventing a metric."
            )

        # A judged objective must be listed among the judge guardrails: readiness
        # does no objective-as-judge auto-detection, so this is what makes its
        # judge_trusted gate require the objective's own judge.
        judge_guardrails = {g.name for g in self.guardrails if g.kind == "judge"}
        if is_judge(self.objective_metric) and self.objective_metric not in judge_guardrails:
            raise GoalContractError(
                f"judged objective {self.objective_metric!r} must also be listed as a "
                "guardrail with kind 'judge' so readiness requires that judge to be "
                "trusted; add it to guardrails."
            )

        # A signed relative target encodes the change direction, which must agree
        # with the optimization direction (negative = reduce ⇒ minimize).
        if self.target.kind == "relative":
            if self.target.value == 0:
                raise GoalContractError("relative target value must be non-zero.")
            if self.direction == "minimize" and self.target.value > 0:
                raise GoalContractError(
                    "minimize goal needs a negative relative target (a reduction), "
                    f"got {self.target.value}."
                )
            if self.direction == "maximize" and self.target.value < 0:
                raise GoalContractError(
                    "maximize goal needs a positive relative target (an increase), "
                    f"got {self.target.value}."
                )

        if isinstance(self.cohort, str) and not self.cohort:
            raise GoalContractError("cohort name must be a non-empty string.")
        return self

    # -- readiness GoalView Protocol --------------------------------------

    @property
    def requires_quality(self) -> bool:
        """Whether proving this goal needs a judged quality signal.

        ``True`` iff a judge is involved: the objective is itself a judged metric,
        or any guardrail is a judge. A pure deterministic token/cost goal is
        ``False`` and skips readiness's quality gates.
        """
        return is_judge(self.objective_metric) or any(g.kind == "judge" for g in self.guardrails)

    @property
    def guardrail_names(self) -> Sequence[str]:
        """The **judge** guardrail names readiness's judge_trusted gate requires.

        Exactly the judge-kind guardrails (the validator guarantees a judged
        objective is among them). Deterministic *metric* guardrails are
        deliberately excluded: readiness treats every name here as a judge that
        must be trusted, so a metric name would make it require a non-existent
        judge. Metric guardrails live in :attr:`guardrails`.
        """
        seen: dict[str, None] = {}
        for g in self.guardrails:
            if g.kind == "judge":
                seen.setdefault(g.name, None)
        return tuple(seen)

    # -- convenience ------------------------------------------------------

    @property
    def cohort_name(self) -> str:
        """The bound cohort's name, whether ``cohort`` is a ``Cohort`` or a string."""
        return self.cohort.name if isinstance(self.cohort, Cohort) else self.cohort

    def confirm(self) -> CompiledGoal:
        """Return a copy with ``human_confirmed=True`` (the human-in-the-loop gate).

        Until a goal is confirmed it is a proposal only and must not drive
        optimization. Returns a new instance; the original is left unconfirmed.
        """
        return self.model_copy(update={"human_confirmed": True})


# --- the injectable LLM seam ----------------------------------------------


class GoalProposerLLM(Protocol):
    """The injectable LLM seam: maps the prompt to a raw (JSON) text response.

    Any callable taking ``system``/``user`` strings and returning the model's raw
    text satisfies it — a Databricks chat endpoint in production, a canned-string
    lambda in tests. This is the single place :func:`compile_goal` touches a
    model; every test injects a mock so no live call is ever made.
    """

    def __call__(self, *, system: str, user: str) -> str: ...


_PROPOSAL_KEYS = ("objective_metric", "direction", "target", "guardrails")
_DEFAULT_ENDPOINT_ENV = "AIL_GOAL_LLM_ENDPOINT"


def _build_system_prompt() -> str:
    """Render the system prompt, listing the live allowlist so the LLM stays in-domain."""
    from ail.goals.allowlist import JUDGE_METRICS, L0_OBJECTIVE_METRICS

    l0 = ", ".join(L0_OBJECTIVE_METRICS)
    judges = ", ".join(sorted(JUDGE_METRICS))
    return (
        "You compile a natural-language agent-optimization goal into a strict JSON "
        "object. Return ONLY the JSON object — no prose, no markdown fences.\n\n"
        "The JSON object has exactly these keys:\n"
        '  - "objective_metric": the single metric to optimize.\n'
        '  - "direction": "minimize" or "maximize".\n'
        '  - "target": {"value": number, "kind": "relative"|"absolute"}. A relative '
        "value is a SIGNED fraction (-0.30 = reduce 30%, 0.10 = increase 10%); its "
        "sign must match the direction (minimize=>negative, maximize=>positive). An "
        "absolute value is the level to reach.\n"
        '  - "guardrails": a list of {"name", "kind": "metric"|"judge", '
        '"must_not_regress": bool, "threshold": number|null}. Each guardrail must '
        "set must_not_regress=true and/or a threshold.\n\n"
        "You MUST choose every name from this allowlist — never invent a metric:\n"
        f"  L0 deterministic metrics (kind 'metric'): {l0}\n"
        f"  judges / quality signals (kind 'judge'): {judges}\n\n"
        "If the objective_metric is itself a judge, you MUST also include it as a "
        "guardrail with kind 'judge'. Do NOT include a cohort or human_confirmed "
        "field. If the goal cannot be mapped to an allowlisted metric, set "
        '"objective_metric" to the closest allowlisted name only if it is a faithful '
        "match; otherwise return a metric name not in the list so compilation fails "
        "loud rather than guessing."
    )


def _strip_code_fences(text: str) -> str:
    """Strip a surrounding ``` / ```json markdown fence if the model added one."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        # drop the opening fence (``` or ```json) and a trailing fence if present
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def _parse_proposal_json(raw: str) -> dict[str, Any]:
    """Parse the LLM's raw text into a proposal dict, failing loud on bad output."""
    text = _strip_code_fences(raw)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        # fall back to the outermost {...} span in case the model added stray prose
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise GoalCompileError(f"LLM did not return valid JSON for the goal: {exc}") from exc
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc2:
            raise GoalCompileError(f"LLM did not return valid JSON for the goal: {exc2}") from exc2
    if not isinstance(data, dict):
        raise GoalCompileError(
            f"LLM goal proposal must be a JSON object, got {type(data).__name__}."
        )
    return data


def _default_databricks_proposer() -> GoalProposerLLM:
    """Build the default proposer: a Databricks chat serving endpoint.

    The single live seam. Resolves the endpoint from the ``AIL_GOAL_LLM_ENDPOINT``
    env var (no fabricated default) and calls it via MLflow's Databricks deploy
    client at temperature 0 for a deterministic structured response. Tests never
    reach this — they always inject a mock ``llm``.
    """
    endpoint = os.environ.get(_DEFAULT_ENDPOINT_ENV)
    if not endpoint:
        raise GoalCompileError(
            "no LLM provided and no default endpoint configured. Pass "
            "compile_goal(..., llm=<proposer>) or set the "
            f"{_DEFAULT_ENDPOINT_ENV} env var to a Databricks chat serving endpoint."
        )

    def _proposer(*, system: str, user: str) -> str:
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

    return _proposer


def compile_goal(
    nl_text: str,
    cohort: Cohort | str,
    *,
    llm: GoalProposerLLM | None = None,
) -> CompiledGoal:
    """Compile a natural-language goal into a validated, **unconfirmed** ``CompiledGoal``.

    The injected ``llm`` proposes the structured mapping (objective, direction,
    target, guardrails); the proposal is parsed and strictly validated against the
    schema and :mod:`ail.goals.allowlist`. The ``cohort`` is supplied by the caller
    (the LLM never chooses it), and the result always has ``human_confirmed=False``
    — call :meth:`CompiledGoal.confirm` after human review before optimizing.

    Args:
        nl_text: The natural-language optimization goal.
        cohort: The cohort to bind the goal to (a :class:`ail.cohorts.Cohort` or
            its name).
        llm: The proposer to use. Defaults to a Databricks chat endpoint resolved
            from ``AIL_GOAL_LLM_ENDPOINT``; inject a mock in tests.

    Returns:
        A validated ``CompiledGoal`` with ``human_confirmed=False``.

    Raises:
        UnmappedMetricError: The goal names a metric/judge outside the allowlist.
        GoalContractError: The goal is internally inconsistent (e.g. a judged
            objective not listed as a guardrail, a bad target sign).
        GoalCompileError: The LLM returned no usable JSON, or its proposal set a
            reserved field.
        pydantic.ValidationError: The proposal is the wrong shape (unknown field,
            wrong type).
    """
    if not nl_text or not nl_text.strip():
        raise GoalCompileError("nl_text must be a non-empty natural-language goal.")

    proposer = llm if llm is not None else _default_databricks_proposer()
    raw = proposer(system=_build_system_prompt(), user=nl_text)
    data = _parse_proposal_json(raw)

    # The LLM proposes only the metric mapping. The caller owns the cohort and the
    # human gate, so the proposal must not set either — fail loud if it tries.
    for reserved in ("cohort", "human_confirmed"):
        if reserved in data:
            raise GoalCompileError(
                f"LLM proposal must not set {reserved!r}; the caller owns the cohort "
                "and the human-confirmation gate."
            )

    return CompiledGoal.model_validate({**data, "cohort": cohort, "human_confirmed": False})
