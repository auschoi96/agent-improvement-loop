"""Extract N evaluation **dimensions** from a free-form requirements blob.

The multi-dimension front door. Where :func:`ail.goals.compile_goal` maps one
sentence to one objective, this maps a whole requirements paragraph to a *set* of
distinct dimensions the user cares about — "cut latency", "no hallucinated tool
calls", "responses should be concise" — each with a name, a description, and the
user's relative priority. It reuses the **exact injectable LLM seam shape** of
:class:`ail.goals.compiler.GoalProposerLLM` (a ``(system, user) -> str`` callable),
so production wires a Databricks chat endpoint and every test injects a mock — no
live model is ever called here.

**Fail-closed.** The LLM's job is only to *structure* what the user already said.
Unparseable output, a non-list, an empty list, or an item missing a name/
description all raise :class:`RequirementsExtractionError` — the extractor never
fabricates a dimension the user did not state. The routing decision (deterministic
L0 metric vs. a MemAlign judge) is deferred to :mod:`ail.requirements.composer`,
which validates the LLM's *candidate* ``metric`` against
:func:`ail.goals.allowlist.is_l0_metric` rather than trusting it.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ail.goals.allowlist import L0_OBJECTIVE_METRICS
from ail.goals.compiler import GoalProposerLLM, _strip_code_fences

__all__ = [
    "RequirementDimension",
    "RequirementsExtractionError",
    "build_extractor_system_prompt",
    "extract_dimensions",
]


class RequirementsExtractionError(Exception):
    """The requirements blob could not be turned into a set of dimensions.

    Raised for an empty/blank blob, LLM output that is not a JSON array of dimension
    objects, an empty array, or an item missing a name/description. Fail-closed: the
    extractor refuses rather than inventing a dimension the user did not state.
    """


class RequirementDimension(BaseModel):
    """One evaluation dimension the user asked for.

    Args:
        name: The dimension's short name (e.g. ``"response conciseness"``,
            ``"no hallucinated tool calls"``). Becomes the authored judge name (via
            :func:`ail.judges.authoring.normalize_judge_name`) for a quality
            dimension. Must be non-blank.
        description: What the dimension means / how to judge it — the criteria a
            judge's rubric is built from, or the human-readable intent of a
            deterministic metric. Must be non-blank.
        user_priority: The user's relative priority, ``1`` = highest. The composer
            makes the highest-priority dimension the goal's primary objective and
            the rest guardrails. Must be ``>= 1``.
        metric: The LLM's **candidate** mapping to a deterministic L0 metric name
            (one of :data:`ail.goals.allowlist.L0_OBJECTIVE_METRICS`) when the
            dimension is an exact, un-gameable measurement (latency, cost, tokens,
            tool-call count, redundancy); ``None`` for a subjective/quality
            dimension that needs a judge. It is only a *candidate* — the composer
            validates it against the real allowlist and fails closed if it names
            something that is not an L0 metric.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str
    user_priority: int = Field(ge=1)
    metric: str | None = None

    @field_validator("name", "description")
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must not be blank")
        return value.strip()

    @field_validator("metric")
    @classmethod
    def _blank_metric_is_none(cls, value: str | None) -> str | None:
        # An empty/whitespace metric string means "no deterministic mapping" — treat
        # it as None rather than a bogus metric name so the composer routes to a judge.
        if value is None or not value.strip():
            return None
        return value.strip()


def build_extractor_system_prompt() -> str:
    """Render the extractor system prompt, listing the live L0 metric allowlist.

    Grounds the LLM in the real deterministic-metric names (so a dimension it maps
    to ``metric`` is chosen from reality, not invented) while making clear that
    subjective/quality dimensions must leave ``metric`` null for a judge.
    """
    l0 = ", ".join(L0_OBJECTIVE_METRICS)
    return (
        "You extract the distinct evaluation dimensions a user cares about from their "
        "free-form requirements for an AI agent. Return ONLY a JSON array — no prose, "
        "no markdown fences.\n\n"
        "Each array element is an object with exactly these keys:\n"
        '  - "name": a short name for the dimension (e.g. "response conciseness", '
        '"no hallucinated tool calls").\n'
        '  - "description": what the dimension means and how to tell a good run from a '
        "bad one — faithful to what the user said, do not invent new requirements.\n"
        '  - "user_priority": an integer, 1 = the most important dimension, 2 = next, '
        "and so on. If the user did not rank them, infer a sensible order.\n"
        '  - "metric": if (and ONLY if) the dimension is an exact, deterministic '
        "measurement, the matching metric name from this list; otherwise null.\n"
        f"      deterministic metrics: {l0}\n\n"
        'Use a non-null "metric" only for un-gameable numeric measurements '
        "(latency => duration_seconds, cost => total_usd, token spend => total_tokens, "
        "number of tool calls => total_tool_calls, repeated/redundant work => "
        "redundancy_rate). For any subjective or quality dimension (helpfulness, "
        "correctness, conciseness, avoiding hallucinated tool calls, tone, following "
        'instructions, ...) set "metric" to null — those are judged, not measured.\n\n'
        "Extract every distinct dimension the user mentions; do not merge unrelated "
        "ones and do not add dimensions they did not ask for. If the requirements "
        "name only one dimension, return an array with one element."
    )


def _parse_dimension_array(raw: str) -> list[Any]:
    """Parse the LLM's raw text into a JSON array, failing loud on bad output.

    Mirrors :func:`ail.goals.compiler._parse_proposal_json` but expects a JSON
    *array*: strips a markdown fence, then falls back to the outermost ``[...]``
    span if the model wrapped the array in stray prose.
    """
    text = _strip_code_fences(raw)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        start, end = text.find("["), text.rfind("]")
        if start == -1 or end <= start:
            raise RequirementsExtractionError(
                f"LLM did not return a valid JSON array of dimensions: {exc}"
            ) from exc
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc2:
            raise RequirementsExtractionError(
                f"LLM did not return a valid JSON array of dimensions: {exc2}"
            ) from exc2
    if not isinstance(data, list):
        raise RequirementsExtractionError(
            f"LLM dimensions output must be a JSON array, got {type(data).__name__}."
        )
    return data


def extract_dimensions(
    nl_text: str,
    *,
    llm: GoalProposerLLM,
) -> list[RequirementDimension]:
    """Extract the distinct evaluation dimensions from a requirements blob.

    The injected ``llm`` (the :class:`ail.goals.compiler.GoalProposerLLM` seam)
    proposes a JSON array of dimension objects; each is strictly validated into a
    :class:`RequirementDimension`. Fail-closed at every step: a blank blob, output
    that is not a JSON array, an empty array, or an item with a blank name/
    description raises :class:`RequirementsExtractionError` — a dimension is never
    fabricated.

    Args:
        nl_text: The user's free-form requirements.
        llm: The proposer to use (inject a mock in tests). Unlike
            :func:`ail.goals.compile_goal` there is **no** implicit live-endpoint
            default — the caller supplies the seam, keeping the extractor offline by
            construction.

    Returns:
        The extracted dimensions, in the LLM's array order.

    Raises:
        RequirementsExtractionError: The blob is empty, or the LLM output is not a
            usable, non-empty array of well-formed dimensions.
    """
    if not nl_text or not nl_text.strip():
        raise RequirementsExtractionError("requirements text must be a non-empty string.")

    raw = llm(system=build_extractor_system_prompt(), user=nl_text)
    data = _parse_dimension_array(raw)

    if not data:
        raise RequirementsExtractionError(
            "LLM returned an empty dimensions array; refusing to fabricate a dimension "
            "(fail-closed). Provide requirements that name at least one thing to evaluate."
        )

    dimensions: list[RequirementDimension] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise RequirementsExtractionError(
                f"dimension #{i} is not a JSON object (got {type(item).__name__})."
            )
        try:
            dimensions.append(RequirementDimension.model_validate(item))
        except ValidationError as exc:
            raise RequirementsExtractionError(f"dimension #{i} is malformed: {exc}") from exc
    return dimensions
