"""The configurable **review rubric** the L3 reviewer scores a trace against.

A :class:`ReviewRubric` is the *input* counterpart to the verdict contract
(:mod:`ail.l3.contract`): it fixes **which guidelines** HALO evaluates, on **what
score scale**, and **whether** it must recommend assets. The reviewer renders the
rubric into HALO's prompt (:func:`ail.l3.reviewer.build_review_prompt`) and the
parser uses it to know which guideline ids to expect and what range a score must
fall in (:func:`ail.l3.parser.parse_halo_report`).

The :data:`DEFAULT_RUBRIC` is the user's five guidelines — four scored dimensions
(tool-calling efficiency, token efficiency, tooling purpose, instruction clarity)
plus the asset-recommendation directive (guideline 5). Every guideline pulls in
the same direction as :data:`DEFAULT_OBJECTIVE`: **the same task quality with
fewer tokens and lower latency**.

This module is deliberately dependency-light (stdlib only — frozen dataclasses,
mirroring :class:`ail.cohorts.Cohort` and
:class:`ail.readiness.compute.ReadinessThresholds`), so the rubric is trivially
constructible and overridable without importing a model, MLflow, or HALO.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "DEFAULT_OBJECTIVE",
    "DEFAULT_GUIDELINES",
    "DEFAULT_RUBRIC",
    "ScoredGuideline",
    "ReviewRubric",
]

#: The standing objective every default guideline pulls toward. Baked into the
#: prompt so HALO judges and recommends against it, not against a vague notion of
#: "good".
DEFAULT_OBJECTIVE = "achieve the same task quality with fewer tokens and lower latency"


@dataclass(frozen=True, slots=True)
class ScoredGuideline:
    """One scored review dimension: a stable id, a title, and what to evaluate.

    ``id`` is the stable key the verdict's :class:`ail.l3.contract.GuidelineAssessment`
    and the attached ``rlm_<id>`` feedback assessment key off (so it must be a
    valid, terse identifier). ``description`` is the unambiguous instruction HALO
    reads — it should say exactly what to look for and which direction a higher
    score means.
    """

    id: str
    title: str
    description: str


@dataclass(frozen=True, slots=True)
class ReviewRubric:
    """A configurable set of scored guidelines plus the asset-recommendation directive.

    Args:
        rubric_id: Stable identifier recorded on the verdict so a consumer knows
            the scale and dimensions a score is on.
        guidelines: The scored dimensions (rubric guidelines 1–4 by default), in
            prompt order. Must be non-empty with unique ids.
        recommend_assets: Whether HALO must also recommend concrete assets
            (guideline 5). Defaults to ``True``.
        score_min / score_max: Inclusive integer score scale (default 1=worst ..
            5=best). The parser enforces this range on every guideline score.
        objective: The standing objective every guideline pulls toward; rendered
            into the prompt.
    """

    rubric_id: str
    guidelines: tuple[ScoredGuideline, ...]
    recommend_assets: bool = True
    score_min: int = 1
    score_max: int = 5
    objective: str = DEFAULT_OBJECTIVE

    def __post_init__(self) -> None:
        if not self.rubric_id:
            raise ValueError("ReviewRubric.rubric_id must be a non-empty string")
        if not self.guidelines:
            raise ValueError("ReviewRubric.guidelines must be non-empty")
        ids = [g.id for g in self.guidelines]
        if len(set(ids)) != len(ids):
            raise ValueError(f"ReviewRubric guideline ids must be unique; got {ids}")
        if self.score_min >= self.score_max:
            raise ValueError(
                f"ReviewRubric score_min ({self.score_min}) must be < score_max ({self.score_max})"
            )

    def guideline_ids(self) -> tuple[str, ...]:
        """The ordered guideline ids (the keys the verdict and feedback assessments use)."""
        return tuple(g.id for g in self.guidelines)

    def guideline(self, guideline_id: str) -> ScoredGuideline | None:
        """The :class:`ScoredGuideline` with ``guideline_id``, or ``None``."""
        return next((g for g in self.guidelines if g.id == guideline_id), None)

    def clamp_score(self, score: int) -> int:
        """Clamp ``score`` into the rubric's inclusive ``[score_min, score_max]`` range."""
        return max(self.score_min, min(self.score_max, score))


#: The user's four scored guidelines (guideline 5 — recommended assets — is the
#: rubric's ``recommend_assets`` directive, shaped by ``AssetRecommendation``).
#: Descriptions are written to be crisp and unambiguous (the rubric eats its own
#: dogfood on guideline 4) and to say which direction a higher score means.
DEFAULT_GUIDELINES: tuple[ScoredGuideline, ...] = (
    ScoredGuideline(
        id="tool_calling_efficiency",
        title="Tool-calling efficiency",
        description=(
            "Did the agent call tools efficiently? Flag redundant, excessive, or "
            "repeated calls: the same file read again, the same command re-run, the "
            "same target re-fetched, or calls whose results were already in context. "
            "A HIGHER score means less wasted tool work."
        ),
    ),
    ScoredGuideline(
        id="token_efficiency",
        title="Token efficiency (quality per token)",
        description=(
            "How much useful progress did the token spend buy? Flag context the "
            "agent re-loaded, verbosity that produced no new information, and "
            "back-and-forth that did not move the task forward. A HIGHER score means "
            "more useful output per token."
        ),
    ),
    ScoredGuideline(
        id="tooling_purpose",
        title="Tooling purpose",
        description=(
            "Was each tool used with a clear, correct purpose — the right tool for "
            "the goal, with a specific intent, rather than flailing or aimless "
            "exploration? Flag wrong-tool choices and thrashing. A HIGHER score "
            "means purposeful, well-targeted tool use."
        ),
    ),
    ScoredGuideline(
        id="instruction_clarity",
        title="Instruction clarity",
        description=(
            "Were the instructions the agent was given (its task prompt and system "
            "prompt, visible in the trace) clear and unambiguous? Identify where "
            "ambiguity or missing context made the agent guess, backtrack, or redo "
            "work. A HIGHER score means clearer instructions and less "
            "ambiguity-driven rework."
        ),
    ),
)

#: The default rubric: the user's five guidelines (four scored + asset
#: recommendation), scored 1 (worst) .. 5 (best).
DEFAULT_RUBRIC = ReviewRubric(
    rubric_id="ail.l3.default/v1",
    guidelines=DEFAULT_GUIDELINES,
    recommend_assets=True,
    score_min=1,
    score_max=5,
)
