"""The candidate-vs-baseline comparison **output contract** — typed, versioned.

Like the L0 (:mod:`ail.metrics.contract`) and L2 (:mod:`ail.judges.contract`)
contracts, these are the stable shapes a downstream consumer (the loop
controller's human gate, the Phase-4 leaderboard's per-intervention attribution)
reads. They are pydantic v2 models that round-trip through JSON
(``model_dump_json`` / ``model_validate_json``) without custom serialization and
forbid unknown fields so drift is loud.

The headline artifact is :class:`ComparisonResult`: for one frozen Task-Suite
task, the baseline-vs-candidate value of every L0 metric, the absolute and
percentage delta of each, the pass/fail of every guardrail with its reason, and
a single :class:`Recommendation` — ``PROMOTE`` only when the objective (a
deterministic L0 token/cost reduction) is met **and** no guardrail regressed.

The guardrail is the **anti-co-adaptation gate** (``docs/ARCHITECTURE.md`` §2):
the objective is the deterministic, un-gameable L0 reduction, but a candidate is
blocked if a correctness guardrail regresses. See :data:`INTERIM_JUDGE_NOTE` for
the explicit interim-vs-aligned-judge stance recorded on every guardrail.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from ail.judges.pools import ScoreValue

#: Version of the comparison output contract. Bump the minor for additive,
#: backward-compatible fields; the major for breaking shape changes.
SCHEMA_VERSION = "ail.compare/v1"

#: The interim-vs-aligned-judge stance, recorded verbatim on every correctness
#: :class:`GuardrailCheck` and surfaced to a reader. The guardrail uses the
#: **base** correctness judge (plus any available L1 programmatic signal) TODAY,
#: because the reference experiment has zero human labels and MemAlign has
#: nothing to align against (``docs/ARCHITECTURE.md`` §8). This is honest, not
#: aligned: it is **not** judge-vs-human calibrated, so a deployer must read it as
#: a provisional gate. It switches to the MemAlign-ALIGNED judge — audited against
#: the Human Anchor with a configurable agreement floor
#: (:mod:`ail.judges.agreement`) — once human labels exist. We do not fake
#: alignment by pretending an unaligned judge is trustworthy.
INTERIM_JUDGE_NOTE = (
    "INTERIM guardrail: scored by the BASE correctness judge (not MemAlign-aligned, "
    "not judge-vs-human calibrated) because no human labels exist yet. Switches to "
    "the MemAlign-aligned, Human-Anchor-audited judge once labels exist; until then "
    "treat this as a provisional gate, not a calibrated verdict."
)


class _Contract(BaseModel):
    """Base for every contract model: forbid unknown fields so drift is loud."""

    model_config = ConfigDict(extra="forbid")


class Recommendation(StrEnum):
    """The overall ship/no-ship verdict for a candidate.

    Deliberately binary. ``BLOCK`` covers both failure modes — a regressed
    guardrail *and* an unmet objective (no token/cost reduction) — so a caller
    that only reads this field can never mistake "changed nothing" or "got worse"
    for a promotion.
    """

    PROMOTE = "promote"
    BLOCK = "block"


class MetricDelta(_Contract):
    """Baseline vs candidate for one scalar L0 metric, with absolute + % delta.

    ``delta_absolute`` is ``candidate - baseline``. ``delta_pct`` is
    ``100 * delta_absolute / baseline`` and is ``None`` when ``baseline`` is 0
    (the percentage is undefined; the absolute delta still carries the change).
    ``improved`` is the direction verdict under :attr:`lower_is_better`: a strict
    move in the desired direction (a tie is not an improvement), which is what
    keeps a no-change candidate from reading as a win.
    """

    metric: str
    unit: str = ""
    lower_is_better: bool = True
    baseline: float
    candidate: float
    delta_absolute: float
    delta_pct: float | None = None
    improved: bool = False


class GuardrailCheck(_Contract):
    """One guardrail's pass/fail with a human-readable reason.

    The correctness guardrail is **non-regression** relative to the baseline: it
    fails only when the candidate is worse than the baseline (``regressed``),
    never merely because the baseline itself was imperfect — the gate protects
    against the intervention *making things worse*, not against a pre-existing
    deficiency. ``interim`` and ``interim_note`` record that the correctness
    guardrail is, for now, the un-aligned base judge (:data:`INTERIM_JUDGE_NOTE`).
    """

    name: str
    passed: bool
    reason: str
    baseline_value: ScoreValue | None = None
    candidate_value: ScoreValue | None = None
    regressed: bool = False
    judge_name: str | None = None
    interim: bool = False
    interim_note: str | None = None


class ComparisonResult(_Contract):
    """The stable artifact one candidate-vs-baseline comparison produces.

    Carries, for a single frozen Task-Suite task: the per-metric baseline and
    candidate values with their deltas (:attr:`deltas`), every guardrail's
    pass/fail with reasons (:attr:`guardrails`), whether the deterministic L0
    objective was met (:attr:`objective_met`), and the single overall
    :attr:`recommendation`. ``PROMOTE`` requires ``objective_met and
    guardrails_passed``; anything else is ``BLOCK`` with the blocking reasons in
    :attr:`reasons`.
    """

    schema_version: str = SCHEMA_VERSION
    task_id: str
    intervention: str | None = None
    objective_metric: str = "total_tokens"
    objective_met: bool = False
    guardrails_passed: bool = False
    recommendation: Recommendation = Recommendation.BLOCK
    reasons: list[str] = Field(default_factory=list)
    baseline_trace_id: str | None = None
    candidate_trace_id: str | None = None
    deltas: list[MetricDelta] = Field(default_factory=list)
    guardrails: list[GuardrailCheck] = Field(default_factory=list)
    generated_at: str | None = None  # ISO-8601
    notes: list[str] = Field(default_factory=list)

    def delta_for(self, metric: str) -> MetricDelta | None:
        """Return the :class:`MetricDelta` for ``metric``, or ``None`` if absent."""
        return next((d for d in self.deltas if d.metric == metric), None)

    def guardrail_for(self, name: str) -> GuardrailCheck | None:
        """Return the :class:`GuardrailCheck` named ``name``, or ``None`` if absent."""
        return next((g for g in self.guardrails if g.name == name), None)
