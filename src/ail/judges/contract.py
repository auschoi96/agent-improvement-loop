"""The L2 judges **output contract** — typed, JSON-shaped, versioned.

Like the L0 contract (:mod:`ail.metrics.contract`), these are the stable shapes
a downstream consumer (the loop controller's guardrail check, the Phase-4
leaderboard's "judge-human agreement trend + drift alarm") reads. They are
pydantic v2 models that round-trip through JSON without custom serialization and
forbid unknown fields so drift is loud.

The headline artifact is :class:`AgreementReport`: judge-vs-human agreement on
the Human Anchor, carrying the configurable floor and the **distrusted** signal
that fires when agreement drops below it. :class:`AlignmentReport` is the
loggable record of a MemAlign alignment cadence.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ail.judges.pools import ScoreValue

#: Version of the L2 judges output contract. Bump the minor for additive,
#: backward-compatible fields; the major for breaking shape changes.
SCHEMA_VERSION = "l2.judges/v1"


class _Contract(BaseModel):
    """Base for every contract model: forbid unknown fields so drift is loud."""

    model_config = ConfigDict(extra="forbid")


class AgreementItem(_Contract):
    """One Human-Anchor item: the judge's score beside the human's gold label.

    ``agree`` is the precomputed verdict for this item (exact match for
    categorical/bool/int labels, within-tolerance for float labels). ``error``
    is set instead of ``judge_value`` when the judge call failed for this item;
    an errored item counts against the rate (a judge that crashes does not agree)
    but is surfaced so failures are not mistaken for disagreements.
    """

    item_id: str
    human_value: ScoreValue
    judge_value: ScoreValue | None = None
    agree: bool = False
    error: str | None = None


class AgreementReport(_Contract):
    """Judge-vs-human agreement on the Human Anchor, with a floor + trust signal.

    The frozen evaluation wall (``docs/ARCHITECTURE.md`` §2) treats a drifting
    judge as a distrusted judge: when :attr:`agreement_rate` falls below
    :attr:`floor`, :attr:`distrusted` is ``True`` and the loop must stop trusting
    this judge's scores until it is re-aligned and re-measured.

    :attr:`distrusted` also fires on **insufficient data**: an empty anchor, or
    fewer scored items than the configured minimum, means the judge is
    *unmeasured*, and an unmeasured judge must never read as trusted (the
    anti-co-adaptation fail-closed rule). :attr:`insufficient_data` distinguishes
    that "we could not measure" case from "we measured and it failed the floor";
    a consumer that only reads :attr:`distrusted` is still safe either way.

    ``cohen_kappa`` is the chance-corrected companion to the raw rate (``None``
    when it does not apply — e.g. float labels compared with a tolerance, or a
    degenerate single label space); the floor is applied to the raw rate, which
    is what the guardrail thresholds on.
    """

    schema_version: str = SCHEMA_VERSION
    judge_name: str
    pool: str = "human_anchor"
    n_items: int = 0
    n_scored: int = 0
    n_agreements: int = 0
    agreement_rate: float = 0.0
    floor: float = 0.0
    distrusted: bool = False
    insufficient_data: bool = False
    cohen_kappa: float | None = None
    numeric_tolerance: float | None = None
    label_space: list[str] = Field(default_factory=list)
    items: list[AgreementItem] = Field(default_factory=list)
    generated_at: str | None = None  # ISO-8601
    notes: list[str] = Field(default_factory=list)


class AlignmentReport(_Contract):
    """The loggable record of one MemAlign alignment cadence.

    Provenance for an alignment run, deliberately decoupled from any agent
    optimization run: it records *which* judge was aligned, on *how many*
    Alignment-Set traces, with *which* optimizer. The aligned ``Judge`` object
    itself is returned alongside this report (see
    :class:`ail.judges.alignment.AlignmentOutcome`); only this serializable
    summary belongs in the contract.
    """

    schema_version: str = SCHEMA_VERSION
    base_judge_name: str
    pool: str = "alignment_set"
    optimizer: str = "MemAlign"
    n_alignment_traces: int = 0
    aligned: bool = False
    generated_at: str | None = None  # ISO-8601
    notes: list[str] = Field(default_factory=list)
