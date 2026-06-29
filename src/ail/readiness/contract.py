"""The readiness & eval-health **output contract** — typed, JSON-shaped, versioned.

Like the L0 (:mod:`ail.metrics.contract`), L2 (:mod:`ail.judges.contract`), and
comparison (:mod:`ail.compare.contract`) contracts, these are the stable shapes a
downstream consumer reads — here the **Readiness panel** of
``docs/READINESS_AND_TRUST.md`` §2 and the **eval-health surface** of §3. They are
pydantic v2 models that round-trip through JSON (``model_dump_json`` /
``model_validate_json``) without custom serialization and forbid unknown fields so
drift is loud.

Two headline artifacts:

* :class:`ReadinessStatus` — per-cohort, per-goal readiness: a :class:`ReadinessTier`,
  one :class:`Gate` per data gate (pass/fail + a human-readable reason), the
  aggregated unmet-gate :attr:`~ReadinessStatus.reasons`, and the embedded
  :class:`EvalHealth`. It **fails closed**: when a gate is unmet the tier is a
  not-ready tier with the reason spelled out — never green on missing data.
* :class:`EvalHealth` — the eval-health / coverage surface: scored-coverage %
  (fraction of cohort traces carrying a real judge verdict), judge-run success
  rate, and the count of **distrusted** judges. An unmeasured judge is distrusted
  by default (§3 risk #1), so a judge never silently reads as trusted.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

#: Version of the readiness output contract. Bump the minor for additive,
#: backward-compatible fields; the major for breaking shape changes.
SCHEMA_VERSION = "ail.readiness/v1"


class _Contract(BaseModel):
    """Base for every contract model: forbid unknown fields so drift is loud."""

    model_config = ConfigDict(extra="forbid")


class ReadinessTier(StrEnum):
    """How far a cohort has progressed toward proving an improvement for a goal.

    The ladder of ``docs/READINESS_AND_TRUST.md`` §2, fail-closed: a cohort only
    climbs as its data gates pass, and the not-ready tiers are first-class states
    (the refusal is the feature), not error conditions.

    * :attr:`COLLECTING` — too few traces to even baseline (0 traces is the
      limiting case). No baseline, no claims; tell the user what to connect.
    * :attr:`BASELINE_ONLY` — enough traces to baseline + diagnose waste, but
      **cannot prove improvement**: not enough traces for statistical power and/or
      (for a quality goal) the quality gates are unmet.
    * :attr:`READY_FOR_QUALITY` — a *quality* goal's gates all pass (frozen suite,
      human labels, a trusted/calibrated judge, scored-coverage) so a trustworthy
      quality signal exists, but trace count is still below the prove floor.
    * :attr:`READY_TO_PROVE` — the data is sufficient to prove an improvement for
      this goal. (The comparison harness still requires candidate runs on the
      frozen Task Suite; this tier gates *data sufficiency*, not the run itself.)
    """

    COLLECTING = "collecting"
    BASELINE_ONLY = "baseline_only"
    READY_FOR_QUALITY = "ready_for_quality"
    READY_TO_PROVE = "ready_to_prove"


class GateName(StrEnum):
    """Stable identifier of a single readiness gate.

    Trace gates apply to every goal; the rest are evaluated only for a goal whose
    :attr:`~ail.readiness.goal.GoalView.requires_quality` is set (a quality claim
    needs a frozen suite, human labels, a trusted judge, and real coverage).
    """

    TRACE_BASELINE = "trace_baseline"
    TRACE_PROVE = "trace_prove"
    FROZEN_SUITE = "frozen_suite"
    HUMAN_LABELS = "human_labels"
    JUDGE_TRUSTED = "judge_trusted"
    SCORED_COVERAGE = "scored_coverage"


class Gate(_Contract):
    """One readiness gate's pass/fail with a human-readable reason.

    ``reason`` is the user-facing string for the Readiness panel — on a failing
    gate it says exactly what is missing ("need N more traces", "need M human
    labels", "no frozen Task Suite", "scored-coverage X% below floor Y%").
    """

    name: GateName
    passed: bool
    reason: str


class JudgeHealth(_Contract):
    """Trust + coverage of a single judge over a cohort.

    ``distrusted`` defaults to ``True``: a judge that has not been measured
    against humans, or is below its agreement floor, is **distrusted by default**
    (``docs/READINESS_AND_TRUST.md`` §3 — never silently trusted). ``measured``
    distinguishes "never measured against humans" from "measured and failed the
    floor". ``coverage`` is the fraction of cohort traces this judge produced a
    real verdict for.
    """

    judge_name: str
    measured: bool = False
    agreement_rate: float | None = None
    agreement_floor: float = 0.0
    distrusted: bool = True
    n_scored_traces: int = 0
    coverage: float = 0.0
    reason: str = ""


class EvalHealth(_Contract):
    """The eval-health / coverage surface for a cohort (``docs/READINESS_AND_TRUST.md`` §3).

    Catches "coverage gap masquerading as health" (§3 risk #2): it reports the
    fraction of traces actually carrying a real verdict and the judge-run success
    rate, not merely that judges are *configured*. ``scored_coverage`` is
    ``n_scored_traces / n_traces`` (``0.0`` when the cohort is empty).
    ``judge_run_success_rate`` is ``None`` when no judge runs were recorded — the
    fail-loud "did not evaluate" signal, distinct from "evaluated and failed".
    """

    schema_version: str = SCHEMA_VERSION
    cohort_name: str
    n_traces: int = 0
    n_scored_traces: int = 0
    scored_coverage: float = 0.0
    coverage_floor: float = 0.0
    judge_runs: int = 0
    judge_run_successes: int = 0
    judge_run_success_rate: float | None = None
    n_judges: int = 0
    n_distrusted_judges: int = 0
    distrusted_judges: list[str] = Field(default_factory=list)
    judges: list[JudgeHealth] = Field(default_factory=list)
    generated_at: str | None = None  # ISO-8601
    notes: list[str] = Field(default_factory=list)


class ReadinessStatus(_Contract):
    """Per-cohort, per-goal readiness — the artifact the Readiness panel consumes.

    Carries the :attr:`tier`, every :class:`Gate` evaluated for this goal with its
    pass/fail and reason, the aggregated unmet-gate :attr:`reasons` ("you need N
    more traces / M more labels / a frozen baseline"), and the embedded
    :class:`EvalHealth`. Provenance (cohort name, objective metric, guardrails,
    trace count) travels with it so no number is opaque.

    Fail-closed by construction: an unmet gate yields a not-ready tier with the
    reason spelled out. :attr:`can_prove_improvement` — the single flag a
    comparison harness consults before emitting a ``PROMOTE`` — is ``True`` only
    at :attr:`ReadinessTier.READY_TO_PROVE`.
    """

    schema_version: str = SCHEMA_VERSION
    cohort_name: str
    objective_metric: str
    requires_quality: bool = False
    guardrail_names: list[str] = Field(default_factory=list)
    trace_count: int = 0
    tier: ReadinessTier = ReadinessTier.COLLECTING
    gates: list[Gate] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    eval_health: EvalHealth
    generated_at: str | None = None  # ISO-8601
    notes: list[str] = Field(default_factory=list)

    @property
    def can_prove_improvement(self) -> bool:
        """Whether the data is sufficient to claim an improvement for this goal.

        ``True`` only at :attr:`ReadinessTier.READY_TO_PROVE`. This is the wall: a
        consumer that gates an improvement claim on this flag cannot turn green on
        missing data. Derived from the serialized :attr:`tier` (the source of
        truth) rather than stored, so the two can never drift; a JSON-only reader
        gets the same answer from ``tier == "ready_to_prove"``.
        """
        return self.tier == ReadinessTier.READY_TO_PROVE

    def gate_for(self, name: GateName) -> Gate | None:
        """Return the :class:`Gate` named ``name``, or ``None`` if not evaluated."""
        return next((g for g in self.gates if g.name == name), None)
