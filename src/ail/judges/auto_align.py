"""The **auto-align trigger**: turn "a human adds labels" into "the judge becomes trusted".

This is the scheduled orchestration that closes the L2 loop. The pieces already
exist — labelling (:mod:`ail.judges.labeling`), MemAlign alignment
(:mod:`ail.judges.alignment`), judge-vs-human agreement
(:mod:`ail.judges.agreement`), and registration
(:mod:`ail.judges.registration`) — but nothing *drives* them on a cadence. Today a
human labels traces in the MLflow UI and a judge stays ``aligned=false`` (and so
DISTRUSTED, per :func:`ail.judges.alignment.unaligned_report`) until someone runs
the align/audit flow by hand. This module makes that automatic: run on a schedule
and, per judged dimension, align the judge with MemAlign once enough human labels
exist, re-align as more accrue, guard trust with the agreement floor, and roll
back a regression.

It is **orchestration only** — it reuses, and never reimplements, the L2 pieces:

* **Which labels** — :func:`read_human_labels` reads the ``HUMAN``-sourced
  assessments whose *name matches the judge* off the experiment's traces (the L1
  ``label-schema-name == judge-name`` convention, see
  :mod:`ail.judges.authoring`). This is the same read shape the manipulate/rollback
  showcase proved live (``docs/MEMALIGN_ROLLBACK.md``).
* **Split + prove the wall** — :func:`ail.judges.labeling.assemble_pools` builds
  the disjoint ``(AlignmentSet, HumanAnchor)`` and re-proves disjointness.
* **Align** — :func:`ail.judges.alignment.align_judge` (MemAlign on the Alignment
  Set only).
* **Audit** — :func:`ail.judges.agreement.score_anchor` on the held-out Human
  Anchor, which fails closed: an **unmeasured** judge (empty/under-sampled anchor)
  reads as DISTRUSTED, never trusted, and the agreement number is never fabricated.
* **Promote** — :func:`ail.judges.registration.register_prealigned_scorer`
  registers *the exact judge whose agreement was measured*.

The trust logic this module adds on top:

#. **Floor gate** — align only when there are at least ``label_floor`` (default
   :data:`DEFAULT_LABEL_FLOOR`) human labels for the judge.
#. **Watermark gate (idempotent + re-align-over-time)** — a per-judge watermark
   (the label count at the last cadence that ran) is persisted, so a run does not
   re-align on the same labels, and *does* re-align once more labels accrue past
   it.
#. **Agreement-floor guard (fail-closed)** — the freshly-aligned judge is only
   promoted when its held-out agreement is not ``distrusted`` (measured **and** at
   or above the floor).
#. **Rollback (fail-closed toward last-known-good)** — a re-alignment whose
   held-out agreement regresses below the previously-promoted version is **not**
   registered; the prior aligned version stays live.

Because each cadence re-aligns a *fresh* judge from the base spec over the full
current label set, "rollback" is simply "do not promote the regressed candidate":
the incumbent registered scorer is left untouched (registry-versioning rollback),
so there is never a window where a regressed judge is live. This is the
complement to MemAlign's ``unalign`` (used by the manipulate showcase for
incremental retraction); here a from-scratch re-alignment makes not-promoting the
correct, simplest fail-closed move.

Model-only, scheduled. The trace-store tables are **views** (no table-update
trigger is possible — the same reason the optimization cycle is scheduled), so
this runs on a cron. It reads labeled traces and calls the reflection/embedding
and judge models through the gateway; it needs no SQL write path of its own.
MLflow is imported lazily (matching the rest of the package), so importing this
module never requires a backend; the model/MLflow-touching pieces are injected
(``source`` / ``store``) so the whole trigger is unit-testable offline — see
``tests/test_judges_auto_align.py``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ail.judges.agreement import AgreementConfig, coerce_score, score_anchor
from ail.judges.alignment import align_judge
from ail.judges.contract import AgreementReport
from ail.judges.labeling import (
    DEFAULT_ANCHOR_FRACTION,
    DEFAULT_LABELER_ID,
    TraceLabel,
    assemble_pools,
)
from ail.judges.registration import (
    DEFAULT_SAMPLING_RATE,
    ScorerRegistration,
    register_prealigned_scorer,
)
from ail.judges.scorers import DEFAULT_SCORERS, ScorerSpec, make_scorer
from ail.pools import ScoreValue

if TYPE_CHECKING:
    from mlflow.genai.judges.base import AlignmentOptimizer

    from ail.ingest.base import TraceSource

__all__ = [
    "DEFAULT_LABEL_FLOOR",
    "AUTOALIGN_TAG_PREFIX",
    "AutoAlignConfig",
    "AutoAlignState",
    "AutoAlignStatus",
    "JudgeAutoAlignResult",
    "AutoAlignReport",
    "WatermarkStore",
    "ExperimentTagWatermarkStore",
    "read_human_labels",
    "auto_align_judge",
    "auto_align_scorers",
]

#: Default minimum number of human labels a judge needs before its **first**
#: alignment. Deliberately a knob, not a constant: too low and MemAlign aligns
#: from a handful of examples the held-out anchor can't audit; ~20 leaves enough
#: to both align and hold out a meaningful Human Anchor (at ~0.3 anchor fraction,
#: ~6 held-out items). Aligns with the readiness ``HUMAN_LABELS`` gate intent.
DEFAULT_LABEL_FLOOR = 20

#: Prefix for the per-judge auto-align **watermark** experiment tags. Mirrors
#: :data:`ail.judges.registration.ALIGNED_TAG_PREFIX` (``ail.judge.<name>.aligned``):
#: the scheduled-scorer API exposes no per-scorer metadata slot, so the watermark
#: is persisted as experiment tags ``ail.autoalign.<name>.{label_count,agreement,
#: aligned_at}``. Queryable and durable across scheduled runs.
AUTOALIGN_TAG_PREFIX = "ail.autoalign."


@dataclass(frozen=True, slots=True)
class AutoAlignConfig:
    """Knobs for one auto-align cadence.

    Args:
        label_floor: Minimum human labels before a judge's first alignment (see
            :data:`DEFAULT_LABEL_FLOOR`).
        agreement: The :class:`~ail.judges.agreement.AgreementConfig` used to
            audit the freshly-aligned judge — its ``floor`` is the trust
            threshold, its ``min_samples`` the fail-closed under-measurement guard.
        anchor_fraction: Fraction of labeled traces held out as the Human Anchor
            (forwarded to :func:`ail.judges.labeling.assemble_pools`).
        seed: Deterministic split seed (reproducible pools across runs).
        sampling_rate: Scheduled-scorer sampling rate applied on promotion.
        max_results: Trace-fetch ceiling when reading labels (``None`` → no cap).
        labeler_id: Preferred labeler when a trace carries labels from several;
            also stamped on the Alignment Set's attached human assessments.
    """

    label_floor: int = DEFAULT_LABEL_FLOOR
    agreement: AgreementConfig = field(default_factory=AgreementConfig)
    anchor_fraction: float = DEFAULT_ANCHOR_FRACTION
    seed: int = 0
    sampling_rate: float = DEFAULT_SAMPLING_RATE
    max_results: int | None = None
    labeler_id: str = DEFAULT_LABELER_ID


@dataclass(frozen=True, slots=True)
class AutoAlignState:
    """The persisted per-judge watermark.

    ``label_count`` is the number of human labels present at the last cadence that
    **ran** alignment for this judge (the idempotency + re-align watermark).
    ``agreement`` is the held-out agreement of the last **promoted** aligned
    version — the last-known-good bar a re-alignment must not regress below
    (``None`` until a version has been promoted). ``aligned_at`` is when that
    promotion happened (ISO-8601), or ``None``.
    """

    label_count: int = 0
    agreement: float | None = None
    aligned_at: str | None = None


class AutoAlignStatus(StrEnum):
    """The outcome of one judge's auto-align cadence."""

    #: Fewer than ``label_floor`` human labels — not aligning yet.
    SKIPPED_BELOW_FLOOR = "skipped_below_floor"
    #: No new labels since the last alignment (watermark) — idempotent no-op.
    SKIPPED_NO_NEW_LABELS = "skipped_no_new_labels"
    #: Aligned and promoted: the aligned judge passed the floor and did not regress.
    ALIGNED = "aligned"
    #: Aligned but held DISTRUSTED (unmeasured or below the agreement floor); the
    #: candidate was not promoted. Fail closed.
    HELD_DISTRUSTED = "held_distrusted"
    #: Aligned but regressed below the prior promoted version; kept the prior
    #: aligned version (rollback). Fail closed toward last-known-good.
    ROLLED_BACK = "rolled_back"
    #: The cadence raised for this judge (surfaced, never swallowed as success).
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class JudgeAutoAlignResult:
    """The record of one judge's auto-align cadence (serializable-friendly)."""

    judge_name: str
    status: AutoAlignStatus
    label_count: int
    watermark: int
    prior_agreement: float | None
    promoted: bool
    agreement: AgreementReport | None = None
    registration: ScorerRegistration | None = None
    state: AutoAlignState | None = None
    error: str | None = None
    notes: tuple[str, ...] = ()

    @property
    def agreement_rate(self) -> float | None:
        """The freshly-aligned judge's held-out agreement rate, if it was measured."""
        return None if self.agreement is None else self.agreement.agreement_rate


@dataclass(frozen=True, slots=True)
class AutoAlignReport:
    """The report of one full cadence over a set of judged dimensions."""

    experiment_id: str
    results: tuple[JudgeAutoAlignResult, ...]
    generated_at: str

    def _count(self, status: AutoAlignStatus) -> int:
        return sum(1 for r in self.results if r.status is status)

    @property
    def n_aligned(self) -> int:
        return self._count(AutoAlignStatus.ALIGNED)

    @property
    def n_rolled_back(self) -> int:
        return self._count(AutoAlignStatus.ROLLED_BACK)

    @property
    def n_held_distrusted(self) -> int:
        return self._count(AutoAlignStatus.HELD_DISTRUSTED)

    @property
    def n_skipped(self) -> int:
        return self._count(AutoAlignStatus.SKIPPED_BELOW_FLOOR) + self._count(
            AutoAlignStatus.SKIPPED_NO_NEW_LABELS
        )

    @property
    def n_failed(self) -> int:
        return self._count(AutoAlignStatus.FAILED)


# --- watermark store -------------------------------------------------------


@runtime_checkable
class WatermarkStore(Protocol):
    """The persistence seam for the per-judge watermark.

    Injectable so the whole trigger is unit-testable with an in-memory store;
    :class:`ExperimentTagWatermarkStore` is the production implementation backed
    by experiment tags.
    """

    def read(self, judge_name: str) -> AutoAlignState: ...

    def write(self, judge_name: str, state: AutoAlignState) -> None: ...


@dataclass
class ExperimentTagWatermarkStore:
    """A :class:`WatermarkStore` persisting state as experiment tags.

    Mirrors :func:`ail.judges.registration._tag_alignment`: the scheduled-scorer
    API has no per-scorer metadata slot, so the watermark lives as three
    experiment tags per judge under :data:`AUTOALIGN_TAG_PREFIX`
    (``label_count`` / ``agreement`` / ``aligned_at``). Reads are best-effort —
    a missing/unreadable tag set yields a zeroed :class:`AutoAlignState` (so a
    judge with no recorded watermark is treated as never-aligned); **writes are
    not** swallowed, because losing a watermark write silently would make the
    trigger re-align the same labels every run.

    ``client`` is injectable (any object exposing ``get_experiment`` /
    ``set_experiment_tag``); when ``None`` an :class:`mlflow.MlflowClient` is built
    against the configured Databricks workspace lazily on first use.
    """

    experiment_id: str
    profile: str | None = None
    tracking_uri: str = "databricks"
    registry_uri: str = "databricks-uc"
    client: Any = None

    def _mlflow_client(self) -> Any:
        if self.client is not None:
            return self.client
        import mlflow
        from mlflow import MlflowClient

        if self.profile:
            import os

            os.environ.setdefault("DATABRICKS_CONFIG_PROFILE", self.profile)
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_registry_uri(self.registry_uri)
        self.client = MlflowClient()
        return self.client

    def _key(self, judge_name: str, field_name: str) -> str:
        return f"{AUTOALIGN_TAG_PREFIX}{judge_name}.{field_name}"

    def read(self, judge_name: str) -> AutoAlignState:
        try:
            experiment = self._mlflow_client().get_experiment(self.experiment_id)
            tags: Mapping[str, str] = dict(getattr(experiment, "tags", None) or {})
        except Exception:  # noqa: BLE001 - unreadable tags -> never-aligned (fail closed)
            return AutoAlignState()
        return AutoAlignState(
            label_count=_as_int(tags.get(self._key(judge_name, "label_count"))),
            agreement=_as_float(tags.get(self._key(judge_name, "agreement"))),
            aligned_at=tags.get(self._key(judge_name, "aligned_at")) or None,
        )

    def write(self, judge_name: str, state: AutoAlignState) -> None:
        client = self._mlflow_client()
        client.set_experiment_tag(
            self.experiment_id, self._key(judge_name, "label_count"), str(state.label_count)
        )
        client.set_experiment_tag(
            self.experiment_id,
            self._key(judge_name, "agreement"),
            "" if state.agreement is None else repr(state.agreement),
        )
        client.set_experiment_tag(
            self.experiment_id, self._key(judge_name, "aligned_at"), state.aligned_at or ""
        )


def _as_int(value: str | None) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# --- reading human labels off the experiment's traces ----------------------


def _is_human(assessment: Any) -> bool:
    """Whether an assessment is ``HUMAN``-sourced (best-effort, by source-type name)."""
    source = getattr(assessment, "source", None)
    return str(getattr(source, "source_type", "")) == "HUMAN"


def _human_label(
    raw: Any, judge_name: str, labeler_id: str | None
) -> tuple[ScoreValue, str | None] | None:
    """The human label + rationale for ``judge_name`` on a raw trace, or ``None``.

    Reads ``raw.info.assessments`` and keeps the ``HUMAN``-sourced ones whose name
    matches the judge (the L1 name-matching convention). ``labeler_id``, when
    given, *prefers* that labeler's assessment but falls back to any human one.
    Returns the first assessment with a coercible value (via the shared
    :func:`ail.judges.agreement.coerce_score`), or ``None`` when the trace carries
    no human label for the judge — the caller skips it rather than inventing one.
    """
    info = getattr(raw, "info", None)
    assessments = getattr(info, "assessments", None) if info is not None else None
    if not assessments:
        return None
    human = [a for a in assessments if getattr(a, "name", None) == judge_name and _is_human(a)]
    if labeler_id is not None:
        preferred = [
            a for a in human if getattr(getattr(a, "source", None), "source_id", None) == labeler_id
        ]
        human = preferred or human
    for assessment in human:
        value = coerce_score(assessment)
        if value is not None:
            return value, getattr(assessment, "rationale", None)
    return None


def read_human_labels(
    source: TraceSource,
    *,
    experiment_id: str,
    judge_name: str,
    max_results: int | None = None,
    labeler_id: str | None = None,
) -> list[TraceLabel]:
    """Read the ``HUMAN`` labels named for ``judge_name`` off an experiment's traces.

    Iterates the experiment's traces through the ingest seam
    (:meth:`ail.ingest.base.TraceSource.iter_traces`) and, for each, reads the
    human label carried on ``trace.info.assessments`` whose name matches the judge
    (:func:`_human_label`). Each labeled trace yields one
    :class:`~ail.judges.labeling.TraceLabel` (``inputs``/``outputs`` are left
    unset — a MemAlign-alignable ``{{ trace }}`` judge is aligned and audited from
    the trace itself). Traces with no resolvable id, or no human label for the
    judge, are skipped — labels are never fabricated.

    This is the read side of the L1 convention: a human labels traces under the
    label schema whose name equals the judge name, and the trigger finds a judge's
    labels by that name.
    """
    labels: list[TraceLabel] = []
    for trace in source.iter_traces(
        experiment_id=experiment_id, max_results=max_results, order_by=["timestamp_ms DESC"]
    ):
        trace_id = getattr(trace, "trace_id", None)
        if not trace_id:
            continue
        graded = _human_label(getattr(trace, "raw", None), judge_name, labeler_id)
        if graded is None:
            continue
        value, rationale = graded
        labels.append(
            TraceLabel(trace_id=trace_id, name=judge_name, value=value, rationale=rationale)
        )
    return labels


# --- the trigger -----------------------------------------------------------


def auto_align_judge(
    spec: ScorerSpec,
    *,
    experiment_id: str,
    source: TraceSource,
    store: WatermarkStore,
    config: AutoAlignConfig | None = None,
    optimizer: AlignmentOptimizer | None = None,
    model: str | None = None,
    register: bool = True,
    profile: str | None = None,
    now: str | None = None,
) -> JudgeAutoAlignResult:
    """Run one auto-align cadence for a single judged dimension.

    The gated flow (all reuse the L2 pieces; this function only decides *whether*
    and *when* to run them and *whether to keep* the result):

    #. Read the judge's ``HUMAN`` labels (:func:`read_human_labels`).
    #. **Floor gate** — skip if fewer than ``config.label_floor`` labels.
    #. **Watermark gate** — skip if no new labels since the last alignment
       (idempotent); proceed only when the count has grown past the watermark.
    #. Assemble the disjoint pools (:func:`ail.judges.labeling.assemble_pools`),
       align a fresh judge (:func:`ail.judges.alignment.align_judge`), and audit it
       on the held-out anchor (:func:`ail.judges.agreement.score_anchor`).
    #. **Agreement-floor guard** — if the audited judge is ``distrusted``
       (unmeasured or below the floor), hold it (do not promote). Fail closed.
    #. **Rollback guard** — if its held-out agreement regresses below the prior
       promoted version, keep the prior version. Fail closed toward last-known-good.
    #. Otherwise **promote**: register the exact aligned judge that was measured
       (:func:`ail.judges.registration.register_prealigned_scorer`) unless
       ``register=False`` (preview), and advance the watermark + agreement bar.

    The watermark is advanced to the current label count on every cadence that
    *ran* alignment — including a held/rolled-back one — so the same labels are
    never re-aligned repeatedly; a future cadence retries only once more labels
    accrue. The last-known-good agreement bar is advanced **only** on promotion.

    Args:
        spec: The scorer definition for the dimension (its ``name`` is the label
            name the trigger matches on).
        experiment_id: MLflow experiment holding the labeled traces.
        source: Trace source used to read labels and assemble pools.
        store: Watermark persistence seam.
        config: Cadence knobs (defaults to :class:`AutoAlignConfig`).
        optimizer: Optional pre-built MemAlign optimizer; ``None`` uses MLflow's
            default MemAlign (see :func:`ail.judges.alignment.align_judge`).
        model: Judge model URI; ``None`` uses MLflow's default judge model.
        register: Register the promoted judge as a scheduled scorer (default
            ``True``); ``False`` runs the full decision but skips registration.
        profile: Databricks CLI profile forwarded to registration.
        now: ISO-8601 timestamp for the reports (defaults to now); handy for tests.

    Returns:
        A :class:`JudgeAutoAlignResult` recording the decision and its provenance.
    """
    cfg = config or AutoAlignConfig()
    generated_at = now or datetime.now(UTC).isoformat()

    labels = read_human_labels(
        source,
        experiment_id=experiment_id,
        judge_name=spec.name,
        max_results=cfg.max_results,
        labeler_id=cfg.labeler_id,
    )
    count = len(labels)
    state = store.read(spec.name)
    watermark = state.label_count
    prior_agreement = state.agreement

    def _result(
        status: AutoAlignStatus,
        *,
        promoted: bool = False,
        agreement: AgreementReport | None = None,
        registration: ScorerRegistration | None = None,
        new_state: AutoAlignState | None = None,
        notes: Sequence[str] = (),
    ) -> JudgeAutoAlignResult:
        return JudgeAutoAlignResult(
            judge_name=spec.name,
            status=status,
            label_count=count,
            watermark=watermark,
            prior_agreement=prior_agreement,
            promoted=promoted,
            agreement=agreement,
            registration=registration,
            state=new_state,
            notes=tuple(notes),
        )

    # (1) Floor gate: not enough human labels to align + hold out an anchor yet.
    if count < cfg.label_floor:
        return _result(
            AutoAlignStatus.SKIPPED_BELOW_FLOOR,
            notes=[f"{count} human label(s) < floor {cfg.label_floor}: not aligning yet"],
        )

    # (2) Watermark gate: idempotent (no re-align on the same labels) and the
    # re-align trigger (proceed only once labels grow past the watermark).
    if count <= watermark:
        return _result(
            AutoAlignStatus.SKIPPED_NO_NEW_LABELS,
            notes=[
                f"{count} label(s) <= watermark {watermark}: no new labels since last alignment"
            ],
        )

    # (3) Split into disjoint pools and prove the wall (assemble_pools re-checks).
    alignment_set, anchor = assemble_pools(
        source,
        labels,
        judge_name=spec.name,
        anchor_fraction=cfg.anchor_fraction,
        seed=cfg.seed,
        labeler_id=cfg.labeler_id,
    )
    if len(alignment_set) == 0 or len(anchor) == 0:
        # Cannot both align and audit (e.g. traces unfetchable). Fail closed: hold,
        # keep any incumbent, and advance the watermark so we don't retry the same
        # labels every run.
        held = AutoAlignState(
            label_count=count, agreement=prior_agreement, aligned_at=state.aligned_at
        )
        store.write(spec.name, held)
        return _result(
            AutoAlignStatus.HELD_DISTRUSTED,
            new_state=held,
            notes=[
                "could not form both a non-empty Alignment Set and Human Anchor "
                f"(alignment={len(alignment_set)}, anchor={len(anchor)}); held (fail closed)"
            ],
        )

    # (4) Align a fresh judge on the full labeled set, then audit it on the anchor.
    judge = make_scorer(spec, model=model)
    outcome = align_judge(judge, alignment_set, optimizer=optimizer, generated_at=generated_at)
    agreement = score_anchor(outcome.judge, anchor, config=cfg.agreement, generated_at=generated_at)

    # (5) Agreement-floor guard: an unmeasured or below-floor judge stays
    # DISTRUSTED and is never promoted. The agreement number is score_anchor's,
    # never fabricated.
    if agreement.distrusted:
        held = AutoAlignState(
            label_count=count, agreement=prior_agreement, aligned_at=state.aligned_at
        )
        store.write(spec.name, held)
        reason = (
            "unmeasured (insufficient anchor data)"
            if agreement.insufficient_data
            else f"agreement {agreement.agreement_rate} < floor {agreement.floor}"
        )
        return _result(
            AutoAlignStatus.HELD_DISTRUSTED,
            agreement=agreement,
            new_state=held,
            notes=[f"aligned judge is DISTRUSTED ({reason}); held, kept prior (fail closed)"],
        )

    # (6) Rollback guard: a re-alignment that regresses below the last promoted
    # version keeps the prior version live (registry-versioning rollback).
    if prior_agreement is not None and agreement.agreement_rate < prior_agreement:
        held = AutoAlignState(
            label_count=count, agreement=prior_agreement, aligned_at=state.aligned_at
        )
        store.write(spec.name, held)
        return _result(
            AutoAlignStatus.ROLLED_BACK,
            agreement=agreement,
            new_state=held,
            notes=[
                f"held-out agreement {agreement.agreement_rate} < prior aligned "
                f"{prior_agreement}: kept prior aligned version (rollback, fail closed)"
            ],
        )

    # (7) Promote: register the exact aligned judge whose agreement was measured.
    registration: ScorerRegistration | None = None
    if register:
        registration = register_prealigned_scorer(
            outcome.judge,
            outcome.report,
            experiment_id=experiment_id,
            sampling_rate=cfg.sampling_rate,
            profile=profile,
        )
    promoted_state = AutoAlignState(
        label_count=count, agreement=agreement.agreement_rate, aligned_at=generated_at
    )
    store.write(spec.name, promoted_state)
    verb = "registered" if register else "measured (register=False, not registered)"
    return _result(
        AutoAlignStatus.ALIGNED,
        promoted=True,
        agreement=agreement,
        registration=registration,
        new_state=promoted_state,
        notes=[
            f"aligned on {len(alignment_set)} trace(s), held-out agreement "
            f"{agreement.agreement_rate} >= floor {agreement.floor}; promoted and {verb}"
        ],
    )


def auto_align_scorers(
    experiment_id: str,
    *,
    source: TraceSource | None = None,
    store: WatermarkStore | None = None,
    scorers: Mapping[str, ScorerSpec] = DEFAULT_SCORERS,
    config: AutoAlignConfig | None = None,
    optimizer: AlignmentOptimizer | None = None,
    model: str | None = None,
    register: bool = True,
    profile: str | None = None,
    now: str | None = None,
) -> AutoAlignReport:
    """Run one auto-align cadence over several judged dimensions.

    Calls :func:`auto_align_judge` for each spec, isolating failures per judge (a
    judge whose cadence raises is recorded as :attr:`AutoAlignStatus.FAILED` and
    reported, never swallowed as success — and never aborts the others). A judge
    with fewer than ``config.label_floor`` labels simply skips, so running over the
    full built-in scorer set is harmless: only dimensions humans are labelling get
    aligned.

    When ``source`` / ``store`` are not supplied they default to a live
    :class:`~ail.ingest.mlflow_source.MLflowTraceSource` and an
    :class:`ExperimentTagWatermarkStore` for ``experiment_id`` (both built lazily,
    so importing this module never touches a backend).
    """
    cfg = config or AutoAlignConfig()
    generated_at = now or datetime.now(UTC).isoformat()
    src = source if source is not None else _default_source(profile)
    watermarks = (
        store
        if store is not None
        else ExperimentTagWatermarkStore(experiment_id=experiment_id, profile=profile)
    )

    results: list[JudgeAutoAlignResult] = []
    for spec in scorers.values():
        try:
            result = auto_align_judge(
                spec,
                experiment_id=experiment_id,
                source=src,
                store=watermarks,
                config=cfg,
                optimizer=optimizer,
                model=model,
                register=register,
                profile=profile,
                now=generated_at,
            )
        except Exception as exc:  # noqa: BLE001 - one judge's failure must not abort the cadence
            result = JudgeAutoAlignResult(
                judge_name=spec.name,
                status=AutoAlignStatus.FAILED,
                label_count=0,
                watermark=0,
                prior_agreement=None,
                promoted=False,
                error=str(exc),
                notes=(f"cadence failed: {exc}",),
            )
        results.append(result)
    return AutoAlignReport(
        experiment_id=experiment_id, results=tuple(results), generated_at=generated_at
    )


def _default_source(profile: str | None) -> TraceSource:
    """Build the live MLflow trace source (lazy import, matching the package)."""
    from ail.ingest.mlflow_source import MLflowTraceSource

    return MLflowTraceSource(profile=profile)
