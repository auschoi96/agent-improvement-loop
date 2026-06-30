"""Record human labels on traces and assemble the two MemAlign-input pools.

MemAlign aligns a judge *against human labels* — but the reference experiment has
**zero** (``docs/ARCHITECTURE.md`` §8). This module is the missing input stage:
it lets a human record labels on a slice of real traces, and assembles those
labels into the two **disjoint** pools the L2 layer consumes:

* an :class:`~ail.pools.AlignmentSet` — raw MLflow traces (carrying the
  human assessments) that :func:`ail.judges.alignment.align_judge` /
  :func:`ail.judges.registration.create_aligned_scorer` learn from; and
* a :class:`~ail.pools.HumanAnchor` — held-out human labels that
  :func:`ail.judges.agreement.score_anchor` audits the aligned judge against.

The two pools are **never mixed**: :func:`split_labels` partitions by *trace*
(every label of a trace lands in one pool), and :func:`assemble_pools` re-proves
disjointness with :func:`~ail.pools.assert_pools_disjoint` (the
:class:`ail.pools.Pool`-keyed wall) before returning. Measuring agreement on the
same labels the judge was aligned against would only report how well alignment
memorized them, which is the co-adaptation §2 forbids.

How labels become aligned judges — the human assessments are MLflow assessments
on the subject trace (``source_type=HUMAN``), exactly the feedback-attachment
model of ``docs/ARCHITECTURE.md`` §11: a human ``token_efficiency`` (``HUMAN``)
and a judge ``token_efficiency`` (``LLM_JUDGE``) coexist on one trace, keyed by
``(name, source_type)``. :func:`record_label` writes the human side; MemAlign and
the agreement layer read it back.

MLflow is imported lazily (matching the rest of the package), so importing this
module never requires a tracking backend; only :func:`record_label` /
:func:`record_labels` touch one. They assume MLflow is already configured — true
inside a Databricks notebook, and otherwise after constructing the
:class:`~ail.ingest.mlflow_source.MLflowTraceSource` used for pool assembly (it
points MLflow at the workspace). See the workflow snippet in this module's
docstring and ``docs/L2_JUDGES_CONTRACT.md``.

Workflow (label ~30–50 traces, then align + audit)::

    import mlflow
    from ail.ingest.mlflow_source import MLflowTraceSource
    from ail.judges.labeling import TraceLabel, record_labels, assemble_pools
    from ail.judges import create_aligned_scorer, score_anchor, TOKEN_EFFICIENCY
    from ail.judges.scorers import build_token_efficiency_inputs

    mlflow.set_tracking_uri("databricks")          # a notebook already has this
    source = MLflowTraceSource(profile="dais-demo")

    # 1. A human grades each trace (here: token efficiency 1–5 with a reason).
    labels = [
        TraceLabel(trace_id="tr-abc", name="token_efficiency", value=2,
                   rationale="re-read foo.py 34x for no gain",
                   inputs=build_token_efficiency_inputs(metrics_abc, task="refactor X"),
                   outputs="<the agent's final response>"),
        # ... ~30–50 of these ...
    ]
    record_labels(labels, labeler_id="austin")     # writes HUMAN assessments

    # 2. Split into disjoint pools and align-then-register the judge.
    alignment_set, anchor = assemble_pools(source, labels, judge_name="token_efficiency")
    reg = create_aligned_scorer(TOKEN_EFFICIENCY, experiment_id="660599403165942",
                                alignment_set=alignment_set)   # reg.aligned is True now

    # 3. Audit the *aligned* judge against the held-out anchor (reg.judge is it).
    report = score_anchor(reg.judge, anchor)
    if report.distrusted:
        ...   # agreement below floor (or too few anchor items): don't trust it yet
"""

from __future__ import annotations

import copy
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ail.pools import (
    AlignmentSet,
    AnchorItem,
    HumanAnchor,
    ScoreValue,
    assert_pools_disjoint,
)

if TYPE_CHECKING:
    from mlflow.entities import Assessment

    from ail.ingest.base import TraceSource

__all__ = [
    "DEFAULT_LABELER_ID",
    "DEFAULT_ANCHOR_FRACTION",
    "TraceLabel",
    "record_label",
    "record_labels",
    "split_labels",
    "to_alignment_set",
    "to_human_anchor",
    "assemble_pools",
]

#: Default ``source_id`` recorded on a human assessment. Override per labeler so
#: assessments are attributable (the ``(name, source_type, source_id)`` triple).
DEFAULT_LABELER_ID = "expert"

#: Default fraction of *labeled traces* held out as the Human Anchor (audit),
#: with the remainder used as the Alignment Set (align). A knob, not a constant:
#: a larger anchor measures agreement more tightly but leaves fewer traces to
#: align from. ~0.3 keeps a useful audit slice while leaving the majority to align.
DEFAULT_ANCHOR_FRACTION = 0.3


@dataclass(frozen=True, slots=True)
class TraceLabel:
    """One human label on a trace, plus the data to audit the judge later.

    ``name`` is the assessment name; align a judge of the **same** name from it
    (e.g. ``"token_efficiency"`` labels align the token-efficiency judge).
    ``value`` is the human's grade (``"yes"``/``"no"`` for a categorical judge,
    ``1``–``5`` for a graded one). ``inputs``/``outputs``/``expectations`` are
    what :func:`ail.judges.agreement.score_anchor` will pass to the judge when
    this label lands in the Human Anchor, so the judge can be re-run on the same
    item the human graded. ``expectations`` is also logged as MLflow
    *expectations* (ground truth) when present.
    """

    trace_id: str
    name: str
    value: ScoreValue
    rationale: str | None = None
    expectations: dict[str, Any] | None = None
    inputs: Any = None
    outputs: Any = None


def record_label(label: TraceLabel, *, labeler_id: str = DEFAULT_LABELER_ID) -> list[Assessment]:
    """Write one human label to its trace as MLflow assessment(s).

    Logs ``label.value`` as a **feedback** assessment (``mlflow.log_feedback``)
    with ``source_type=HUMAN`` and ``name=label.name``, so MemAlign and the
    agreement layer read it back as the human's grade. Each entry in
    ``label.expectations`` is additionally logged as an **expectation**
    (``mlflow.log_expectation``, also ``HUMAN``) — ground truth for judges that
    compare an output against an expected result.

    Assumes MLflow is already pointed at the tracking backend (a Databricks
    notebook, or after constructing the :class:`~ail.ingest.mlflow_source.MLflowTraceSource`
    used for pool assembly). Returns the assessments written.
    """
    import mlflow
    from mlflow.entities import AssessmentSource
    from mlflow.entities.assessment_source import AssessmentSourceType

    source = AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id=labeler_id)
    written: list[Assessment] = [
        mlflow.log_feedback(
            trace_id=label.trace_id,
            name=label.name,
            value=label.value,
            rationale=label.rationale,
            source=source,
        )
    ]
    for exp_name, exp_value in (label.expectations or {}).items():
        written.append(
            mlflow.log_expectation(
                trace_id=label.trace_id,
                name=exp_name,
                value=exp_value,
                source=source,
            )
        )
    return written


def record_labels(labels: Iterable[TraceLabel], *, labeler_id: str = DEFAULT_LABELER_ID) -> int:
    """Record many labels (see :func:`record_label`); return the assessment count."""
    return sum(len(record_label(label, labeler_id=labeler_id)) for label in labels)


def split_labels(
    labels: Iterable[TraceLabel],
    *,
    anchor_fraction: float = DEFAULT_ANCHOR_FRACTION,
    seed: int = 0,
) -> tuple[list[TraceLabel], list[TraceLabel]]:
    """Partition labels into ``(alignment_labels, anchor_labels)`` — disjoint by trace.

    The split is at the **trace** level, not the label level: every label of a
    given trace lands in the same pool, so a trace can never appear in both the
    Alignment Set and the Human Anchor (the leak the frozen wall forbids). The
    partition is deterministic for a given ``seed`` (a seeded shuffle of the
    sorted trace ids), so a labeling run is reproducible.

    With ``n`` distinct labeled traces, ``round(n * anchor_fraction)`` go to the
    anchor (at least 1 when ``n >= 2``), and the rest to the alignment set, so
    both pools are non-empty whenever there are at least two labeled traces.
    """
    if not 0.0 <= anchor_fraction <= 1.0:
        raise ValueError(f"anchor_fraction must be in [0, 1], got {anchor_fraction!r}")
    by_trace: dict[str, list[TraceLabel]] = {}
    for label in labels:
        by_trace.setdefault(label.trace_id, []).append(label)

    trace_ids = sorted(by_trace)
    n = len(trace_ids)
    if n == 0:
        return [], []
    rng = random.Random(seed)
    rng.shuffle(trace_ids)
    if n == 1:
        n_anchor = 0  # cannot make two non-empty disjoint pools from one trace
    else:
        n_anchor = min(max(1, round(n * anchor_fraction)), n - 1)
    anchor_ids = set(trace_ids[:n_anchor])

    alignment_labels = [
        lab for tid, group in by_trace.items() if tid not in anchor_ids for lab in group
    ]
    anchor_labels = [lab for tid in anchor_ids for lab in by_trace[tid]]
    return alignment_labels, anchor_labels


def to_alignment_set(
    source: TraceSource,
    labels: Iterable[TraceLabel],
    *,
    labeler_id: str = DEFAULT_LABELER_ID,
) -> AlignmentSet:
    """Fetch the raw MLflow traces for ``labels`` and wrap them as an AlignmentSet.

    Reuses the ingest seam (:class:`~ail.ingest.base.TraceSource`) to read each
    labeled trace and takes the **raw** MLflow ``Trace`` object
    (:attr:`ail.ingest.base.NormalizedTrace.raw`), because MemAlign's
    ``judge.align(traces=...)`` consumes MLflow traces, not the normalized
    projection. Crucially, MemAlign learns *only* from a trace's **human
    assessments** — it reads ``trace.info.assessments`` for ``HUMAN``-sourced
    feedback whose name matches the judge — and the raw object fetched back does
    **not** reliably carry those (the just-recorded feedback is dropped from the
    re-read trace), so alignment fails with "No valid feedback records found".

    This attaches the human assessment we already hold (in each
    :class:`TraceLabel`) onto its raw trace's ``info.assessments`` as a ``HUMAN``
    :class:`mlflow.entities.Feedback`, so the Alignment Set always carries the
    feedback MemAlign reads — independent of what the backend echoes back. Labels
    are grouped by ``trace_id`` (one fetch per trace, order preserved); a
    pre-existing human assessment of the same name is replaced by the label's, and
    other assessments are left untouched. Traces that cannot be fetched (or carry
    no raw object) are skipped.
    """
    by_trace: dict[str, list[TraceLabel]] = {}
    for label in labels:
        by_trace.setdefault(label.trace_id, []).append(label)

    raws: list[Any] = []
    for tid, group in by_trace.items():  # de-duped by trace, insertion order preserved
        normalized = source.get_trace(tid)
        if normalized is not None and normalized.raw is not None:
            _attach_human_assessments(normalized.raw, group, labeler_id=labeler_id)
            raws.append(normalized.raw)
    return AlignmentSet.of(raws)


def _attach_human_assessments(raw: Any, labels: Sequence[TraceLabel], *, labeler_id: str) -> None:
    """Attach each label's value to ``raw.info.assessments`` as a HUMAN feedback.

    This is the read shape MemAlign expects (``trace.info.assessments`` filtered
    to ``HUMAN`` source and the judge's name). A pre-existing human assessment of a
    name we are writing is dropped first so re-assembly is idempotent; assessments
    of other names (and non-human assessments) are preserved.
    """
    from mlflow.entities import AssessmentSource, Feedback
    from mlflow.entities.assessment_source import AssessmentSourceType

    info = getattr(raw, "info", None)
    if info is None:  # no place to carry assessments; nothing we can do for this trace
        return
    source = AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id=labeler_id)
    trace_id = getattr(info, "trace_id", None)
    written_names = {label.name for label in labels}
    feedbacks = [
        Feedback(
            name=label.name,
            value=label.value,
            rationale=label.rationale,
            source=source,
            trace_id=trace_id,
        )
        for label in labels
    ]
    existing = [
        assessment
        for assessment in (getattr(info, "assessments", None) or [])
        if not (_is_human(assessment) and getattr(assessment, "name", None) in written_names)
    ]
    info.assessments = existing + feedbacks


def _is_human(assessment: Any) -> bool:
    """Whether an assessment is HUMAN-sourced (best-effort, by source type name)."""
    source = getattr(assessment, "source", None)
    return str(getattr(source, "source_type", "")) == "HUMAN"


def to_human_anchor(
    labels: Iterable[TraceLabel],
    *,
    name: str | None = None,
    source: TraceSource | None = None,
) -> HumanAnchor:
    """Build a Human Anchor from labels (optionally filtered to one judge ``name``).

    Each label becomes an :class:`~ail.pools.AnchorItem` keyed by its
    ``trace_id`` with ``human_label=value`` and the ``inputs``/``outputs``/
    ``expectations`` to re-run the judge. The anchor is per-judge: pass ``name``
    to keep only that assessment's labels (so each trace contributes one item).
    A trace labeled twice for the same judge raises (duplicate ``item_id``), the
    same guard :class:`~ail.pools.HumanAnchor` already enforces.

    When ``source`` is given, each item also carries its **raw trace**, so a
    ``{{ trace }}``-based judge can be scored on the anchor (see
    :func:`ail.judges.agreement.score_anchor`). That trace is **blinded** first:
    its ``HUMAN`` assessments are stripped, so the human gold the agreement is
    measured against (which the live trace still carries, e.g. a label added in
    the MLflow UI) is never visible to the judge. The gold lives only on
    :attr:`~ail.pools.AnchorItem.human_label`. This is the inverse of the
    Alignment Set, whose traces deliberately *carry* their human assessments for
    MemAlign to learn from — measuring agreement on a trace the judge can read the
    answer off would be circular (``docs/ARCHITECTURE.md`` §2).
    """
    selected = [lab for lab in labels if name is None or lab.name == name]
    return HumanAnchor.of(
        AnchorItem(
            item_id=lab.trace_id,
            human_label=lab.value,
            inputs=lab.inputs,
            outputs=lab.outputs,
            expectations=lab.expectations,
            trace=_blind_anchor_trace(source, lab.trace_id) if source is not None else None,
        )
        for lab in selected
    )


def _blind_anchor_trace(source: TraceSource, trace_id: str) -> Any:
    """Raw MLflow trace for the anchor with its human gold stripped (``None`` if unfetchable).

    Fetches the trace and removes every ``HUMAN`` assessment, so the judge that
    scores the anchor cannot read the human label it is being graded against off
    ``trace.info.assessments``. Non-human (LLM-judge/AI) assessments are kept.
    """
    normalized = source.get_trace(trace_id)
    if normalized is None or normalized.raw is None:
        return None
    return _strip_human_assessments(normalized.raw)


def _strip_human_assessments(raw: Any) -> Any:
    """Return a copy of ``raw`` with its ``HUMAN`` assessments removed (else ``raw``).

    Shallow-copies the trace and its ``info`` so the source's object is left
    untouched, then reassigns ``info.assessments`` to the non-human subset. When
    there is nothing human to strip, the original is returned unchanged.
    """
    info = getattr(raw, "info", None)
    assessments = getattr(info, "assessments", None) if info is not None else None
    if not assessments:
        return raw
    kept = [assessment for assessment in assessments if not _is_human(assessment)]
    if len(kept) == len(assessments):
        return raw  # nothing human to strip; avoid a needless copy
    blinded = copy.copy(raw)
    blinded.info = copy.copy(info)
    blinded.info.assessments = kept
    return blinded


def assemble_pools(
    source: TraceSource,
    labels: Iterable[TraceLabel],
    *,
    judge_name: str | None = None,
    anchor_fraction: float = DEFAULT_ANCHOR_FRACTION,
    seed: int = 0,
    labeler_id: str = DEFAULT_LABELER_ID,
) -> tuple[AlignmentSet, HumanAnchor]:
    """Split labels into a disjoint ``(AlignmentSet, HumanAnchor)`` and prove it.

    Partitions ``labels`` by trace (:func:`split_labels`), fetches the alignment
    traces as raw MLflow traces **carrying their human assessments**
    (:func:`to_alignment_set`), builds the anchor (:func:`to_human_anchor`,
    filtered to ``judge_name`` when given, with each item carrying its raw trace),
    then calls :func:`~ail.pools.assert_pools_disjoint` to **prove** no trace id
    leaked across the two pools before returning — disjointness is guaranteed by
    construction and re-checked by the Pool-keyed wall.

    Args:
        source: Trace source used to fetch the alignment and anchor traces.
        labels: Human labels recorded on the slice (see :func:`record_labels`).
        judge_name: When set, the anchor keeps only this assessment's labels (a
            per-judge anchor). The Alignment Set holds all alignment-pool traces;
            each judge's ``align`` reads its own assessment name off them.
        anchor_fraction / seed: Forwarded to :func:`split_labels`.
        labeler_id: Recorded as the ``source_id`` of the human assessments
            attached to the Alignment Set's traces.

    Returns:
        ``(alignment_set, human_anchor)`` — disjoint by trace id.
    """
    alignment_labels, anchor_labels = split_labels(
        labels, anchor_fraction=anchor_fraction, seed=seed
    )
    alignment_set = to_alignment_set(source, alignment_labels, labeler_id=labeler_id)
    anchor = to_human_anchor(anchor_labels, name=judge_name, source=source)
    # Prove the wall: no trace id may appear in both pools.
    assert_pools_disjoint(alignment_set=alignment_set, human_anchor=anchor)
    return alignment_set, anchor
