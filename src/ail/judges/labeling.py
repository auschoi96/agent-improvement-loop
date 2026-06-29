"""Record human labels on traces and assemble the two MemAlign-input pools.

MemAlign aligns a judge *against human labels* — but the reference experiment has
**zero** (``docs/ARCHITECTURE.md`` §8). This module is the missing input stage:
it lets a human record labels on a slice of real traces, and assembles those
labels into the two **disjoint** pools the L2 layer consumes:

* an :class:`~ail.judges.pools.AlignmentSet` — raw MLflow traces (carrying the
  human assessments) that :func:`ail.judges.alignment.align_judge` /
  :func:`ail.judges.registration.create_aligned_scorer` learn from; and
* a :class:`~ail.judges.pools.HumanAnchor` — held-out human labels that
  :func:`ail.judges.agreement.score_anchor` audits the aligned judge against.

The two pools are **never mixed**: :func:`split_labels` partitions by *trace*
(every label of a trace lands in one pool), and :func:`assemble_pools` re-proves
disjointness with :func:`~ail.judges.pools.assert_pools_disjoint` (the
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

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ail.judges.pools import (
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


def to_alignment_set(source: TraceSource, trace_ids: Sequence[str]) -> AlignmentSet:
    """Fetch the raw MLflow traces for ``trace_ids`` and wrap them as an AlignmentSet.

    Reuses the ingest seam (:class:`~ail.ingest.base.TraceSource`) to read each
    trace, then takes the **raw** MLflow ``Trace`` object
    (:attr:`ail.ingest.base.NormalizedTrace.raw`) because MemAlign's
    ``judge.align(traces=...)`` consumes MLflow traces (carrying their human
    assessments), not the normalized projection. Duplicate ids are de-duped;
    traces that cannot be fetched (or carry no raw object) are skipped.
    """
    raws: list[Any] = []
    for tid in dict.fromkeys(trace_ids):  # de-dupe, preserve order
        normalized = source.get_trace(tid)
        if normalized is not None and normalized.raw is not None:
            raws.append(normalized.raw)
    return AlignmentSet.of(raws)


def to_human_anchor(labels: Iterable[TraceLabel], *, name: str | None = None) -> HumanAnchor:
    """Build a Human Anchor from labels (optionally filtered to one judge ``name``).

    Each label becomes an :class:`~ail.judges.pools.AnchorItem` keyed by its
    ``trace_id`` with ``human_label=value`` and the ``inputs``/``outputs``/
    ``expectations`` to re-run the judge. The anchor is per-judge: pass ``name``
    to keep only that assessment's labels (so each trace contributes one item).
    A trace labeled twice for the same judge raises (duplicate ``item_id``), the
    same guard :class:`~ail.judges.pools.HumanAnchor` already enforces.
    """
    selected = [lab for lab in labels if name is None or lab.name == name]
    return HumanAnchor.of(
        AnchorItem(
            item_id=lab.trace_id,
            human_label=lab.value,
            inputs=lab.inputs,
            outputs=lab.outputs,
            expectations=lab.expectations,
        )
        for lab in selected
    )


def assemble_pools(
    source: TraceSource,
    labels: Iterable[TraceLabel],
    *,
    judge_name: str | None = None,
    anchor_fraction: float = DEFAULT_ANCHOR_FRACTION,
    seed: int = 0,
) -> tuple[AlignmentSet, HumanAnchor]:
    """Split labels into a disjoint ``(AlignmentSet, HumanAnchor)`` and prove it.

    Partitions ``labels`` by trace (:func:`split_labels`), fetches the alignment
    traces as raw MLflow traces (:func:`to_alignment_set`), builds the anchor
    (:func:`to_human_anchor`, filtered to ``judge_name`` when given), then calls
    :func:`~ail.judges.pools.assert_pools_disjoint` to **prove** no trace id
    leaked across the two pools before returning — disjointness is guaranteed by
    construction and re-checked by the Pool-keyed wall.

    Args:
        source: Trace source used to fetch the alignment traces.
        labels: Human labels recorded on the slice (see :func:`record_labels`).
        judge_name: When set, the anchor keeps only this assessment's labels (a
            per-judge anchor). The Alignment Set holds all alignment-pool traces;
            each judge's ``align`` reads its own assessment name off them.
        anchor_fraction / seed: Forwarded to :func:`split_labels`.

    Returns:
        ``(alignment_set, human_anchor)`` — disjoint by trace id.
    """
    alignment_labels, anchor_labels = split_labels(
        labels, anchor_fraction=anchor_fraction, seed=seed
    )
    alignment_set = to_alignment_set(source, [lab.trace_id for lab in alignment_labels])
    anchor = to_human_anchor(anchor_labels, name=judge_name)
    # Prove the wall: no trace id may appear in both pools.
    assert_pools_disjoint(alignment_set=alignment_set, human_anchor=anchor)
    return alignment_set, anchor
