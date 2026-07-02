"""L4 — the in-app labeling server side (``docs/LABELING_UI.md``).

The app is a thin AppKit (Node/React) app; its authenticated
``server/plugins/labeling`` routes resolve the labeler from the platform identity
headers and hand a JSON action to :mod:`ail.labeling.service` — the same bridge
shape as :mod:`ail.onboarding.service` and :mod:`ail.loop.apply_service` (a single
CLI reads a JSON action on stdin and prints a typed JSON result on stdout). L4 is
the human-facing surface that **produces the HUMAN labels** the scheduled L2
auto-align (:func:`ail.judges.auto_align.auto_align_scorers`) pairs to align each
judge.

The one condition that makes MemAlign ``align()`` work is that a submitted label is
written as a HUMAN assessment whose ``name`` **exactly matches** the target judge's
name. This package never reinvents that write or that naming: the ``label`` action
reuses :func:`ail.judges.labeling.record_label` (which logs a ``HUMAN``
``mlflow.log_feedback`` keyed by the judge name), and it offers labeling only along
the names of the **registered** judges, read from
:func:`ail.judges.registration.list_registered_scorers`. The label-floor progress
is the readiness floor (:attr:`ail.readiness.ReadinessThresholds.quality_min_labels`)
surfaced verbatim — never a number re-derived in TypeScript.
"""

from ail.labeling.service import (
    DEFAULT_SCAN_LIMIT,
    DEFAULT_WORKLIST_LIMIT,
    DimensionProgress,
    DimensionsResult,
    ErrorResult,
    LabelingOutcome,
    LabelInput,
    LabelResult,
    TraceTarget,
    apply_label,
    build_dimensions_state,
    main,
    run_action,
    run_dimensions,
    run_label,
)

__all__ = [
    "DEFAULT_SCAN_LIMIT",
    "DEFAULT_WORKLIST_LIMIT",
    "LabelingOutcome",
    "LabelInput",
    "DimensionProgress",
    "TraceTarget",
    "DimensionsResult",
    "LabelResult",
    "ErrorResult",
    "build_dimensions_state",
    "apply_label",
    "run_dimensions",
    "run_label",
    "run_action",
    "main",
]
