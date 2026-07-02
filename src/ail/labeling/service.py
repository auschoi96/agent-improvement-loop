"""The labeling page's **server side** — the Python the app's authenticated routes
invoke (``docs/LABELING_UI.md``; L4, the last Phase-1 lane).

The thin AppKit (Node/React) ``server/plugins/labeling`` routes resolve the labeler
from the platform identity headers and hand a JSON action to this module — the same
bridge shape as :mod:`ail.onboarding.service` (a single CLI reads a JSON action on
stdin and prints a typed JSON result on stdout, invoked as
``python -m ail.labeling.service``). It does exactly two things, **reusing** the
existing L1/L2 helpers rather than reimplementing any:

* ``dimensions`` (read) — the set of **registered** judges (their exact names, from
  :func:`ail.judges.registration.list_registered_scorers`), each with its label
  progress toward the floor and the recent traces that still need a label for it.
* ``label`` (write) — record one HUMAN label on a trace by **reusing**
  :func:`ail.judges.labeling.record_label`, which logs an ``mlflow.log_feedback``
  with ``source_type=HUMAN`` and ``name`` **equal to the judge name**. That
  name-match is the single condition that makes MemAlign ``align()`` pair the label
  to the judge (a mismatch silently breaks alignment), so the write and its naming
  are never reinvented here — the ``label`` action refuses any name that is not a
  currently-registered judge.

**How it writes (grounded, not guessed):** the write goes through the MLflow Python
SDK inside this process (``ail.judges.labeling.record_label`` →
``mlflow.log_feedback``), invoked over a Python subprocess bridge — exactly how the
onboarding plugin performs its MLflow writes (``ail.onboarding.service`` via the
MLflow Python client). MLflow is pointed at the workspace the same way
:class:`ail.ingest.mlflow_source.MLflowTraceSource` /
:class:`ail.onboarding.experiment.MlflowExperimentClient` do it: tracking URI
``databricks``, registry URI ``databricks-uc``, the active CLI profile (or ambient
service-principal auth) for credentials. A deployed Node-only image would trigger a
Databricks Job instead (the analogue of the approvals ``jobTriggerApplyBridge``); it
is a documented follow-on and does not change the action contract below.

**Two-tier — no fabricated numbers.** The label floor is the readiness floor
(:attr:`ail.readiness.ReadinessThresholds.quality_min_labels`) surfaced verbatim; the
progress counts are computed here in Python. TypeScript renders them and never
re-derives or hardcodes the floor (the trap caught in the onboarding wizard).

**Fail-closed / no fabrication.** An empty labeler, a missing trace/name/value, a
name that is not a registered judge, an inability to determine the registered judges,
or any write failure (auth, permission, trace not found) all yield a
``refused``/``error`` result — never a fabricated ``labeled`` success. The labeler is
the **authenticated** identity the route passes; it is never trusted from the browser.

**Unit-testable with no live workspace.** The pure orchestration
(:func:`build_dimensions_state`, :func:`apply_label`) runs against injected fakes; the
live wiring (:func:`run_dimensions`, :func:`run_label`, :func:`main`) is a thin
composition on top.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Iterable
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from ail.judges.auto_align import read_human_labels
from ail.judges.labeling import TraceLabel, record_label
from ail.readiness import ReadinessThresholds

if TYPE_CHECKING:
    from ail.ingest.base import TraceSource

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

#: How many recent traces the read side scans to compute per-dimension label
#: progress and the "needs a label" worklist. Chosen well above the label floor so
#: the progress count is accurate during active labeling; when the experiment has
#: more traces the result reports ``scan_capped=True`` and counts reflect the most
#: recent traces (honest, and conservative — it never over-reports readiness).
DEFAULT_SCAN_LIMIT = 200

#: How many "needs a label" traces the worklist returns (a UI page, not the whole
#: corpus). Counts above are over the full scan; only the returned rows are capped.
DEFAULT_WORKLIST_LIMIT = 50

#: A recorder writes one label and returns whatever MLflow returns. The default is a
#: thin adapter over :func:`ail.judges.labeling.record_label`; tests inject a fake.
LabelRecorder = Callable[[TraceLabel, str], Any]


class LabelingOutcome(StrEnum):
    """The outcome the app surfaces for a labeling action."""

    DIMENSIONS = "dimensions"
    #: A HUMAN label was written to the trace (name-matched to a registered judge).
    LABELED = "labeled"
    #: A fail-closed decision-level refusal (empty labeler, unknown judge name,
    #: missing field) — nothing was written; surface :attr:`refused_reason`.
    REFUSED = "refused"
    #: An infrastructure / access error — never a fabricated success.
    ERROR = "error"


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LabelInput(_Contract):
    """The value control a dimension's label schema implies — a UI hint, best-effort.

    Derived from the judge's L1 label schema (the ``feedback`` schema whose name
    equals the judge name) so the labeler enters a value of the type the judge
    expects — see :func:`ail.judges.authoring.create_matching_label_schema`. ``kind`` is
    ``"numeric"`` (with ``min``/``max``), ``"pass_fail"`` (with ``positive``/``negative``
    labels), or ``"free"`` when the schema could not be read — the client then renders
    a free-form field. It never blocks labeling; it only shapes the input control.
    """

    kind: str
    min: float | None = None
    max: float | None = None
    positive: str | None = None
    negative: str | None = None


class DimensionProgress(_Contract):
    """One registered judge's name + its label progress toward the floor.

    ``label_floor`` is :attr:`ail.readiness.ReadinessThresholds.quality_min_labels`
    surfaced verbatim; ``labels_so_far`` is counted here from the scanned traces.
    ``summary`` is a Python-composed, source-of-truth line the client renders
    verbatim (the two-tier discipline — the floor number is never authored in TS).
    """

    name: str
    labels_so_far: int
    label_floor: int
    remaining: int
    complete: bool
    input: LabelInput | None = None
    summary: str = ""


class TraceTarget(_Contract):
    """One trace on the labeling worklist, with which dimensions it still lacks.

    ``labeled`` maps each registered judge name to whether this trace already carries
    a HUMAN label for it, so the UI shows exactly which dimensions remain. A trace is
    on the worklist only when it is missing at least one dimension's label.
    """

    trace_id: str
    request_time: str | None = None
    preview: str | None = None
    labeled: dict[str, bool] = Field(default_factory=dict)


class DimensionsResult(_Contract):
    """The labeling page's read result — registered dimensions + progress + worklist."""

    outcome: str = "dimensions"
    experiment_id: str
    label_floor: int
    dimensions: list[DimensionProgress] = Field(default_factory=list)
    traces: list[TraceTarget] = Field(default_factory=list)
    scanned: int = 0
    scan_capped: bool = False
    actor: str = ""
    #: A Python-composed one-line note (with the real floor number) rendered verbatim.
    summary: str = ""


class LabelResult(_Contract):
    """The result of writing (or refusing) one label."""

    outcome: LabelingOutcome
    experiment_id: str
    trace_id: str
    name: str
    value: Any = None
    labeler: str = ""
    #: Updated progress for the labeled dimension (best-effort re-read after a
    #: successful write); omitted when it could not be recomputed.
    labels_so_far: int | None = None
    label_floor: int | None = None
    remaining: int | None = None
    complete: bool | None = None
    refused_reason: str | None = None
    error: str | None = None


class ErrorResult(_Contract):
    """A dispatch-level error (unknown/malformed action)."""

    outcome: LabelingOutcome = LabelingOutcome.ERROR
    action: str = ""
    error: str


# ---------------------------------------------------------------------------
# Reading human labels off a raw trace (the L1 name-matching convention)
# ---------------------------------------------------------------------------


def _is_human(assessment: Any) -> bool:
    """Whether an assessment is ``HUMAN``-sourced (best-effort, by source-type name)."""
    source = getattr(assessment, "source", None)
    return str(getattr(source, "source_type", "")) == "HUMAN"


def _human_labeled_names(raw: Any) -> set[str]:
    """The set of judge names this trace already carries a ``HUMAN`` label for.

    Reads ``raw.info.assessments`` (the read shape the L2 layer uses) and keeps the
    names of the ``HUMAN``-sourced ones. Used to decide, per dimension, whether a
    trace still needs a label — never to fabricate one.
    """
    info = getattr(raw, "info", None)
    assessments = getattr(info, "assessments", None) if info is not None else None
    if not assessments:
        return set()
    return {
        name
        for assessment in assessments
        if _is_human(assessment) and (name := getattr(assessment, "name", None)) is not None
    }


# ---------------------------------------------------------------------------
# Pure orchestration (injected seams → unit-testable, no live workspace)
# ---------------------------------------------------------------------------


def _dimension_summary(name: str, labels_so_far: int, floor: int, remaining: int) -> str:
    """A source-of-truth progress line with the real floor number (rendered verbatim)."""
    if remaining <= 0:
        return (
            f"{labels_so_far} / {floor} human labels — floor met; the {name} judge "
            "has enough labels to auto-align."
        )
    return (
        f"{labels_so_far} / {floor} human labels — need {remaining} more to align the {name} judge."
    )


def _overall_summary(dimensions: list[DimensionProgress], floor: int) -> str:
    """The page-level note (with the real floor), composed in Python for verbatim render."""
    n = len(dimensions)
    if n == 0:
        return ""
    noun = "dimension" if n == 1 else "dimensions"
    return (
        f"Label traces along the {n} registered judged {noun}. Each needs {floor} human "
        "labels before its judge auto-aligns; your labels are written as HUMAN "
        "assessments named for the judge, which is what MemAlign pairs them by."
    )


def build_dimensions_state(
    *,
    experiment_id: str,
    judge_names: Iterable[str],
    source: TraceSource,
    label_floor: int,
    label_inputs: dict[str, LabelInput] | None = None,
    scan_limit: int | None = DEFAULT_SCAN_LIMIT,
    worklist_limit: int = DEFAULT_WORKLIST_LIMIT,
    actor: str = "",
) -> DimensionsResult:
    """Assemble the read result: per-dimension progress + the "needs a label" worklist.

    Scans the experiment's recent traces once (through the ingest seam), and for each
    registered judge counts how many scanned traces carry a HUMAN label named for it
    (progress toward ``label_floor``) and collects the traces still missing one. The
    dimensions offered are **exactly** ``judge_names`` (the registered judges) — no
    dimension is invented. All numbers are computed here; the client renders them.
    """
    names = list(dict.fromkeys(judge_names))
    inputs = label_inputs or {}

    counts: dict[str, int] = dict.fromkeys(names, 0)
    worklist: list[TraceTarget] = []
    scanned = 0
    for trace in source.iter_traces(
        experiment_id=experiment_id, max_results=scan_limit, order_by=["timestamp_ms DESC"]
    ):
        scanned += 1
        trace_id = getattr(trace, "trace_id", None)
        if not trace_id:
            continue
        labeled_names = _human_labeled_names(getattr(trace, "raw", None))
        labeled = {name: name in labeled_names for name in names}
        for name in names:
            if labeled[name]:
                counts[name] += 1
        if names and not all(labeled.values()) and len(worklist) < worklist_limit:
            worklist.append(
                TraceTarget(
                    trace_id=trace_id,
                    request_time=_iso(getattr(trace, "request_time", None)),
                    preview=_preview(trace),
                    labeled=labeled,
                )
            )

    dimensions: list[DimensionProgress] = []
    for name in names:
        so_far = counts[name]
        remaining = max(0, label_floor - so_far)
        dimensions.append(
            DimensionProgress(
                name=name,
                labels_so_far=so_far,
                label_floor=label_floor,
                remaining=remaining,
                complete=remaining <= 0,
                input=inputs.get(name),
                summary=_dimension_summary(name, so_far, label_floor, remaining),
            )
        )

    return DimensionsResult(
        experiment_id=experiment_id,
        label_floor=label_floor,
        dimensions=dimensions,
        traces=worklist,
        scanned=scanned,
        scan_capped=scan_limit is not None and scanned >= scan_limit,
        actor=actor,
        summary=_overall_summary(dimensions, label_floor),
    )


def apply_label(
    *,
    experiment_id: str,
    trace_id: str,
    name: str,
    value: Any,
    labeler: str,
    judge_names: Iterable[str],
    rationale: str | None = None,
    source: TraceSource | None = None,
    label_floor: int | None = None,
    record: LabelRecorder | None = None,
    scan_limit: int | None = DEFAULT_SCAN_LIMIT,
) -> LabelResult:
    """Write one HUMAN label — name-matched to a registered judge — or refuse.

    Fail-closed before writing: an empty labeler, a missing trace/name/value, or a
    ``name`` that is not among ``judge_names`` (the registered judges) all ``REFUSED``
    — nothing is written. The name guard is the load-bearing one: a label whose name
    is not a registered judge could never align, so it is refused rather than written.

    Otherwise it reuses :func:`ail.judges.labeling.record_label` (the sole, canonical
    writer — a ``HUMAN`` ``mlflow.log_feedback`` keyed by ``name``). A write failure
    (auth, permission, trace not found) is surfaced as an honest ``ERROR`` — never a
    fabricated ``labeled``. On success, best-effort re-reads the dimension's label
    count for updated progress (a count failure never turns a real write into a
    failure).
    """
    clean_name = name.strip()
    clean_trace = trace_id.strip()
    registered = set(dict.fromkeys(judge_names))

    if not labeler.strip():
        return _label_refused(
            experiment_id,
            clean_trace,
            clean_name,
            value,
            labeler,
            "refusing an anonymous label — no authenticated labeler identity",
        )
    if not clean_trace:
        return _label_refused(
            experiment_id, clean_trace, clean_name, value, labeler, "a trace id is required"
        )
    if not clean_name:
        return _label_refused(
            experiment_id, clean_trace, clean_name, value, labeler, "a dimension name is required"
        )
    if _is_empty(value):
        return _label_refused(
            experiment_id, clean_trace, clean_name, value, labeler, "a label value is required"
        )
    if clean_name not in registered:
        return _label_refused(
            experiment_id,
            clean_trace,
            clean_name,
            value,
            labeler,
            f"{clean_name!r} is not a registered judge — refusing to write a label that "
            "could never align (a label must be named for a registered judge)",
        )

    writer = record or _default_recorder
    label = TraceLabel(
        trace_id=clean_trace,
        name=clean_name,
        value=_clean_value(value),
        rationale=(rationale.strip() if isinstance(rationale, str) and rationale.strip() else None),
    )
    try:
        writer(label, labeler.strip())
    except Exception as exc:  # noqa: BLE001 - any write failure is an honest error, never a fake label
        return LabelResult(
            outcome=LabelingOutcome.ERROR,
            experiment_id=experiment_id,
            trace_id=clean_trace,
            name=clean_name,
            value=value,
            labeler=labeler,
            error=f"{type(exc).__name__}: {exc}",
        )

    result = LabelResult(
        outcome=LabelingOutcome.LABELED,
        experiment_id=experiment_id,
        trace_id=clean_trace,
        name=clean_name,
        value=_clean_value(value),
        labeler=labeler.strip(),
    )
    if source is not None and label_floor is not None:
        _attach_progress(result, source, experiment_id, clean_name, label_floor, scan_limit)
    return result


def _attach_progress(
    result: LabelResult,
    source: TraceSource,
    experiment_id: str,
    name: str,
    label_floor: int,
    scan_limit: int | None,
) -> None:
    """Best-effort: re-read the dimension's label count and set the progress fields.

    Reuses :func:`ail.judges.auto_align.read_human_labels` — the same read side the
    auto-align trigger uses — so the count matches what will actually gate alignment.
    A read failure leaves the write successful with progress unset (never demoted).
    """
    try:
        so_far = len(
            read_human_labels(
                source, experiment_id=experiment_id, judge_name=name, max_results=scan_limit
            )
        )
    except Exception:  # noqa: BLE001 - the write succeeded; a count read failure is not fatal
        return
    remaining = max(0, label_floor - so_far)
    result.labels_so_far = so_far
    result.label_floor = label_floor
    result.remaining = remaining
    result.complete = remaining <= 0


def _label_refused(
    experiment_id: str, trace_id: str, name: str, value: Any, labeler: str, reason: str
) -> LabelResult:
    return LabelResult(
        outcome=LabelingOutcome.REFUSED,
        experiment_id=experiment_id,
        trace_id=trace_id,
        name=name,
        value=value,
        labeler=labeler,
        refused_reason=reason,
    )


def _default_recorder(label: TraceLabel, labeler: str) -> Any:
    """Adapt :func:`ail.judges.labeling.record_label` to the :data:`LabelRecorder` shape."""
    return record_label(label, labeler_id=labeler)


def _is_empty(value: Any) -> bool:
    """Whether a label value counts as missing (``None`` or a blank string)."""
    if value is None:
        return True
    return isinstance(value, str) and not value.strip()


def _clean_value(value: Any) -> Any:
    """Trim a string value; pass numbers/booleans through unchanged."""
    return value.strip() if isinstance(value, str) else value


def _iso(value: Any) -> str | None:
    """ISO-8601 string for a datetime-like value, else its ``str`` (or ``None``)."""
    if value is None:
        return None
    isoformat = getattr(value, "isoformat", None)
    return isoformat() if callable(isoformat) else str(value)


def _preview(trace: Any) -> str | None:
    """A short request preview for the worklist row (from the normalized trace)."""
    preview = getattr(trace, "request_preview", None)
    if preview is None:
        return None
    text = str(preview)
    return text if len(text) <= 240 else text[:237] + "…"


# ---------------------------------------------------------------------------
# Live wiring — resolve the registered judges, configure MLflow, fail-closed
# ---------------------------------------------------------------------------


def _configure_mlflow(profile: str | None) -> None:
    """Point MLflow at the workspace, mirroring MLflowTraceSource / the onboarding client."""
    import os

    import mlflow

    if profile:
        os.environ.setdefault("DATABRICKS_CONFIG_PROFILE", profile)
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")


def _build_source(profile: str | None) -> TraceSource:
    """The live trace source for the read side (reuses the ingest seam)."""
    from ail.ingest.mlflow_source import MLflowTraceSource

    return MLflowTraceSource(profile=profile)


def _registered_judge_names(experiment_id: str, profile: str | None) -> list[str]:
    """The exact names of the judges currently registered on ``experiment_id``.

    Thin reuse of :func:`ail.judges.registration.list_registered_scorers`. Raises
    when the registered set cannot be determined (backend missing, permission), so
    the caller fails closed rather than inventing dimensions.
    """
    from ail.judges.registration import list_registered_scorers

    scorers = list_registered_scorers(experiment_id, profile=profile)
    return [s.name for s in scorers]


def _read_label_inputs(  # pragma: no cover - best-effort live MLflow read
    names: Iterable[str], experiment_id: str, profile: str | None
) -> dict[str, LabelInput]:
    """Best-effort per-judge value control from its L1 label schema (never fatal).

    Reads each judge's ``feedback`` label schema (name == judge name) and maps its
    input control to a :class:`LabelInput` hint. Any failure for a given name is
    swallowed — the dimension simply renders a free-form field.
    """
    out: dict[str, LabelInput] = {}
    try:
        from mlflow.genai.label_schemas import get_label_schema
    except Exception:  # noqa: BLE001 - label-schema API absent: every dimension is free-form
        return out
    for name in names:
        try:
            schema = get_label_schema(name)
        except Exception:  # noqa: BLE001 - unreadable schema for this name: free-form
            continue
        coerced = _coerce_label_input(getattr(schema, "input", None))
        if coerced is not None:
            out[name] = coerced
    return out


def _coerce_label_input(raw_input: Any) -> LabelInput | None:
    """Map an MLflow label-schema input control to a :class:`LabelInput` hint.

    Recognizes the two controls the L1 authoring path emits — numeric (``min_value``/
    ``max_value``) and pass/fail (``positive_label``/``negative_label``) — and returns
    ``kind="free"`` for anything else. ``None`` only when there is no input at all.
    """
    if raw_input is None:
        return None
    min_value = getattr(raw_input, "min_value", None)
    max_value = getattr(raw_input, "max_value", None)
    if min_value is not None or max_value is not None:
        return LabelInput(
            kind="numeric",
            min=float(min_value) if min_value is not None else None,
            max=float(max_value) if max_value is not None else None,
        )
    positive = getattr(raw_input, "positive_label", None)
    negative = getattr(raw_input, "negative_label", None)
    if positive is not None or negative is not None:
        return LabelInput(kind="pass_fail", positive=positive, negative=negative)
    return LabelInput(kind="free")


def run_dimensions(
    experiment_id: str,
    *,
    actor: str = "",
    profile: str | None = None,
    scan_limit: int | None = DEFAULT_SCAN_LIMIT,
    worklist_limit: int = DEFAULT_WORKLIST_LIMIT,
) -> DimensionsResult | ErrorResult:
    """Build the read result live (fail-closed: cannot determine judges → honest error)."""
    exp = experiment_id.strip()
    if not exp:
        return ErrorResult(action="dimensions", error="an experiment id is required")
    try:
        names = _registered_judge_names(exp, profile)
    except Exception as exc:  # noqa: BLE001 - do NOT invent dimensions when the set is unknown
        return ErrorResult(
            action="dimensions",
            error=(
                "cannot determine the registered judges for this experiment "
                f"({type(exc).__name__}: {exc}); refusing to invent labeling dimensions. "
                "Register at least one judge (ail.judges authoring) and ensure the app can "
                "list scorers (needs the databricks-agents backend and read access)."
            ),
        )
    try:
        _configure_mlflow(profile)
        source = _build_source(profile)
        inputs = _read_label_inputs(names, exp, profile)
        floor = ReadinessThresholds().quality_min_labels
        return build_dimensions_state(
            experiment_id=exp,
            judge_names=names,
            source=source,
            label_floor=floor,
            label_inputs=inputs,
            scan_limit=scan_limit,
            worklist_limit=worklist_limit,
            actor=actor,
        )
    except Exception as exc:  # noqa: BLE001 - any infra failure is an honest error
        return ErrorResult(action="dimensions", error=f"{type(exc).__name__}: {exc}")


def run_label(
    *,
    experiment_id: str,
    trace_id: str,
    name: str,
    value: Any,
    labeler: str,
    rationale: str | None = None,
    profile: str | None = None,
    scan_limit: int | None = DEFAULT_SCAN_LIMIT,
) -> LabelResult | ErrorResult:
    """Write one label live (fail-closed; a write failure is ERROR, never labeled)."""
    exp = experiment_id.strip()
    if not labeler.strip():
        return _label_refused(
            exp,
            trace_id.strip(),
            name.strip(),
            value,
            labeler,
            "refusing an anonymous label — no authenticated labeler identity",
        )
    if not exp:
        return ErrorResult(action="label", error="an experiment id is required")
    try:
        names = _registered_judge_names(exp, profile)
    except Exception as exc:  # noqa: BLE001 - cannot validate the name-match → fail closed
        return ErrorResult(
            action="label",
            error=(
                "cannot determine the registered judges to validate the label name "
                f"({type(exc).__name__}: {exc}); refusing to write a label whose name "
                "cannot be confirmed to match a registered judge."
            ),
        )
    try:
        _configure_mlflow(profile)
        source = _build_source(profile)
        floor = ReadinessThresholds().quality_min_labels
        return apply_label(
            experiment_id=exp,
            trace_id=trace_id,
            name=name,
            value=value,
            labeler=labeler,
            judge_names=names,
            rationale=rationale,
            source=source,
            label_floor=floor,
            scan_limit=scan_limit,
        )
    except Exception as exc:  # noqa: BLE001 - any infra failure is an honest error, never labeled
        return LabelResult(
            outcome=LabelingOutcome.ERROR,
            experiment_id=exp,
            trace_id=trace_id.strip(),
            name=name.strip(),
            value=value,
            labeler=labeler.strip(),
            error=f"{type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
# Dispatch + CLI (the bridge invokes `python -m ail.labeling.service`)
# ---------------------------------------------------------------------------


def run_action(payload: dict[str, Any]) -> BaseModel:
    """Dispatch one JSON action to its live handler; unknown/malformed → ERROR.

    ``actor`` is the AUTHENTICATED labeler the route injects; it — never a client-
    supplied ``labeler`` — is used as the label's source. Any ``labeler`` in the
    payload is ignored.
    """
    action = str(payload.get("action") or "")
    actor = str(payload.get("actor") or "")
    profile = payload.get("profile")
    experiment_id = str(payload.get("experiment_id") or "")

    if action == "dimensions":
        return run_dimensions(experiment_id, actor=actor, profile=profile)
    if action == "label":
        rationale = payload.get("rationale")
        return run_label(
            experiment_id=experiment_id,
            trace_id=str(payload.get("trace_id") or ""),
            name=str(payload.get("name") or ""),
            value=payload.get("value"),
            labeler=actor,
            rationale=str(rationale) if isinstance(rationale, str) else None,
            profile=profile,
        )
    return ErrorResult(action=action, error=f"unknown labeling action {action!r}")


def main(argv: list[str] | None = None) -> int:
    """CLI bridge: read a JSON action on stdin, print a JSON result on stdout.

    The Node/AppKit labeling route (which authenticates the labeler and injects it as
    ``actor`` — never trusted from the browser) invokes this as a subprocess. Always
    prints a parseable result and returns ``0`` for an action-level outcome (including
    a fail-closed REFUSED/ERROR); returns non-zero only when stdin is itself
    unparseable.
    """
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw) if raw.strip() else {}
    except ValueError as exc:
        print(json.dumps({"outcome": "error", "error": f"unparseable stdin: {exc}"}))
        return 2
    if not isinstance(payload, dict):
        print(json.dumps({"outcome": "error", "error": "stdin must be a JSON object"}))
        return 2
    result = run_action(payload)
    print(result.model_dump_json())
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
