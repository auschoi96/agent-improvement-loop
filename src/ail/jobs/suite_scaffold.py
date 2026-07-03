"""``ail-suite-scaffold`` — draft YOUR OWN Task Suite from YOUR OWN traces.

A stranger with a live experiment runs ``ail-suite-scaffold --experiment <id>`` and
gets a **draft** Task Suite: N representative traces, stratified across the L0
distribution, each turned into a :class:`~ail.task_suite.schema.Task` whose prompt
is derived from the trace. It is the on-ramp to the frozen evaluation wall
(``docs/ARCHITECTURE.md`` §2) — the counterpart of the hand-curated
:mod:`ail.task_suite.seed` — but built from *their* corpus instead of the seed's.

It is a thin **driver** that reuses, and never reimplements, the existing
machinery:

* Traces come from the ingest seam
  (:class:`ail.ingest.mlflow_source.MLflowTraceSource`, i.e. ``mlflow.search_traces``),
  the same short read-only path :mod:`ail.jobs.readiness_preflight` uses — so it
  takes a ``--profile`` (this is a brief read, not a long-lived writer like
  :mod:`ail.jobs.companion_planner`, which must refuse OAuth).
* The stratification axes are the **L0 metrics** (:mod:`ail.metrics.l0_deterministic`):
  token count and tool-call volume. We pick a rank-quantile spread across the token
  distribution — a stratified sample, not the top-N — so the draft covers the heavy
  tail *and* the typical short session.
* The draft is written with the existing loader
  (:func:`ail.task_suite.save_task_suite`), unfrozen.

**The anti-fake-suite discipline (same as GRP ground truth).** This tool does
**not** fabricate a success check. The :class:`~ail.task_suite.schema.Task` schema
carries no success-check field (it is ``frozen`` / ``extra="forbid"``), and — like
the runnable ``phase2-mini`` suite, whose checks live out-of-band in
``eval/phase2_fixtures/<task_id>/verify/`` keyed by ``task_id`` — the checks here
live in a **companion ``checks.yaml``** keyed by ``task_id``, beside ``tasks.yaml``.
Scaffold writes every check as the :data:`PLACEHOLDER_CHECK` string; a **human must
replace each one** with a real deterministic check. This tool never freezes — that
is :mod:`ail.jobs.suite_freeze`, which fails closed on any placeholder that remains.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import yaml

from ail.ingest.base import NormalizedTrace, TraceSource
from ail.metrics.contract import TraceMetrics
from ail.metrics.l0_deterministic import compute_l0, compute_trace_metrics
from ail.task_suite import (
    Difficulty,
    Task,
    TaskCategory,
    TaskSuite,
    save_task_suite,
    task_suite_path,
)

__all__ = [
    "PLACEHOLDER_CHECK",
    "CHECKS_FILENAME",
    "DEFAULT_COUNT",
    "DEFAULT_VERSION",
    "ScaffoldError",
    "ScaffoldAccessError",
    "SuitePathError",
    "select_representative",
    "build_draft_suite",
    "write_draft",
    "load_checks",
    "root_version_from_tasks_path",
    "scaffold",
    "main",
]

#: The required placeholder written for every task's success check. A human MUST
#: replace each one before freezing; :mod:`ail.jobs.suite_freeze` refuses to freeze
#: while any check still equals this (or is empty). This is the guard that forces
#: human authoring — the tool never fabricates a real check.
PLACEHOLDER_CHECK = "TODO: author a deterministic success check"

#: The companion file (beside ``tasks.yaml``) that holds ``{task_id: check}``. The
#: :class:`~ail.task_suite.schema.Task` schema has no success-check field, so — as
#: with the ``phase2-mini`` fixtures' out-of-band ``verify/`` checks — the checks
#: are keyed to ``task_id`` in a sibling file, the human's authoring surface.
CHECKS_FILENAME = "checks.yaml"

#: Default number of representative tasks to draft.
DEFAULT_COUNT = 8

#: Default artifact-directory / suite content label for a scaffolded draft.
DEFAULT_VERSION = "draft"

#: A trace whose exact-signature redundancy is at least this fraction is labelled
#: :attr:`~ail.task_suite.schema.TaskCategory.REPEATED_TARGET_BOILERPLATE`. Coarse,
#: like ``difficulty`` — a label on the drafted task, never a fabricated check.
_REDUNDANCY_BOILERPLATE_THRESHOLD = 0.3


class ScaffoldError(RuntimeError):
    """Base class for scaffold errors."""


class ScaffoldAccessError(ScaffoldError):
    """Could not read the experiment's traces (auth / permission / no traces).

    Carries an actionable message naming the profile and warehouse, mirroring
    :class:`ail.jobs.readiness_preflight.PreflightAccessError`. :func:`main` turns
    it into a non-zero exit — never a fabricated draft on error.
    """


class SuitePathError(ScaffoldError, ValueError):
    """The tasks.yaml path is not under ``eval/task_suite/<version>/tasks.yaml``.

    That layout is required so the write/reload path can reuse the existing loader
    (:func:`ail.task_suite.save_task_suite` / :func:`ail.task_suite.load_task_suite`),
    which computes the artifact path from ``(root, version)``.
    """


# ---------------------------------------------------------------------------
# Path helpers (reuse the loader's ``eval/task_suite/<version>/tasks.yaml`` layout)
# ---------------------------------------------------------------------------


def root_version_from_tasks_path(path: str | os.PathLike[str]) -> tuple[Path, str]:
    """Split a ``.../eval/task_suite/<version>/tasks.yaml`` path into ``(root, version)``.

    The inverse of :func:`ail.task_suite.task_suite_path`, so the same
    ``(root, version)`` round-trips through the existing loader. Raises
    :class:`SuitePathError` if the path does not match that layout.
    """
    p = Path(path).resolve()
    if p.name != "tasks.yaml":
        raise SuitePathError(f"expected a path ending in 'tasks.yaml', got {p}")
    version_dir = p.parent
    if version_dir.parent.name != "task_suite" or version_dir.parent.parent.name != "eval":
        raise SuitePathError(
            f"tasks.yaml must live at 'eval/task_suite/<version>/tasks.yaml' (so the "
            f"existing loader can read/write it), got {p}"
        )
    root = version_dir.parent.parent.parent
    return root, version_dir.name


def _checks_path(tasks_path: str | os.PathLike[str]) -> Path:
    """The companion ``checks.yaml`` beside a ``tasks.yaml``."""
    return Path(tasks_path).parent / CHECKS_FILENAME


# ---------------------------------------------------------------------------
# Stratified selection over the L0 distribution
# ---------------------------------------------------------------------------


def select_representative(traces: Sequence[NormalizedTrace], count: int) -> list[NormalizedTrace]:
    """Pick ``count`` traces spread across the token distribution (a stratified sample).

    Sorts by ``(total_tokens, trace_id)`` and takes rank-quantile midpoints, so the
    selection spans the heavy tail *and* the low median rather than clustering on the
    largest sessions (which a top-N would). Deterministic (no randomness). Returns all
    traces when there are at most ``count`` of them.
    """
    if count <= 0:
        return []
    ordered = sorted(traces, key=lambda t: (t.total_tokens, t.trace_id))
    m = len(ordered)
    if m <= count:
        return list(ordered)
    picked: list[NormalizedTrace] = []
    seen: set[int] = set()
    for i in range(count):
        idx = min(int((i + 0.5) * m / count), m - 1)
        while idx in seen and idx < m - 1:
            idx += 1
        if idx in seen:
            continue
        seen.add(idx)
        picked.append(ordered[idx])
    return picked


def _p90(values: Sequence[int]) -> float:
    """Linear-interpolated 90th percentile (matches the L0 report's percentile)."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    pos = 0.9 * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


# ---------------------------------------------------------------------------
# Trace -> draft Task (prompt derived from the trace; check is a placeholder)
# ---------------------------------------------------------------------------


def _categorize(metrics: TraceMetrics, token_p90: float, toolcall_p90: float) -> TaskCategory:
    """Coarse L0 label for a trace (a label on the task, never a fabricated check)."""
    if token_p90 > 0 and metrics.tokens.total_tokens >= token_p90:
        return TaskCategory.HEAVY_TAIL_HIGH_TOKEN
    if toolcall_p90 > 0 and metrics.total_tool_calls >= toolcall_p90:
        return TaskCategory.HIGH_TOOL_CALL_VOLUME
    if metrics.redundancy.redundancy_rate >= _REDUNDANCY_BOILERPLATE_THRESHOLD:
        return TaskCategory.REPEATED_TARGET_BOILERPLATE
    return TaskCategory.TYPICAL_SHORT_SESSION


def _difficulty(total_tokens: int, median: float, p90: float) -> Difficulty:
    """Coarse difficulty from the session's token magnitude (as :mod:`seed` does)."""
    if p90 > 0 and total_tokens >= p90:
        return Difficulty.HARD
    if median > 0 and total_tokens >= median:
        return Difficulty.MEDIUM
    return Difficulty.EASY


def _top_tools(metrics: TraceMetrics, limit: int = 3) -> str:
    """A short "Bash x35, Read x6" summary of the trace's tool mix."""
    ranked = sorted(metrics.tool_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ", ".join(f"{name} x{count}" for name, count in ranked[:limit]) or "no tool calls"


def _reconstructed_prompt(trace: NormalizedTrace, metrics: TraceMetrics) -> str:
    """A prompt derived from the observable L0 profile when no request preview exists.

    Grounded in the trace (magnitude + tool mix) and flagged as needing enrichment —
    the same discipline as :mod:`ail.task_suite.seed`. This derives the *prompt*; the
    success *check* is still an unauthored placeholder the human must write.
    """
    return (
        f"Reproduce the task this session performed "
        f"(~{metrics.tokens.total_tokens:,} tokens, {metrics.total_tool_calls} tool calls; "
        f"tools: {_top_tools(metrics)}). No request preview was recorded on the trace — "
        f"enrich this prompt from the source trace before relying on it."
    )


def _provenance_notes(trace: NormalizedTrace, metrics: TraceMetrics, *, from_preview: bool) -> str:
    """Provenance for a drafted task: L0 profile + how the prompt was derived."""
    origin = (
        "prompt is the trace's request preview"
        if from_preview
        else "prompt is a reconstruction from the L0 profile (no preview) — enrich before use"
    )
    return (
        f"Draft scaffolded from trace {trace.trace_id}: "
        f"{metrics.tokens.total_tokens:,} tokens, {metrics.total_tool_calls} tool calls "
        f"({_top_tools(metrics)}); {origin}. "
        f"Success check must be authored in {CHECKS_FILENAME} before freezing."
    )


def build_draft_suite(
    traces: Sequence[NormalizedTrace],
    *,
    count: int = DEFAULT_COUNT,
    version: str = DEFAULT_VERSION,
    created_at: str | None = None,
) -> tuple[TaskSuite, dict[str, str]]:
    """Build an unfrozen draft suite + its placeholder checks from ``traces``.

    Returns ``(suite, checks)`` where ``suite`` is a ``frozen=False``
    :class:`~ail.task_suite.schema.TaskSuite` and ``checks`` maps every drafted
    ``task_id`` to :data:`PLACEHOLDER_CHECK`. Pure (no I/O, no MLflow) so it is
    directly unit-testable. Never fabricates a success check.
    """
    report = compute_l0(traces)
    stats = report.aggregate.token_stats
    toolcall_p90 = _p90([t.total_tool_calls for t in traces])

    tasks: list[Task] = []
    checks: dict[str, str] = {}
    for i, trace in enumerate(select_representative(traces, count), start=1):
        metrics = compute_trace_metrics(trace)
        task_id = f"ts-draft-{i:03d}"
        preview = (trace.request_preview or "").strip()
        prompt = preview or _reconstructed_prompt(trace, metrics)
        tasks.append(
            Task(
                task_id=task_id,
                prompt=prompt,
                category=_categorize(metrics, stats.p90, toolcall_p90),
                source_trace_id=trace.trace_id,
                difficulty=_difficulty(metrics.tokens.total_tokens, stats.median, stats.p90),
                notes=_provenance_notes(trace, metrics, from_preview=bool(preview)),
            )
        )
        checks[task_id] = PLACEHOLDER_CHECK

    suite = TaskSuite(version=version, created_at=created_at, tasks=tuple(tasks))
    return suite, checks


# ---------------------------------------------------------------------------
# checks.yaml (the human authoring surface)
# ---------------------------------------------------------------------------

_CHECKS_HEADER = (
    "# Success checks for the DRAFT Task Suite, keyed by task_id.\n"
    "# Replace each placeholder below with a REAL deterministic success check\n"
    "# BEFORE running `ail-suite-freeze` — freeze refuses any remaining placeholder.\n"
)


def _dump_checks(checks: dict[str, str]) -> str:
    """Serialize ``{task_id: check}`` to the commented companion-file form."""
    body = yaml.safe_dump({"checks": checks}, sort_keys=False, allow_unicode=True, width=100)
    return _CHECKS_HEADER + body


def load_checks(tasks_path: str | os.PathLike[str]) -> dict[str, str]:
    """Read ``{task_id: check}`` from the companion ``checks.yaml`` (``{}`` if absent)."""
    path = _checks_path(tasks_path)
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw = data.get("checks") if isinstance(data, dict) else None
    if not isinstance(raw, dict):
        return {}
    return {str(k): ("" if v is None else str(v)) for k, v in raw.items()}


def write_draft(
    tasks_path: str | os.PathLike[str],
    suite: TaskSuite,
    checks: dict[str, str],
    *,
    force: bool = False,
) -> Path:
    """Write the draft ``tasks.yaml`` (via the loader) and its companion ``checks.yaml``.

    Reuses :func:`ail.task_suite.save_task_suite`, which refuses to clobber an
    existing **frozen** artifact unless ``force`` is set. Returns the tasks.yaml path.
    """
    root, artifact_version = root_version_from_tasks_path(tasks_path)
    path = save_task_suite(suite, artifact_version, root=root, overwrite=force)
    _checks_path(path).write_text(_dump_checks(checks), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Live trace fetch (injected away in tests)
# ---------------------------------------------------------------------------


def _access_hint(
    experiment_id: str, profile: str | None, warehouse_id: str | None, exc: Exception
) -> str:
    """Actionable message for a trace-store access failure (mirrors the preflight)."""
    prof = profile or os.environ.get("DATABRICKS_CONFIG_PROFILE") or "(default/ambient)"
    wh = warehouse_id or os.environ.get("AIL_WAREHOUSE_ID") or "(none supplied)"
    return (
        f"could not read traces for experiment {experiment_id!r} "
        f"(profile={prof}, warehouse_id={wh}): {type(exc).__name__}: {exc}. "
        "Check the Databricks profile points at the right workspace and that the identity "
        "has CAN_USE on the SQL warehouse backing the UC trace store (and CAN_VIEW on the "
        "experiment). No draft was written."
    )


def _fetch_traces(
    experiment_id: str,
    *,
    profile: str | None,
    warehouse_id: str | None,
    source: TraceSource | None,
    max_results: int | None,
) -> list[NormalizedTrace]:
    """Read normalized traces for an experiment; wrap any failure as an access error."""
    if warehouse_id:
        os.environ.setdefault("AIL_WAREHOUSE_ID", warehouse_id)
    if source is None:
        from ail.ingest.mlflow_source import MLflowTraceSource

        source = MLflowTraceSource(profile=profile)
    try:
        return source.fetch_traces(experiment_id=experiment_id, max_results=max_results)
    except Exception as exc:  # noqa: BLE001 - any read failure becomes an actionable error
        raise ScaffoldAccessError(_access_hint(experiment_id, profile, warehouse_id, exc)) from exc


def scaffold(
    experiment_id: str,
    *,
    out: str | os.PathLike[str],
    count: int = DEFAULT_COUNT,
    version: str = DEFAULT_VERSION,
    profile: str | None = None,
    warehouse_id: str | None = None,
    source: TraceSource | None = None,
    max_results: int | None = None,
    created_at: str | None = None,
    force: bool = False,
) -> Path:
    """Draft a suite from an experiment's traces and write it (+ its checks) to ``out``.

    Raises :class:`ScaffoldAccessError` if the traces cannot be read or the experiment
    yields none. Returns the written tasks.yaml path.
    """
    traces = _fetch_traces(
        experiment_id,
        profile=profile,
        warehouse_id=warehouse_id,
        source=source,
        max_results=max_results,
    )
    if not traces:
        raise ScaffoldAccessError(
            f"experiment {experiment_id!r} returned no traces; nothing to draft. "
            "Confirm the experiment id and that it has recorded traces."
        )
    suite, checks = build_draft_suite(traces, count=count, version=version, created_at=created_at)
    return write_draft(out, suite, checks, force=force)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ail-suite-scaffold",
        description=(
            "Draft a Task Suite from an experiment's traces: stratify by L0 (token/tool-call "
            "distribution), derive a prompt per representative trace, and write an UNFROZEN "
            "draft plus a companion checks.yaml of placeholders a human must author. Never "
            "fabricates a success check; never freezes (see ail-suite-freeze)."
        ),
    )
    parser.add_argument("--experiment", required=True, help="MLflow experiment id to read.")
    parser.add_argument(
        "--count", type=int, default=DEFAULT_COUNT, help="Number of representative tasks to draft."
    )
    parser.add_argument(
        "--version",
        default=DEFAULT_VERSION,
        help="Suite content label and default artifact-dir name.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Path to write tasks.yaml (default: eval/task_suite/<version>/tasks.yaml).",
    )
    parser.add_argument(
        "--profile",
        default=os.environ.get("DATABRICKS_CONFIG_PROFILE"),
        help="Databricks CLI profile selecting the workspace (short read-only trace read).",
    )
    parser.add_argument(
        "--warehouse-id",
        default=os.environ.get("AIL_WAREHOUSE_ID"),
        help="SQL warehouse backing the UC trace store (named in access errors).",
    )
    parser.add_argument(
        "--max-results", type=int, default=None, help="Cap traces scanned (default: all)."
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite an existing suite at the output path."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out = args.out or task_suite_path(args.version)
    try:
        path = scaffold(
            args.experiment,
            out=out,
            count=args.count,
            version=args.version,
            profile=args.profile,
            warehouse_id=args.warehouse_id,
            max_results=args.max_results,
            created_at=datetime.now(UTC).isoformat(),
            force=args.force,
        )
    except ScaffoldAccessError as exc:
        print(f"[ail-suite-scaffold] {exc}", file=sys.stderr)
        return 1
    except SuitePathError as exc:
        print(f"[ail-suite-scaffold] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - e.g. refusing to clobber a frozen suite
        print(
            f"[ail-suite-scaffold] refusing to write {out}: {type(exc).__name__}: {exc} "
            "(pass --force to overwrite).",
            file=sys.stderr,
        )
        return 2

    checks = _checks_path(path)
    print(
        f"[ail-suite-scaffold] wrote DRAFT suite ({args.count} tasks requested) to {path}\n"
        f"  authoring surface: {checks}\n"
        f"  NEXT: replace every '{PLACEHOLDER_CHECK}' with a real deterministic check, then run "
        f"`ail-suite-freeze {path}`."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
