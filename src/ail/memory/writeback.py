"""Write distilled memory to ``agent_memory`` — the custom ``@tool`` the agent
calls, the escaped-INSERT builder it runs, and the validation + provenance gate
that stands between model output and the governed table.

Mirrors the reference agent's ``submit_findings``: a single custom tool the Claude
Agent SDK exposes so the model, after distilling, hands back structured rows that
are INSERTed via the SQL Statement API (here the SDK's
``statement_execution`` seam, :func:`ail.publish._execute`). Two things the memory
store adds on top of the reference:

* **Validation** — a candidate must have a non-empty guideline, a 0–1 score, a
  known signal, and at least one ``source_trace_id`` (no provenance ⇒ no memory).
* **The provenance wall** — :func:`ail.memory.provenance.partition_rows` drops any
  surviving row whose ``source_trace_ids`` touch the frozen pools. Both run
  **inside** the tool, before any INSERT, so nothing eval-derived can reach the
  table even if the model asks for it. Dropped/invalid rows are recorded (never
  written) so the run reports exactly what it refused.

Inserts are escaped (via :func:`ail.publish._lit`, which doubles quotes and
backslashes), not raw-interpolated — the same discipline as the reference.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ail.memory.provenance import DroppedRow, ReservedPools, partition_rows
from ail.memory.schema import MEMORY_COLUMNS, MEMORY_TABLE, MemoryRow
from ail.publish import _execute, _lit

#: A source_signal must name one of the two feedback families the distiller reads.
_VALID_SIGNAL_PREFIXES = ("rlm", "judge:")


# ---------------------------------------------------------------------------
# Escaped INSERT builder
# ---------------------------------------------------------------------------


def _array_literal(values: Sequence[Any] | None) -> str:
    """``ARRAY('a', 'b')`` (escaped) or ``NULL`` for a nullable array column."""
    if values is None:
        return "NULL"
    return "ARRAY(" + ", ".join(_lit(v) for v in values) + ")"


def _row_values(row: MemoryRow) -> str:
    """One ``(...)`` VALUES tuple for ``row``, columns in :data:`MEMORY_COLUMNS` order."""
    rendered = [
        _lit(row.memory_id),
        _lit(row.cohort),
        _lit(row.category),
        _lit(row.guideline_text),
        _lit(row.score),
        _array_literal(row.source_trace_ids),
        _lit(row.source_signal),
        _lit(row.created_at),
        _array_literal(row.embedding),
    ]
    return "(" + ", ".join(rendered) + ")"


def build_memory_insert(catalog: str, schema: str, rows: Sequence[MemoryRow]) -> str:
    """The escaped ``INSERT INTO agent_memory`` for ``rows`` (must be non-empty).

    Column list and order come from :data:`MEMORY_COLUMNS`; every value is rendered
    through :func:`ail.publish._lit` / :func:`_array_literal` so quotes, backslashes,
    and array elements are escaped, never interpolated raw.
    """
    if not rows:
        raise ValueError("build_memory_insert requires at least one row")
    fqn = f"`{catalog}`.`{schema}`.{MEMORY_TABLE}"
    cols = ", ".join(MEMORY_COLUMNS)
    values = ",\n".join(_row_values(r) for r in rows)
    return f"INSERT INTO {fqn} ({cols}) VALUES\n{values}"


def insert_memory_rows(
    execute: Callable[[str], None],
    catalog: str,
    schema: str,
    rows: Sequence[MemoryRow],
) -> int:
    """Run :func:`build_memory_insert` through ``execute``; return rows written."""
    if not rows:
        return 0
    execute(build_memory_insert(catalog, schema, rows))
    return len(rows)


# ---------------------------------------------------------------------------
# Candidate validation (model output -> typed rows, or a drop reason)
# ---------------------------------------------------------------------------


def _validate_candidate(
    candidate: Any,
    *,
    cohort: str,
    created_at: str,
    id_factory: Callable[[], str],
) -> MemoryRow | str:
    """Coerce one raw candidate dict to a :class:`MemoryRow`, or return a reason string."""
    if not isinstance(candidate, dict):
        return f"candidate is not an object: {type(candidate).__name__}"

    category = str(candidate.get("category", "")).strip()
    guideline = str(candidate.get("guideline_text", "")).strip()
    signal = str(candidate.get("source_signal", "")).strip()
    if not category:
        return "missing category"
    if not guideline:
        return "missing guideline_text"
    if not signal.startswith(_VALID_SIGNAL_PREFIXES):
        return f"invalid source_signal {signal!r} (expected 'rlm' or 'judge:<name>')"

    raw_ids = candidate.get("source_trace_ids") or []
    if isinstance(raw_ids, str):
        raw_ids = [raw_ids]
    trace_ids = tuple(str(t).strip() for t in raw_ids if str(t).strip())
    if not trace_ids:
        return "missing source_trace_ids (a memory row must cite its provenance)"

    raw_score = candidate.get("score")
    if not isinstance(raw_score, (int, float, str)) or isinstance(raw_score, bool):
        return f"score not a number: {raw_score!r}"
    try:
        score = float(raw_score)
    except ValueError:
        return f"score not a number: {raw_score!r}"
    if not 0.0 <= score <= 1.0:
        return f"score {score} out of range [0, 1]"

    return MemoryRow(
        memory_id=id_factory(),
        cohort=cohort,
        category=category,
        guideline_text=guideline,
        score=score,
        source_trace_ids=trace_ids,
        source_signal=signal,
        created_at=created_at,
    )


@dataclass(frozen=True, slots=True)
class PreparedRows:
    """Validated candidates split into typed rows and per-candidate rejections."""

    valid: tuple[MemoryRow, ...]
    invalid: tuple[tuple[Any, str], ...]  # (raw candidate, reason)


def prepare_memory_rows(
    candidates: Iterable[Any],
    *,
    cohort: str,
    created_at: str,
    id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
) -> PreparedRows:
    """Validate raw candidate dicts into typed rows (+ rejections), no provenance yet."""
    valid: list[MemoryRow] = []
    invalid: list[tuple[Any, str]] = []
    for candidate in candidates:
        result = _validate_candidate(
            candidate, cohort=cohort, created_at=created_at, id_factory=id_factory
        )
        if isinstance(result, MemoryRow):
            valid.append(result)
        else:
            invalid.append((candidate, result))
    return PreparedRows(valid=tuple(valid), invalid=tuple(invalid))


# ---------------------------------------------------------------------------
# The submit_memory tool + its running tally
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class WriteTally:
    """Accumulates outcomes across every ``submit_memory`` call in one run."""

    written: int = 0
    dropped_provenance: list[DroppedRow] = field(default_factory=list)
    invalid: list[tuple[Any, str]] = field(default_factory=list)
    written_rows: list[MemoryRow] = field(default_factory=list)


def apply_and_write(
    candidates: Iterable[Any],
    *,
    execute: Callable[[str], None],
    catalog: str,
    schema: str,
    cohort: str,
    created_at: str,
    reserved: ReservedPools,
    tally: WriteTally,
    id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
) -> str:
    """Validate → wall → INSERT one batch of candidates; update ``tally``; return a summary.

    The pure heart of the ``submit_memory`` tool, split out so the write path is
    fully testable without the Claude Agent SDK or a live model. Order is
    load-bearing: validation and the provenance wall both run BEFORE any INSERT, so
    an invalid or eval-derived row is never written.
    """
    prepared = prepare_memory_rows(
        candidates, cohort=cohort, created_at=created_at, id_factory=id_factory
    )
    partition = partition_rows(prepared.valid, reserved)

    written = insert_memory_rows(execute, catalog, schema, partition.kept)

    tally.written += written
    tally.written_rows.extend(partition.kept)
    tally.dropped_provenance.extend(partition.dropped)
    tally.invalid.extend(prepared.invalid)

    return (
        f"submit_memory: wrote {written} row(s); "
        f"dropped {len(partition.dropped)} on the provenance wall; "
        f"rejected {len(prepared.invalid)} invalid candidate(s)."
    )


def create_submit_memory_tool(
    *,
    client: Any,
    warehouse_id: str,
    catalog: str,
    schema: str,
    cohort: str,
    reserved: ReservedPools,
    tally: WriteTally,
    now: Callable[[], str],
) -> Any:
    """The Claude Agent SDK ``@tool`` the distiller exposes to write memory.

    The agent calls it with a JSON array of candidate guideline rows; the tool
    validates them, applies the provenance wall, and INSERTs the survivors via the
    SQL Statement API (the SDK ``statement_execution`` seam). Escaped, never
    raw-interpolated. Lazy-imports ``claude_agent_sdk`` so the core package imports
    without the optional agent runtime.
    """
    from claude_agent_sdk import tool

    def execute(sql: str) -> None:
        _execute(client, warehouse_id, sql)

    @tool(
        "submit_memory",
        (
            f"Write distilled advisory-memory guideline rows to "
            f"{catalog}.{schema}.{MEMORY_TABLE}.\n\n"
            "Args:\n"
            "    memories_json: JSON array of guideline objects. Each object must have:\n"
            "        - category (str): short bucket, e.g. 'token_efficiency', 'tool_use'\n"
            "        - guideline_text (str): ONE actionable, self-contained guideline\n"
            "        - score (float): 0-1 confidence the feedback supports this guideline\n"
            "        - source_trace_ids (list[str]): trace id(s) this came from\n"
            "        - source_signal (str): 'rlm' or 'judge:<name>' (e.g. 'judge:correctness')\n\n"
            "Rows citing frozen eval-set traces are dropped automatically; cite only the "
            "trace ids provided in the prompt."
        ),
        {"memories_json": str},
    )
    async def submit_memory(args: dict) -> dict:
        raw = args.get("memories_json", "[]")
        try:
            candidates = json.loads(raw)
        except json.JSONDecodeError as exc:
            return {
                "content": [
                    {"type": "text", "text": f"Error: memories_json is not valid JSON: {exc}"}
                ],
                "is_error": True,
            }
        if isinstance(candidates, dict):
            candidates = [candidates]
        if not isinstance(candidates, list):
            return {
                "content": [{"type": "text", "text": "Error: memories_json must be a JSON array"}],
                "is_error": True,
            }
        try:
            summary = apply_and_write(
                candidates,
                execute=execute,
                catalog=catalog,
                schema=schema,
                cohort=cohort,
                created_at=now(),
                reserved=reserved,
                tally=tally,
            )
        except Exception as exc:  # surfaced to the model, never a silent partial write
            return {"content": [{"type": "text", "text": f"Error: {exc}"}], "is_error": True}
        return {"content": [{"type": "text", "text": summary}]}

    return submit_memory
