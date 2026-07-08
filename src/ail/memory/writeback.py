"""Write distilled memory to ``agent_memory`` — the custom ``@tool`` the agent
calls, the idempotent MERGE it runs, and the grounding + validation + provenance
gate that stands between model output and the governed table.

Mirrors the reference agent's ``submit_findings`` (a single custom tool the Claude
Agent SDK exposes so the model hands back structured rows written via the SQL
Statement API — here the SDK ``statement_execution`` seam, :func:`ail.publish._execute`),
but the memory store adds the guarantees this feature exists to provide:

* **Grounding (anti-fabrication)** — every ``source_trace_id`` a candidate cites
  must be in the set of trace ids actually READ this run. The model cannot cite a
  plausible-but-unread id; a row that does is dropped, fail-closed, with a reason.
* **Strict signal** — ``source_signal`` must be EXACTLY ``rlm`` or one of the four
  ``judge:<name>`` names (no ``judge:anything``, no ``rlm_review_failed``).
* **The provenance wall** — :func:`ail.memory.provenance.partition_rows` then drops
  any surviving row whose ``source_trace_ids`` touch the frozen eval pools.
* **Durable idempotency** — ``memory_id`` is a deterministic content hash and the
  write is a ``MERGE ... WHEN NOT MATCHED`` upsert, so reprocessing the same
  feedback window (e.g. after a watermark-write failure) inserts zero duplicates.

Failures (a SQL error, or a provenance-wall *assertion* that signals a drop-logic
regression) are recorded on the :class:`WriteTally` so the driver can refuse to
advance the watermark and surface them loudly — never swallowed. Values are escaped
via :func:`ail.publish._lit` (doubles quotes and backslashes), never interpolated raw.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ail.memory.assessments import JUDGE_ASSESSMENT_NAMES
from ail.memory.provenance import DroppedRow, ReservedPools, partition_rows
from ail.memory.schema import MEMORY_COLUMNS, MEMORY_TABLE, MemoryRow
from ail.publish import _execute, _lit

#: The EXACT allowed ``source_signal`` values — ``rlm`` or a specific ``judge:<name>``.
#: Prefix-permissive matching would let the model smuggle ``judge:anything`` or the
#: non-feedback ``rlm_review_failed`` marker through, so the set is closed.
VALID_SIGNALS: frozenset[str] = frozenset(
    {"rlm"} | {f"judge:{name}" for name in JUDGE_ASSESSMENT_NAMES}
)

#: Separator for the deterministic id payload — the unit separator control char,
#: which never appears in a cohort/category/guideline/trace-id.
_ID_SEP = "\x1f"


def memory_id_for(
    cohort: str, category: str, guideline_text: str, source_trace_ids: Sequence[str]
) -> str:
    """A DETERMINISTIC id for a memory row: sha256 of its content + sorted sources.

    Reprocessing the same distilled feedback yields the same id, so the ``MERGE``
    upsert (:func:`build_memory_merge`) inserts it at most once — durable idempotency
    even if a prior run wrote the row but failed to advance the watermark.
    """
    payload = _ID_SEP.join([cohort, category, guideline_text, *sorted(source_trace_ids)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Escaped, idempotent MERGE builder
# ---------------------------------------------------------------------------


def _string_array_literal(values: Sequence[str]) -> str:
    """``ARRAY('a', 'b')`` (escaped) for a non-empty ``ARRAY<STRING>`` column."""
    return "ARRAY(" + ", ".join(_lit(v) for v in values) + ")"


def _embedding_literal(values: Sequence[float] | None) -> str:
    """The ``ARRAY<FLOAT>`` embedding literal, ``CAST(NULL AS ARRAY<FLOAT>)`` when unset.

    The explicit cast keeps the ``VALUES`` table constructor's column typed as
    ``ARRAY<FLOAT>`` even when every row is NULL, so the MERGE's ``INSERT`` never
    fails trying to assign an untyped ``void`` NULL to the array column.
    """
    if values is None:
        return "CAST(NULL AS ARRAY<FLOAT>)"
    return "ARRAY(" + ", ".join(_lit(float(v)) for v in values) + ")"


def _row_values(row: MemoryRow) -> str:
    """One ``(...)`` VALUES tuple for ``row``, columns in :data:`MEMORY_COLUMNS` order."""
    rendered = [
        _lit(row.memory_id),
        _lit(row.cohort),
        _lit(row.category),
        _lit(row.guideline_text),
        _lit(row.score),
        _string_array_literal(row.source_trace_ids),
        _lit(row.source_signal),
        _lit(row.created_at),
        _embedding_literal(row.embedding),
    ]
    return "(" + ", ".join(rendered) + ")"


def build_memory_merge(catalog: str, schema: str, rows: Sequence[MemoryRow]) -> str:
    """An idempotent, escaped ``MERGE INTO agent_memory`` for ``rows`` (non-empty).

    Upserts on the deterministic ``memory_id``: ``WHEN NOT MATCHED THEN INSERT``
    only, so a row already present (a reprocessed window) is left untouched and no
    duplicate is written. Column list/order come from :data:`MEMORY_COLUMNS`; every
    value is escaped via :func:`ail.publish._lit` / the array literal helpers.
    """
    if not rows:
        raise ValueError("build_memory_merge requires at least one row")
    fqn = f"`{catalog}`.`{schema}`.{MEMORY_TABLE}"
    cols = ", ".join(MEMORY_COLUMNS)
    source_cols = ", ".join(f"s.{c}" for c in MEMORY_COLUMNS)
    values = ",\n    ".join(_row_values(r) for r in rows)
    # Column aliases are illegal directly on a MERGE ``USING`` source, so name the
    # columns on the inner derived table (VALUES ... AS v(cols)) and give the USING
    # source a bare alias.
    return (
        f"MERGE INTO {fqn} AS t\n"
        f"USING (SELECT * FROM (VALUES\n    {values}\n) AS v ({cols})) AS s\n"
        "ON t.memory_id = s.memory_id\n"
        f"WHEN NOT MATCHED THEN INSERT ({cols}) VALUES ({source_cols})"
    )


def merge_memory_rows(
    execute: Callable[[str], None],
    catalog: str,
    schema: str,
    rows: Sequence[MemoryRow],
) -> int:
    """Run :func:`build_memory_merge` through ``execute``; return the rows merged."""
    if not rows:
        return 0
    execute(build_memory_merge(catalog, schema, rows))
    return len(rows)


# ---------------------------------------------------------------------------
# Candidate validation (model output -> typed rows, or a drop reason)
# ---------------------------------------------------------------------------


def _validate_candidate(
    candidate: Any,
    *,
    cohort: str,
    created_at: str,
    read_trace_ids: frozenset[str],
) -> MemoryRow | str:
    """Coerce one raw candidate dict to a :class:`MemoryRow`, or return a reason string.

    Fail-closed anti-fabrication: ``source_trace_ids`` must be non-empty AND every id
    must be in ``read_trace_ids`` (the traces whose feedback was actually read this
    run). A row citing an unread id is rejected — a memory row is grounded ONLY in
    assessments this run actually saw.
    """
    if not isinstance(candidate, dict):
        return f"candidate is not an object: {type(candidate).__name__}"

    category = str(candidate.get("category", "")).strip()
    guideline = str(candidate.get("guideline_text", "")).strip()
    signal = str(candidate.get("source_signal", "")).strip()
    if not category:
        return "missing category"
    if not guideline:
        return "missing guideline_text"
    if signal not in VALID_SIGNALS:
        return f"invalid source_signal {signal!r} (expected one of {sorted(VALID_SIGNALS)})"

    raw_ids = candidate.get("source_trace_ids") or []
    if isinstance(raw_ids, str):
        raw_ids = [raw_ids]
    trace_ids = tuple(str(t).strip() for t in raw_ids if str(t).strip())
    if not trace_ids:
        return "missing source_trace_ids (a memory row must cite its provenance)"
    unread = [t for t in trace_ids if t not in read_trace_ids]
    if unread:
        return f"cites unread trace id(s) {unread} — must be grounded in feedback read this run"

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
        memory_id=memory_id_for(cohort, category, guideline, trace_ids),
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
    read_trace_ids: frozenset[str],
) -> PreparedRows:
    """Validate raw candidate dicts into typed rows (+ rejections), no provenance yet.

    ``read_trace_ids`` is the anti-fabrication grounding set (see
    :func:`_validate_candidate`). ``memory_id`` is derived deterministically from
    content, so the same distilled row always gets the same id.
    """
    valid: list[MemoryRow] = []
    invalid: list[tuple[Any, str]] = []
    for candidate in candidates:
        result = _validate_candidate(
            candidate, cohort=cohort, created_at=created_at, read_trace_ids=read_trace_ids
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
    """Accumulates outcomes across every ``submit_memory`` call in one run.

    ``errors`` records any FATAL failure (a SQL/MERGE error, or a provenance-wall
    assertion signalling a drop-logic regression). The driver refuses to advance the
    watermark and raises if it is non-empty — such failures are never swallowed.
    """

    written: int = 0
    dropped_provenance: list[DroppedRow] = field(default_factory=list)
    invalid: list[tuple[Any, str]] = field(default_factory=list)
    written_rows: list[MemoryRow] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def apply_and_write(
    candidates: Iterable[Any],
    *,
    execute: Callable[[str], None],
    catalog: str,
    schema: str,
    cohort: str,
    created_at: str,
    reserved: ReservedPools,
    read_trace_ids: frozenset[str],
    tally: WriteTally,
) -> str:
    """Validate → ground → wall → MERGE one batch of candidates; update ``tally``.

    The pure heart of the ``submit_memory`` tool, split out so the write path is
    fully testable without the Claude Agent SDK or a live model. Order is
    load-bearing: grounding + validation and the provenance wall both run BEFORE any
    write, so an ungrounded, invalid, or eval-derived row is never merged.

    Raises on a fatal failure — :class:`ail.pools.PoolOverlapError` if the wall's
    re-verification catches a drop-logic regression, or the underlying error if the
    MERGE fails. The caller (:func:`create_submit_memory_tool`) records it on the
    tally so the driver fails closed.
    """
    prepared = prepare_memory_rows(
        candidates, cohort=cohort, created_at=created_at, read_trace_ids=read_trace_ids
    )
    partition = partition_rows(prepared.valid, reserved)  # raises on drop-logic regression

    written = merge_memory_rows(execute, catalog, schema, partition.kept)  # raises on SQL failure

    tally.written += written
    tally.written_rows.extend(partition.kept)
    tally.dropped_provenance.extend(partition.dropped)
    tally.invalid.extend(prepared.invalid)

    return (
        f"submit_memory: merged {written} row(s); "
        f"dropped {len(partition.dropped)} on the provenance wall; "
        f"rejected {len(prepared.invalid)} invalid/ungrounded candidate(s)."
    )


def create_submit_memory_tool(
    *,
    client: Any,
    warehouse_id: str,
    catalog: str,
    schema: str,
    cohort: str,
    reserved: ReservedPools,
    read_trace_ids: frozenset[str],
    tally: WriteTally,
    now: Callable[[], str],
) -> Any:
    """The Claude Agent SDK ``@tool`` the distiller exposes to write memory.

    The agent calls it with a JSON array of candidate guideline rows; the tool
    grounds them (``read_trace_ids``), validates them, applies the provenance wall,
    and MERGEs the survivors via the SQL Statement API. A fatal failure (SQL error or
    provenance-wall regression) is recorded on ``tally.errors`` AND surfaced to the
    model — the driver then refuses to advance the watermark and raises, so nothing
    is silently swallowed. Lazy-imports ``claude_agent_sdk``.
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
            "        - source_signal (str): exactly 'rlm' or 'judge:<name>' where <name> is one of "
            "correctness, modularity, groundedness, token_efficiency\n\n"
            "Cite ONLY trace ids present in the prompt; rows citing unread or frozen "
            "eval-set traces are dropped automatically."
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
                read_trace_ids=read_trace_ids,
                tally=tally,
            )
        except Exception as exc:
            # FATAL (SQL failure or provenance-wall regression): record it so the
            # driver refuses to advance the watermark and raises — never swallowed.
            tally.errors.append(f"submit_memory failed: {exc}")
            return {"content": [{"type": "text", "text": f"Error: {exc}"}], "is_error": True}
        return {"content": [{"type": "text", "text": summary}]}

    return submit_memory
