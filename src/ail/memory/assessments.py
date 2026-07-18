"""Read the eval-feedback the distiller turns into memory: RLM/HALO ``rlm_*`` and
the L2 MLflow-judge assessments, straight off the trace store's annotation table.

Read mechanism (verified against the live workspace)
----------------------------------------------------
The monitoring service and the RLM reviewer attach feedback to a subject trace as
MLflow assessments (``mlflow.log_feedback``). Those assessments are queryable two
ways: ``mlflow.search_traces(...)`` (assessments on ``trace.info.assessments``) or
SQL against the UC-backed annotation table (``<traces_schema>.<prefix>_otel_annotations``).

We use the **SQL path**. On the live workspace a single ``SELECT`` against
``austin_choi_omni_agent_catalog.mlflow_traces.cc_otel_annotations`` returns BOTH
signal families with everything the distiller needs and nothing it doesn't:

* ``name`` — the assessment (``rlm_*`` / ``correctness`` / ``modularity`` /
  ``groundedness`` / ``token_efficiency``);
* ``target_id`` — the subject **trace id** (the provenance the wall checks);
* ``value`` — the score (``"5.0"`` / ``"yes"`` / …);
* ``comment`` — the reviewer/judge **rationale** (the substance to distill); and
* ``created_at`` — a monotone timestamp that is a natural idempotency watermark.

``mlflow.search_traces`` would pull whole traces and give no clean per-assessment
``created_at`` to watermark on, so the annotation table is both cheaper and the
better fit. The trace store is implicitly experiment-scoped by its ``<prefix>_``
convention, so ``annotations_table`` names exactly the experiment's harness store.

Fail-closed
-----------
This layer only *reads*. It returns ``[]`` when the window holds no assessments
(the caller then writes nothing); it never invents a row. A genuine read failure
(bad table / no grant) raises out of the shared :func:`ail.publish._execute`-style
statement runner, so the Job fails and writes nothing rather than distilling air.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Prefix of every RLM/HALO assessment name (``rlm_review``, ``rlm_token_efficiency``…).
RLM_PREFIX = "rlm_"

#: The four L2 MLflow-judge assessment names the monitoring service writes.
JUDGE_ASSESSMENT_NAMES: frozenset[str] = frozenset(
    {"correctness", "modularity", "groundedness", "token_efficiency"}
)

#: RLM names that are NOT distillable feedback — the fail-closed "review attempted
#: but did not complete" marker carries no verdict, so it is excluded from the read
#: (mirrors :data:`ail.l3.continuous.REVIEW_FAILED_FEEDBACK_NAME`).
_EXCLUDED_NAMES: frozenset[str] = frozenset({"rlm_review_failed"})


@dataclass(frozen=True, slots=True)
class AssessmentRow:
    """One eval-feedback assessment read off the trace store.

    ``trace_id`` is the subject trace (annotation ``target_id``); ``source_signal``
    is the coarse family the distiller records on any memory derived from this row
    (``"rlm"`` or ``"judge:<name>"``).
    """

    name: str
    trace_id: str
    value: str
    comment: str
    created_at: str
    source_signal: str
    #: Raw assessment metadata. For ``rlm_review`` this carries ``verdict_json``
    #: (including HALO recommendations/failure modes); judge rows carry scorer
    #: provenance. Kept as JSON text so existing memory consumers remain simple.
    metadata_json: str = ""


def source_signal_for(name: str) -> str:
    """Coarse ``source_signal`` for an assessment name: ``rlm`` or ``judge:<name>``."""
    if name.startswith(RLM_PREFIX):
        return "rlm"
    return f"judge:{name}"


def _escape(value: str) -> str:
    """Escape a value for a single-quoted SQL string literal."""
    return value.replace("\\", "\\\\").replace("'", "''")


def build_assessment_query(
    annotations_table: str,
    *,
    since_created_at: str | None = None,
    max_results: int = 500,
    judge_names: frozenset[str] | None = None,
    ascending: bool = False,
) -> str:
    """The ``SELECT`` that returns distillable RLM + judge FEEDBACK, newest first.

    Scoped to live (``deleted_at IS NULL``) ``FEEDBACK`` annotations whose name is
    an RLM name (``rlm%``) or one of :data:`JUDGE_ASSESSMENT_NAMES`, excluding
    :data:`_EXCLUDED_NAMES`. When ``since_created_at`` is given, only rows strictly
    after it are returned — the watermark predicate that makes a re-run a no-op.
    ``annotations_table`` is a trusted, operator-configured identifier (a bundle
    var), never model output.
    """
    judges = JUDGE_ASSESSMENT_NAMES if judge_names is None else judge_names
    judge_list = ", ".join(f"'{_escape(n)}'" for n in sorted(judges)) or "''"
    excluded_list = ", ".join(f"'{_escape(n)}'" for n in sorted(_EXCLUDED_NAMES))
    where = [
        "deleted_at IS NULL",
        "annotation_type = 'FEEDBACK'",
        "target_type = 'TRACE'",
        f"(name LIKE 'rlm%' OR name IN ({judge_list}))",
        f"name NOT IN ({excluded_list})",
    ]
    if since_created_at:
        where.append(f"created_at > TIMESTAMP '{_escape(since_created_at)}'")
    where_sql = "\n  AND ".join(where)
    return (
        "SELECT name, target_id, CAST(value AS STRING) AS value_str, "
        "COALESCE(comment, '') AS comment, CAST(metadata AS STRING) AS metadata_json, "
        "CAST(created_at AS STRING) AS created_at\n"
        f"FROM {annotations_table}\n"
        f"WHERE {where_sql}\n"
        f"ORDER BY created_at {'ASC' if ascending else 'DESC'}\n"
        f"LIMIT {int(max_results)}"
    )


def read_assessments(
    client: Any,
    warehouse_id: str,
    *,
    annotations_table: str,
    since_created_at: str | None = None,
    max_results: int = 500,
    judge_names: frozenset[str] | None = None,
    ascending: bool = False,
) -> list[AssessmentRow]:
    """Read distillable RLM + judge assessments since ``since_created_at``.

    Runs :func:`build_assessment_query` on ``warehouse_id`` via the shared
    :func:`ail.jobs.bootstrap_tables._read_rows` seam and projects each row onto an
    :class:`AssessmentRow`. Returns ``[]`` when the window is empty (fail-closed:
    the caller writes nothing). Rows whose ``target_id`` (trace id) is missing are
    skipped — a memory row must carry resolvable provenance for the wall.
    """
    from ail.jobs.bootstrap_tables import _read_rows

    query = build_assessment_query(
        annotations_table,
        since_created_at=since_created_at,
        max_results=max_results,
        judge_names=judge_names,
        ascending=ascending,
    )
    rows = _read_rows(client, warehouse_id, query)
    out: list[AssessmentRow] = []
    for row in rows:
        trace_id = row.get("target_id")
        name = row.get("name")
        if not trace_id or not name:
            continue
        out.append(
            AssessmentRow(
                name=str(name),
                trace_id=str(trace_id),
                value=str(row.get("value_str") or ""),
                comment=str(row.get("comment") or ""),
                created_at=str(row.get("created_at") or ""),
                source_signal=source_signal_for(str(name)),
                metadata_json=str(row.get("metadata_json") or ""),
            )
        )
    return out


def max_created_at(rows: list[AssessmentRow]) -> str | None:
    """The newest ``created_at`` across ``rows`` (the next watermark), or ``None``.

    ``CAST(created_at AS STRING)`` yields a fixed ISO-ish format, so a lexicographic
    max is a chronological max.
    """
    stamps = [r.created_at for r in rows if r.created_at]
    return max(stamps) if stamps else None
