"""Persistence and deterministic identity for recommendation decision memory."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from ail.memory.assessments import AssessmentRow
from ail.memory.provenance import ReservedPools
from ail.publish import _execute, _lit
from ail.recommendations.schema import (
    ACTION_PATTERN_TABLE,
    ACTION_TABLE,
    COHORT_TABLE,
    EVIDENCE_TABLE,
    INGESTION_WATERMARK_TABLE,
    OUTCOME_TABLE,
    PATTERN_EVENT_TABLE,
    PATTERN_TABLE,
)
from ail.registry import Agent

_MIN_RESERVED_PREFIX = 12
T = TypeVar("T")


def stable_id(namespace: str, *parts: object, length: int = 24) -> str:
    """A deterministic lower-case identifier suitable for Delta MERGE keys."""
    payload = "\0".join([namespace, *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode()).hexdigest()[:length]


def evidence_id_for(agent: Agent, row: AssessmentRow) -> str:
    """Content-address one assessment; an edited assessment becomes new evidence."""
    return stable_id(
        "recommendation-evidence",
        agent.agent_name,
        agent.experiment_id,
        row.trace_id,
        row.name,
        row.value,
        row.comment,
        row.metadata_json,
        row.created_at,
        length=32,
    )


def pattern_id_for(agent: Agent, canonical_key: str) -> str:
    return stable_id(
        "recommendation-pattern", agent.agent_name, agent.experiment_id, canonical_key
    )


def action_id_for(agent: Agent, canonical_key: str) -> str:
    return stable_id("recommendation-action", agent.agent_name, agent.experiment_id, canonical_key)


def cohort_id_for(agent: Agent, trace_ids: Sequence[str], evidence_ids: Sequence[str]) -> str:
    return stable_id(
        "recommendation-cohort",
        agent.agent_name,
        agent.experiment_id,
        *sorted(trace_ids),
        *sorted(evidence_ids),
        length=32,
    )


def _array(values: Iterable[str]) -> str:
    encoded = json.dumps(list(dict.fromkeys(str(value) for value in values)))
    return f"from_json({_lit(encoded)}, 'ARRAY<STRING>')"


def _table(catalog: str, schema: str, name: str) -> str:
    return f"`{catalog}`.`{schema}`.{name}"


def _chunks(items: Sequence[T], size: int) -> Iterable[Sequence[T]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def _reserved_pool(trace_id: str, reserved: ReservedPools) -> str | None:
    for name, ids in (
        ("task_suite", reserved.task_suite_ids),
        ("human_anchor", reserved.human_anchor_ids),
    ):
        for reserved_id in ids:
            if trace_id == reserved_id:
                return name
            n = min(len(trace_id), len(reserved_id))
            if n >= _MIN_RESERVED_PREFIX and trace_id[:n] == reserved_id[:n]:
                return name
    return None


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    evidence_id: str
    row: AssessmentRow
    subject_or_reviewer: str
    reserved_pool: str | None
    ingested_at: str


def build_evidence_items(
    agent: Agent,
    rows: Sequence[AssessmentRow],
    *,
    reserved: ReservedPools,
    ingested_at: str,
) -> list[EvidenceItem]:
    """Classify a read batch while retaining reviewer/reserved evidence for audit."""
    reviewer_ids: set[str] = set()
    for row in rows:
        if row.name != "rlm_review" or not row.metadata_json:
            continue
        try:
            metadata = json.loads(row.metadata_json)
        except (TypeError, ValueError):
            continue
        reviewer_uri = metadata.get("reviewer_trace_id") if isinstance(metadata, dict) else None
        reviewer_id = str(reviewer_uri or "").rsplit("/", 1)[-1]
        if reviewer_id:
            reviewer_ids.add(reviewer_id)
    return [
        EvidenceItem(
            evidence_id=evidence_id_for(agent, row),
            row=row,
            subject_or_reviewer="reviewer" if row.trace_id in reviewer_ids else "subject",
            reserved_pool=_reserved_pool(row.trace_id, reserved),
            ingested_at=ingested_at,
        )
        for row in rows
    ]


def merge_evidence(
    client: Any,
    warehouse_id: str,
    catalog: str,
    schema: str,
    agent: Agent,
    items: Sequence[EvidenceItem],
) -> None:
    """Idempotently append content-addressed assessment evidence in bounded batches."""
    table = _table(catalog, schema, EVIDENCE_TABLE)
    for chunk in _chunks(items, 40):
        projected: list[str] = []
        aliases = [
            "agent_name",
            "experiment_id",
            "evidence_id",
            "trace_id",
            "cohort_id",
            "assessment_name",
            "source_signal",
            "value",
            "comment",
            "metadata_json",
            "assessment_created_at",
            "subject_or_reviewer",
            "reserved_pool",
            "ingested_at",
        ]
        for item in chunk:
            row = item.row
            raw = [
                _lit(agent.agent_name),
                _lit(agent.experiment_id),
                _lit(item.evidence_id),
                _lit(row.trace_id),
                "CAST(NULL AS STRING)",
                _lit(row.name),
                _lit(row.source_signal),
                _lit(row.value),
                _lit(row.comment),
                _lit(row.metadata_json),
                _lit(row.created_at),
                _lit(item.subject_or_reviewer),
                _lit(item.reserved_pool),
                _lit(item.ingested_at),
            ]
            projected.append(
                "SELECT "
                + ", ".join(
                    f"{value} AS {name}" for value, name in zip(raw, aliases, strict=True)
                )
            )
        source = "\nUNION ALL\n".join(projected)
        _execute(
            client,
            warehouse_id,
            f"""MERGE INTO {table} AS t
USING ({source}) AS s
ON t.agent_name = s.agent_name AND t.experiment_id = s.experiment_id
   AND t.evidence_id = s.evidence_id
WHEN MATCHED AND s.subject_or_reviewer = 'reviewer'
  THEN UPDATE SET t.subject_or_reviewer = 'reviewer'
WHEN NOT MATCHED THEN INSERT *""",
        )


def read_ingestion_watermark(
    client: Any, warehouse_id: str, catalog: str, schema: str, agent: Agent
) -> str | None:
    from ail.jobs.bootstrap_tables import _read_rows

    rows = _read_rows(
        client,
        warehouse_id,
        f"SELECT last_created_at FROM {_table(catalog, schema, INGESTION_WATERMARK_TABLE)} "
        f"WHERE agent_name = {_lit(agent.agent_name)} "
        f"AND experiment_id = {_lit(agent.experiment_id)} LIMIT 1",
    )
    return str(rows[0]["last_created_at"]) if rows and rows[0].get("last_created_at") else None


def write_ingestion_watermark(
    client: Any,
    warehouse_id: str,
    catalog: str,
    schema: str,
    agent: Agent,
    *,
    last_created_at: str,
    updated_at: str,
    n_assessments: int,
) -> None:
    table = _table(catalog, schema, INGESTION_WATERMARK_TABLE)
    _execute(
        client,
        warehouse_id,
        f"""MERGE INTO {table} AS t
USING (SELECT {_lit(agent.agent_name)} AS agent_name,
              {_lit(agent.experiment_id)} AS experiment_id,
              {_lit(last_created_at)} AS last_created_at,
              {_lit(updated_at)} AS updated_at,
              {int(n_assessments)} AS n_assessments_ingested) AS s
ON t.agent_name = s.agent_name AND t.experiment_id = s.experiment_id
WHEN MATCHED THEN UPDATE SET last_created_at = s.last_created_at,
    updated_at = s.updated_at,
    n_assessments_ingested = COALESCE(t.n_assessments_ingested, 0)
        + s.n_assessments_ingested
WHEN NOT MATCHED THEN INSERT *""",
    )


def read_eligible_trace_ids(
    client: Any,
    warehouse_id: str,
    catalog: str,
    schema: str,
    agent: Agent,
    *,
    judge_names: Sequence[str],
    judge_grace_minutes: int,
    max_traces: int,
) -> list[str]:
    """Oldest unassigned subject traces with RLM complete and judges ready/timed out."""
    from ail.jobs.bootstrap_tables import _read_rows

    table = _table(catalog, schema, EVIDENCE_TABLE)
    judges = ", ".join(_lit(name) for name in sorted(set(judge_names))) or "''"
    required = len(set(judge_names))
    rows = _read_rows(
        client,
        warehouse_id,
        f"""SELECT trace_id,
       MAX(CASE WHEN assessment_name = 'rlm_review' THEN assessment_created_at END) AS ready_at
FROM {table} e
WHERE agent_name = {_lit(agent.agent_name)}
  AND experiment_id = {_lit(agent.experiment_id)}
  AND cohort_id IS NULL
  AND subject_or_reviewer = 'subject'
  AND reserved_pool IS NULL
  AND NOT EXISTS (
      SELECT 1 FROM {table} prior
      WHERE prior.agent_name = e.agent_name
        AND prior.experiment_id = e.experiment_id
        AND prior.trace_id = e.trace_id
        AND prior.cohort_id IS NOT NULL
  )
GROUP BY trace_id
HAVING SUM(CASE WHEN assessment_name = 'rlm_review' THEN 1 ELSE 0 END) > 0
   AND ({required} = 0
        OR COUNT(DISTINCT CASE WHEN assessment_name IN ({judges})
                              THEN assessment_name END) >= {required}
        OR MAX(CASE WHEN assessment_name = 'rlm_review'
                    THEN TRY_TO_TIMESTAMP(assessment_created_at) END)
           <= current_timestamp() - INTERVAL {int(judge_grace_minutes)} MINUTES)
ORDER BY ready_at ASC, trace_id ASC
LIMIT {int(max_traces)}""",
    )
    return [str(row["trace_id"]) for row in rows if row.get("trace_id")]


def read_evidence_for_traces(
    client: Any,
    warehouse_id: str,
    catalog: str,
    schema: str,
    agent: Agent,
    trace_ids: Sequence[str],
) -> list[tuple[str, AssessmentRow]]:
    from ail.jobs.bootstrap_tables import _read_rows

    if not trace_ids:
        return []
    ids = ", ".join(_lit(trace_id) for trace_id in trace_ids)
    rows = _read_rows(
        client,
        warehouse_id,
        f"""SELECT evidence_id, assessment_name, trace_id, value, comment,
       metadata_json, assessment_created_at, source_signal
FROM {_table(catalog, schema, EVIDENCE_TABLE)}
WHERE agent_name = {_lit(agent.agent_name)}
  AND experiment_id = {_lit(agent.experiment_id)}
  AND trace_id IN ({ids})
  AND subject_or_reviewer = 'subject'
  AND reserved_pool IS NULL
ORDER BY trace_id, assessment_created_at, evidence_id""",
    )
    return [
        (
            str(row["evidence_id"]),
            AssessmentRow(
                name=str(row.get("assessment_name") or ""),
                trace_id=str(row.get("trace_id") or ""),
                value=str(row.get("value") or ""),
                comment=str(row.get("comment") or ""),
                created_at=str(row.get("assessment_created_at") or ""),
                source_signal=str(row.get("source_signal") or ""),
                metadata_json=str(row.get("metadata_json") or ""),
            ),
        )
        for row in rows
        if row.get("evidence_id") and row.get("trace_id") and row.get("assessment_name")
    ]


def next_cohort_sequence(
    client: Any, warehouse_id: str, catalog: str, schema: str, agent: Agent
) -> int:
    from ail.jobs.bootstrap_tables import _read_rows

    rows = _read_rows(
        client,
        warehouse_id,
        f"SELECT COALESCE(MAX(cohort_sequence), 0) + 1 AS next_sequence "
        f"FROM {_table(catalog, schema, COHORT_TABLE)} "
        f"WHERE agent_name = {_lit(agent.agent_name)} "
        f"AND experiment_id = {_lit(agent.experiment_id)}",
    )
    return int(rows[0].get("next_sequence") or 1) if rows else 1


def begin_cohort(
    client: Any,
    warehouse_id: str,
    catalog: str,
    schema: str,
    agent: Agent,
    *,
    cohort_id: str,
    sequence: int,
    min_traces: int,
    trace_ids: Sequence[str],
    evidence_ids: Sequence[str],
    evidence_cutoff_at: str,
    queue_snapshot_at: str,
    planner_model: str,
    planner_prompt_version: str,
    planner_run_id: str,
    created_at: str,
) -> None:
    table = _table(catalog, schema, COHORT_TABLE)
    _execute(
        client,
        warehouse_id,
        f"""MERGE INTO {table} AS t
USING (SELECT {_lit(agent.agent_name)} AS agent_name,
              {_lit(agent.experiment_id)} AS experiment_id,
              {_lit(cohort_id)} AS cohort_id,
              {int(sequence)} AS cohort_sequence,
              'planning' AS status,
              {int(min_traces)} AS min_traces,
              {len(set(trace_ids))} AS trace_count,
              {len(set(evidence_ids))} AS assessment_count,
              {_array(trace_ids)} AS trace_ids,
              {_array(evidence_ids)} AS evidence_ids,
              {_lit(evidence_cutoff_at)} AS evidence_cutoff_at,
              {_lit(queue_snapshot_at)} AS queue_snapshot_at,
              {_lit(planner_model)} AS planner_model,
              {_lit(planner_prompt_version)} AS planner_prompt_version,
              {_lit(planner_run_id)} AS planner_run_id,
              {_lit(created_at)} AS created_at,
              {_lit(created_at)} AS started_at,
              CAST(NULL AS STRING) AS completed_at,
              CAST(NULL AS STRING) AS error) AS s
ON t.agent_name = s.agent_name AND t.experiment_id = s.experiment_id
   AND t.cohort_id = s.cohort_id
WHEN MATCHED THEN UPDATE SET status = 'planning', planner_run_id = s.planner_run_id,
    started_at = s.started_at, error = NULL
WHEN NOT MATCHED THEN INSERT *""",
    )


def assign_evidence_to_cohort(
    client: Any,
    warehouse_id: str,
    catalog: str,
    schema: str,
    agent: Agent,
    cohort_id: str,
    evidence_ids: Sequence[str],
) -> None:
    if not evidence_ids:
        return
    ids = ", ".join(_lit(evidence_id) for evidence_id in evidence_ids)
    _execute(
        client,
        warehouse_id,
        f"UPDATE {_table(catalog, schema, EVIDENCE_TABLE)} SET cohort_id = {_lit(cohort_id)} "
        f"WHERE agent_name = {_lit(agent.agent_name)} "
        f"AND experiment_id = {_lit(agent.experiment_id)} "
        f"AND evidence_id IN ({ids}) AND cohort_id IS NULL",
    )


def finish_cohort(
    client: Any,
    warehouse_id: str,
    catalog: str,
    schema: str,
    agent: Agent,
    cohort_id: str,
    *,
    status: str,
    completed_at: str,
    error: str | None = None,
) -> None:
    _execute(
        client,
        warehouse_id,
        f"UPDATE {_table(catalog, schema, COHORT_TABLE)} "
        f"SET status = {_lit(status)}, completed_at = {_lit(completed_at)}, error = {_lit(error)} "
        f"WHERE agent_name = {_lit(agent.agent_name)} "
        f"AND experiment_id = {_lit(agent.experiment_id)} "
        f"AND cohort_id = {_lit(cohort_id)}",
    )


def read_patterns(
    client: Any, warehouse_id: str, catalog: str, schema: str, agent: Agent
) -> list[dict[str, Any]]:
    from ail.jobs.bootstrap_tables import _read_rows

    return _read_rows(
        client,
        warehouse_id,
        f"SELECT * FROM {_table(catalog, schema, PATTERN_TABLE)} "
        f"WHERE agent_name = {_lit(agent.agent_name)} "
        f"AND experiment_id = {_lit(agent.experiment_id)} "
        "ORDER BY updated_at DESC LIMIT 300",
    )


def read_action_index(
    client: Any, warehouse_id: str, catalog: str, schema: str, agent: Agent
) -> dict[str, dict[str, str]]:
    """Small queue-sync index; unchanged history requires no write on a scheduled run."""
    from ail.jobs.bootstrap_tables import _read_rows

    rows = _read_rows(
        client,
        warehouse_id,
        f"SELECT action_id, proposal_id, status FROM {_table(catalog, schema, ACTION_TABLE)} "
        f"WHERE agent_name = {_lit(agent.agent_name)} "
        f"AND experiment_id = {_lit(agent.experiment_id)}",
    )
    return {
        str(row["action_id"]): {
            "proposal_id": str(row.get("proposal_id") or ""),
            "status": str(row.get("status") or ""),
        }
        for row in rows
        if row.get("action_id")
    }


def read_pattern_event_trace_ids(
    client: Any,
    warehouse_id: str,
    catalog: str,
    schema: str,
    agent: Agent,
    pattern_id: str,
) -> set[str]:
    from ail.jobs.bootstrap_tables import _read_rows

    rows = _read_rows(
        client,
        warehouse_id,
        f"SELECT source_trace_ids FROM {_table(catalog, schema, PATTERN_EVENT_TABLE)} "
        f"WHERE agent_name = {_lit(agent.agent_name)} "
        f"AND experiment_id = {_lit(agent.experiment_id)} "
        f"AND pattern_id = {_lit(pattern_id)}",
    )
    out: set[str] = set()
    for row in rows:
        value = row.get("source_trace_ids") or []
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                value = []
        out.update(str(item) for item in value if item)
    return out


def merge_pattern_event(
    client: Any,
    warehouse_id: str,
    catalog: str,
    schema: str,
    agent: Agent,
    row: dict[str, Any],
) -> None:
    table = _table(catalog, schema, PATTERN_EVENT_TABLE)
    _execute(
        client,
        warehouse_id,
        f"""MERGE INTO {table} AS t
USING (SELECT {_lit(agent.agent_name)} AS agent_name,
              {_lit(agent.experiment_id)} AS experiment_id,
              {_lit(row['event_id'])} AS event_id,
              {_lit(row['pattern_id'])} AS pattern_id,
              {_lit(row['cohort_id'])} AS cohort_id,
              {_lit(row['event_type'])} AS event_type,
              {_array(row['evidence_ids'])} AS evidence_ids,
              {_array(row['source_trace_ids'])} AS source_trace_ids,
              {_lit(row['observation_summary'])} AS observation_summary,
              {float(row['severity'])} AS severity,
              {float(row['confidence'])} AS confidence,
              {_lit(row['created_at'])} AS created_at) AS s
ON t.agent_name = s.agent_name AND t.experiment_id = s.experiment_id
   AND t.event_id = s.event_id
WHEN NOT MATCHED THEN INSERT *""",
    )


def merge_pattern(
    client: Any,
    warehouse_id: str,
    catalog: str,
    schema: str,
    agent: Agent,
    row: dict[str, Any],
) -> None:
    table = _table(catalog, schema, PATTERN_TABLE)
    _execute(
        client,
        warehouse_id,
        f"""MERGE INTO {table} AS t
USING (SELECT {_lit(agent.agent_name)} AS agent_name,
              {_lit(agent.experiment_id)} AS experiment_id,
              {_lit(row['pattern_id'])} AS pattern_id,
              {_lit(row['canonical_key'])} AS canonical_key,
              {_lit(row['category'])} AS category,
              {_lit(row['title'])} AS title,
              {_lit(row['root_cause'])} AS root_cause,
              {_lit(row['status'])} AS status,
              {_lit(row['first_seen_cohort_id'])} AS first_seen_cohort_id,
              {_lit(row['last_seen_cohort_id'])} AS last_seen_cohort_id,
              {int(row['cohort_count'])} AS cohort_count,
              {int(row['distinct_trace_count'])} AS distinct_trace_count,
              {int(row['recent_trace_count'])} AS recent_trace_count,
              {float(row['recent_prevalence'])} AS recent_prevalence,
              {float(row['severity'])} AS severity,
              {float(row['confidence'])} AS confidence,
              {float(row['trend_score'])} AS trend_score,
              {_lit(row['trend_label'])} AS trend_label,
              {_lit(row.get('current_action_id'))} AS current_action_id,
              CAST(NULL AS ARRAY<FLOAT>) AS summary_embedding,
              {_lit(row['created_at'])} AS created_at,
              {_lit(row['updated_at'])} AS updated_at) AS s
ON t.agent_name = s.agent_name AND t.experiment_id = s.experiment_id
   AND t.pattern_id = s.pattern_id
WHEN MATCHED THEN UPDATE SET canonical_key = s.canonical_key, category = s.category,
    title = s.title, root_cause = s.root_cause, status = s.status,
    last_seen_cohort_id = s.last_seen_cohort_id, cohort_count = s.cohort_count,
    distinct_trace_count = s.distinct_trace_count,
    recent_trace_count = s.recent_trace_count,
    recent_prevalence = s.recent_prevalence, severity = s.severity,
    confidence = s.confidence, trend_score = s.trend_score,
    trend_label = s.trend_label, current_action_id = s.current_action_id,
    updated_at = s.updated_at
WHEN NOT MATCHED THEN INSERT *""",
    )


def merge_action(
    client: Any,
    warehouse_id: str,
    catalog: str,
    schema: str,
    agent: Agent,
    row: dict[str, Any],
) -> None:
    merge_actions(client, warehouse_id, catalog, schema, agent, [row])


def merge_actions(
    client: Any,
    warehouse_id: str,
    catalog: str,
    schema: str,
    agent: Agent,
    rows: Sequence[dict[str, Any]],
) -> None:
    """Batch-upsert queue/action lineage so a large backlog costs one statement."""
    if not rows:
        return
    table = _table(catalog, schema, ACTION_TABLE)
    for chunk in _chunks(rows, 100):
        selects = []
        for row in chunk:
            selects.append(
                f"""SELECT {_lit(agent.agent_name)} AS agent_name,
              {_lit(agent.experiment_id)} AS experiment_id,
              {_lit(row['action_id'])} AS action_id,
              {_lit(row['canonical_action_key'])} AS canonical_action_key,
              {_lit(row['category'])} AS category,
              {_lit(row['title'])} AS title,
              {_lit(row['plan'])} AS plan,
              {_lit(row['status'])} AS status,
              {_lit(row.get('proposal_id'))} AS proposal_id,
              {_lit(row.get('first_proposed_cohort_id'))} AS first_proposed_cohort_id,
              {_lit(row.get('last_supported_cohort_id'))} AS last_supported_cohort_id,
              {_lit(row.get('supersedes_action_id'))} AS supersedes_action_id,
              {_lit(row.get('merged_into_action_id'))} AS merged_into_action_id,
              {_lit(row.get('human_decided_at'))} AS human_decided_at,
              {_lit(row.get('applied_at'))} AS applied_at,
              {_lit(row['created_at'])} AS created_at,
              {_lit(row['updated_at'])} AS updated_at"""
            )
        source = "\nUNION ALL\n".join(selects)
        _execute(
            client,
            warehouse_id,
            f"""MERGE INTO {table} AS t
USING ({source}) AS s
ON t.agent_name = s.agent_name AND t.experiment_id = s.experiment_id
   AND t.action_id = s.action_id
WHEN MATCHED THEN UPDATE SET category = s.category, title = s.title, plan = s.plan,
    status = s.status, proposal_id = COALESCE(s.proposal_id, t.proposal_id),
    last_supported_cohort_id = COALESCE(s.last_supported_cohort_id,
                                       t.last_supported_cohort_id),
    supersedes_action_id = COALESCE(s.supersedes_action_id, t.supersedes_action_id),
    merged_into_action_id = COALESCE(s.merged_into_action_id, t.merged_into_action_id),
    human_decided_at = COALESCE(s.human_decided_at, t.human_decided_at),
    applied_at = COALESCE(s.applied_at, t.applied_at), updated_at = s.updated_at
WHEN NOT MATCHED THEN INSERT *""",
        )


def merge_action_pattern(
    client: Any,
    warehouse_id: str,
    catalog: str,
    schema: str,
    agent: Agent,
    *,
    action_id: str,
    pattern_id: str,
    relation: str,
    cohort_id: str,
    now: str,
) -> None:
    link_id = stable_id("recommendation-action-pattern", action_id, pattern_id, relation)
    table = _table(catalog, schema, ACTION_PATTERN_TABLE)
    _execute(
        client,
        warehouse_id,
        f"""MERGE INTO {table} AS t
USING (SELECT {_lit(agent.agent_name)} AS agent_name,
              {_lit(agent.experiment_id)} AS experiment_id,
              {_lit(link_id)} AS link_id, {_lit(action_id)} AS action_id,
              {_lit(pattern_id)} AS pattern_id, {_lit(relation)} AS relation,
              {_lit(cohort_id)} AS first_linked_cohort_id,
              {_lit(cohort_id)} AS last_linked_cohort_id,
              {_lit(now)} AS created_at, {_lit(now)} AS updated_at) AS s
ON t.agent_name = s.agent_name AND t.experiment_id = s.experiment_id
   AND t.link_id = s.link_id
WHEN MATCHED THEN UPDATE SET last_linked_cohort_id = s.last_linked_cohort_id,
    updated_at = s.updated_at
WHEN NOT MATCHED THEN INSERT *""",
    )


def merge_outcome(
    client: Any,
    warehouse_id: str,
    catalog: str,
    schema: str,
    agent: Agent,
    row: dict[str, Any],
) -> None:
    table = _table(catalog, schema, OUTCOME_TABLE)
    baseline = (
        "CAST(NULL AS DOUBLE)"
        if row.get("baseline_value") is None
        else repr(float(row["baseline_value"]))
    )
    candidate = (
        "CAST(NULL AS DOUBLE)"
        if row.get("candidate_value") is None
        else repr(float(row["candidate_value"]))
    )
    delta = (
        "CAST(NULL AS DOUBLE)" if row.get("delta") is None else repr(float(row["delta"]))
    )
    _execute(
        client,
        warehouse_id,
        f"""MERGE INTO {table} AS t
USING (SELECT {_lit(agent.agent_name)} AS agent_name,
              {_lit(agent.experiment_id)} AS experiment_id,
              {_lit(row['outcome_id'])} AS outcome_id,
              {_lit(row['action_id'])} AS action_id,
              {_lit(row.get('proposal_id'))} AS proposal_id,
              {_lit(row['observed_at'])} AS observed_at,
              {_lit(row['source'])} AS source,
              {_lit(row.get('metric_name'))} AS metric_name,
              {baseline} AS baseline_value,
              {candidate} AS candidate_value,
              {delta} AS delta,
              {_lit(row['result'])} AS result,
              {int(row.get('n_traces') or 0)} AS n_traces,
              {_lit(row.get('window_start'))} AS window_start,
              {_lit(row.get('window_end'))} AS window_end,
              {_lit(row.get('details_json') or '{}')} AS details_json) AS s
ON t.agent_name = s.agent_name AND t.experiment_id = s.experiment_id
   AND t.outcome_id = s.outcome_id
WHEN NOT MATCHED THEN INSERT *""",
    )
