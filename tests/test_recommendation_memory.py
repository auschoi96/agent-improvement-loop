from __future__ import annotations

from typing import Any

from ail.memory.assessments import AssessmentRow, build_assessment_query
from ail.memory.provenance import ReservedPools
from ail.recommendations import schema, state
from ail.registry import Agent


def _agent() -> Agent:
    return Agent(
        agent_name="agent-a",
        experiment_id="exp-a",
        annotations_table="cat.trace.annotations",
    )


def _row(trace_id: str = "trace-1", comment: str = "feedback") -> AssessmentRow:
    return AssessmentRow(
        name="rlm_review",
        trace_id=trace_id,
        value="yes",
        comment=comment,
        created_at="2026-07-17 10:00:00",
        source_signal="rlm",
        metadata_json="{}",
    )


def test_schema_defines_every_governed_recommendation_table() -> None:
    statements = schema._ddl("cat", "sch")
    sql = "\n".join(statements)
    assert len(schema.RECOMMENDATION_TABLES) == 8
    for table in schema.RECOMMENDATION_TABLES:
        assert f"`cat`.`sch`.{table}" in sql
    assert "memory_json" not in sql
    assert "source_trace_ids ARRAY<STRING>" in sql


def test_evidence_identity_is_content_addressed_and_agent_isolated() -> None:
    first = state.evidence_id_for(_agent(), _row())
    assert first == state.evidence_id_for(_agent(), _row())
    assert first != state.evidence_id_for(_agent(), _row(comment="changed"))
    other = _agent().model_copy(update={"agent_name": "agent-b"})
    assert first != state.evidence_id_for(other, _row())


def test_reserved_traces_are_retained_for_audit_but_marked_ineligible() -> None:
    items = state.build_evidence_items(
        _agent(),
        [_row("abcdef1234567890")],
        reserved=ReservedPools(task_suite_ids=frozenset({"abcdef123456"})),
        ingested_at="now",
    )
    assert items[0].reserved_pool == "task_suite"
    assert items[0].subject_or_reviewer == "subject"


def test_planner_assessment_query_is_oldest_first_and_dynamic_judge_aware() -> None:
    sql = build_assessment_query(
        "cat.trace.annotations",
        since_created_at="before",
        max_results=500,
        judge_names=frozenset({"custom_helpfulness"}),
        ascending=True,
    )
    assert "custom_helpfulness" in sql
    assert "modularity" not in sql
    assert "ORDER BY created_at ASC" in sql
    assert "LIMIT 500" in sql


def test_eligible_trace_query_counts_distinct_traces_and_waits_for_judges(
    monkeypatch: Any,
) -> None:
    captured: list[str] = []

    def fake_read_rows(client: Any, warehouse_id: str, sql: str) -> list[dict[str, str]]:
        captured.append(sql)
        return [{"trace_id": "t1"}, {"trace_id": "t2"}]

    monkeypatch.setattr("ail.jobs.bootstrap_tables._read_rows", fake_read_rows)
    ids = state.read_eligible_trace_ids(
        object(),
        "wh",
        "cat",
        "sch",
        _agent(),
        judge_names=["correctness", "custom_helpfulness"],
        judge_grace_minutes=30,
        max_traces=25,
    )
    assert ids == ["t1", "t2"]
    sql = captured[0]
    assert "COUNT(DISTINCT" in sql
    assert "rlm_review" in sql
    assert "INTERVAL 30 MINUTES" in sql
    assert "LIMIT 25" in sql
    assert "cohort_id IS NULL" in sql


def test_cohort_id_is_stable_across_input_order() -> None:
    first = state.cohort_id_for(_agent(), ["t2", "t1"], ["e2", "e1"])
    second = state.cohort_id_for(_agent(), ["t1", "t2"], ["e1", "e2"])
    assert first == second
