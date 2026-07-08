"""The assessment reader (:mod:`ail.memory.assessments`): the query pulls RLM +
L2-judge FEEDBACK, honors the watermark, and projects rows correctly.
"""

from __future__ import annotations

from ail.memory.assessments import (
    JUDGE_ASSESSMENT_NAMES,
    AssessmentRow,
    build_assessment_query,
    max_created_at,
    read_assessments,
    source_signal_for,
)


def test_query_filters_rlm_and_judge_feedback() -> None:
    q = build_assessment_query("cat.mlflow_traces.cc_otel_annotations")
    assert "FROM cat.mlflow_traces.cc_otel_annotations" in q
    assert "deleted_at IS NULL" in q
    assert "annotation_type = 'FEEDBACK'" in q
    assert "name LIKE 'rlm%'" in q
    for judge in JUDGE_ASSESSMENT_NAMES:
        assert f"'{judge}'" in q
    # The failed-review marker is excluded.
    assert "rlm_review_failed" in q  # only in the NOT IN clause
    assert "NOT IN ('rlm_review_failed')" in q


def test_query_without_watermark_has_no_time_bound() -> None:
    q = build_assessment_query("t")
    assert "created_at >" not in q


def test_query_with_watermark_adds_time_bound() -> None:
    q = build_assessment_query("t", since_created_at="2026-07-03 07:57:07.085")
    assert "created_at > TIMESTAMP '2026-07-03 07:57:07.085'" in q


def test_query_escapes_watermark_literal() -> None:
    q = build_assessment_query("t", since_created_at="2026'; DROP")
    assert "2026''; DROP" in q  # single quote doubled


def test_source_signal_mapping() -> None:
    assert source_signal_for("rlm_token_efficiency") == "rlm"
    assert source_signal_for("rlm_review") == "rlm"
    assert source_signal_for("correctness") == "judge:correctness"
    assert source_signal_for("modularity") == "judge:modularity"


def test_read_assessments_projects_rows(fake_sql_client) -> None:
    columns = ["name", "target_id", "value_str", "comment", "created_at"]
    data = [
        ["token_efficiency", "trace1", "4.0", "read too much", "2026-07-03 07:57:07.085"],
        ["rlm_review", "trace2", "80", "", "2026-06-30 02:07:18.127"],
        ["correctness", "", "yes", "no trace id", "2026-07-01 00:00:00.000"],  # skipped: no target
    ]

    client = fake_sql_client({"SELECT name, target_id": (columns, data)})
    rows = read_assessments(client, "wh", annotations_table="t")

    assert [r.trace_id for r in rows] == ["trace1", "trace2"]
    assert rows[0].source_signal == "judge:token_efficiency"
    assert rows[0].comment == "read too much"
    assert rows[1].source_signal == "rlm"


def test_max_created_at() -> None:
    rows = [
        AssessmentRow("n", "t1", "1", "", "2026-07-03 07:57:07.085", "rlm"),
        AssessmentRow("n", "t2", "1", "", "2026-07-05 01:00:00.000", "rlm"),
    ]
    assert max_created_at(rows) == "2026-07-05 01:00:00.000"
    assert max_created_at([]) is None
