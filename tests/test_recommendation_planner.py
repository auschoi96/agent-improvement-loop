from __future__ import annotations

import json
from typing import Any

import pytest

from ail.jobs import recommendation_planner as rp
from ail.loop.proposals import ActionKind, ChangeKind, ProposalStatus
from ail.memory.assessments import AssessmentRow
from ail.registry import Agent


def _row(
    name: str,
    trace_id: str,
    *,
    value: str = "1",
    comment: str = "feedback",
    created_at: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> AssessmentRow:
    return AssessmentRow(
        name=name,
        trace_id=trace_id,
        value=value,
        comment=comment,
        created_at=created_at or f"2026-07-17 10:00:{trace_id.lstrip('t') or '00'}",
        source_signal="rlm" if name.startswith("rlm_") else f"judge:{name}",
        metadata_json=json.dumps(metadata or {}),
    )


def _agent() -> Agent:
    return Agent(
        agent_name="claude_code",
        experiment_id="exp-1",
        annotations_table="cat.traces.cc_otel_annotations",
        goal_config={"objective_metric": "total_tokens"},
    )


def _config(**kwargs: Any) -> rp.PlannerConfig:
    return rp.PlannerConfig(warehouse_id="wh", catalog="cat", schema="sch", **kwargs)


def _cohort_rows(n: int = 10) -> list[AssessmentRow]:
    return [
        _row("rlm_review", f"t{i}", created_at=f"2026-07-17 10:00:{i:02d}")
        for i in range(n)
    ]


def _lookup(rows: list[AssessmentRow]) -> dict[tuple[str, str, str], str]:
    return {
        (row.trace_id, row.name, row.created_at): f"e-{index}" for index, row in enumerate(rows)
    }


def _observation(trace_ids: list[str], *, key: str = "batch_file_edits") -> dict[str, Any]:
    return {
        "operation": "create",
        "canonical_key": key,
        "category": "tooling",
        "title": "Fragmented file edits",
        "root_cause": "The agent applies same-file changes in many small calls.",
        "observation_summary": "Repeated fragmented edits recur across the cohort.",
        "source_trace_ids": trace_ids,
        "severity": 0.7,
        "confidence": 0.9,
        "trend_label": "new",
    }


def _action(trace_ids: list[str], *, key: str = "batch_file_edits") -> dict[str, Any]:
    return {
        "canonical_key": key,
        "category": "novel_open_category",
        "title": "Batch same-file changes",
        "rationale": "The pattern recurs across distinct traces.",
        "implementation_plan": "Add and validate a multi-hunk editing workflow.",
        "pattern_keys": [key],
        "source_trace_ids": trace_ids,
    }


def _patch_writes(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    captured: dict[str, list[Any]] = {
        "events": [],
        "patterns": [],
        "actions": [],
        "links": [],
        "proposals": [],
    }
    monkeypatch.setattr(rp, "read_pattern_event_trace_ids", lambda *args, **kwargs: set())
    monkeypatch.setattr(
        rp, "merge_pattern_event", lambda *args, **kwargs: captured["events"].append(args[-1])
    )
    monkeypatch.setattr(
        rp, "merge_pattern", lambda *args, **kwargs: captured["patterns"].append(args[-1])
    )
    monkeypatch.setattr(
        rp, "merge_action", lambda *args, **kwargs: captured["actions"].append(args[-1])
    )
    monkeypatch.setattr(
        rp,
        "merge_action_pattern",
        lambda *args, **kwargs: captured["links"].append(kwargs),
    )
    monkeypatch.setattr(
        rp,
        "insert_proposal_if_absent",
        lambda proposal, **kwargs: captured["proposals"].append(proposal) or True,
    )
    return captured


def test_subject_assessments_excludes_halo_reviewer_trace() -> None:
    rows = [
        _row(
            "rlm_review",
            "subject-1",
            metadata={"reviewer_trace_id": "trace:/cat.schema.exp/reviewer-1"},
        ),
        _row("modularity", "reviewer-1", comment="judge accidentally scored HALO"),
        _row("modularity", "subject-1", comment="real agent feedback"),
    ]
    kept = rp.subject_assessments(rows)
    assert [(row.name, row.trace_id) for row in kept] == [
        ("rlm_review", "subject-1"),
        ("modularity", "subject-1"),
    ]


def test_prompt_includes_evidence_pattern_bank_full_queue_and_open_categories() -> None:
    verdict = {
        "failure_modes": [{"title": "Repeated context"}],
        "recommendations": ["Add durable context summaries"],
        "raw_report": "must not duplicate the huge report",
    }
    rows = [
        _row("rlm_review", "t1", metadata={"verdict_json": json.dumps(verdict)}),
        _row("modularity", "t1", value="2", comment="Functions mixed concerns."),
    ]
    prompt = rp.build_recommendation_prompt(
        _agent(),
        rows,
        evidence_ids=_lookup(rows),
        patterns=[{"canonical_key": "cache_file_reads", "status": "active"}],
        existing_actions=[{"proposal_id": "p1", "status": "rejected"}],
    )
    assert "Add durable context summaries" in prompt
    assert "Functions mixed concerns" in prompt
    assert "raw_report" not in prompt
    assert "cache_file_reads" in prompt
    assert '"status": "rejected"' in prompt
    assert "Categories and change types are open-ended" in prompt
    assert "evidence_id=e-" in prompt


def test_recurring_pattern_creates_one_open_category_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_writes(monkeypatch)
    rows = _cohort_rows()
    tally = rp.PlannerTally(
        submitted=1,
        pattern_observations=[_observation(["t0", "t1", "t2", "t3"])],
        action_candidates=[_action(["t0", "t1", "t2", "t3"])],
    )
    message = rp.apply_cohort_analysis(
        agent=_agent(),
        rows=rows,
        evidence_lookup=_lookup(rows),
        cohort_id="cohort-1",
        existing_patterns=[],
        existing_actions=[],
        client=object(),
        config=_config(),
        tally=tally,
        created_at="now",
    )
    assert "queued 1" in message
    assert len(captured["events"]) == 1
    assert len(captured["proposals"]) == 1
    proposal = captured["proposals"][0]
    assert proposal.action_kind is ActionKind.AGENT_TASK
    assert proposal.change.kind is ChangeKind.AGENT_TASK_PLAN
    assert proposal.status is ProposalStatus.PENDING
    assert proposal.trigger.asset_type == "novel_open_category"
    assert proposal.trigger.n_traces == 4


def test_one_off_signal_stays_in_pattern_memory_without_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_writes(monkeypatch)
    rows = _cohort_rows()
    tally = rp.PlannerTally(
        submitted=1,
        pattern_observations=[_observation(["t0"])],
        action_candidates=[_action(["t0"])],
    )
    rp.apply_cohort_analysis(
        agent=_agent(),
        rows=rows,
        evidence_lookup=_lookup(rows),
        cohort_id="cohort-1",
        existing_patterns=[],
        existing_actions=[],
        client=object(),
        config=_config(),
        tally=tally,
        created_at="now",
    )
    assert captured["patterns"]
    assert captured["patterns"][0]["status"] == "emerging"
    assert captured["proposals"] == []


def test_pending_action_is_strengthened_not_paraphrased(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_writes(monkeypatch)
    rows = _cohort_rows()
    existing = {
        "proposal_id": "p1",
        "action_id": "a1",
        "canonical_action_key": "batch_file_edits",
        "status": "pending",
        "title": "Batch edits",
    }
    tally = rp.PlannerTally(
        submitted=1,
        pattern_observations=[_observation(["t0", "t1", "t2"])],
        action_candidates=[_action(["t0", "t1", "t2"])],
    )
    rp.apply_cohort_analysis(
        agent=_agent(),
        rows=rows,
        evidence_lookup=_lookup(rows),
        cohort_id="cohort-2",
        existing_patterns=[],
        existing_actions=[existing],
        client=object(),
        config=_config(),
        tally=tally,
        created_at="now",
    )
    assert captured["proposals"] == []
    assert captured["links"][0]["action_id"] == "a1"
    assert captured["links"][0]["relation"] == "covered_by"


def test_invalid_pattern_reference_writes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_writes(monkeypatch)
    rows = _cohort_rows()
    action = _action(["t0", "t1", "t2"])
    action["pattern_keys"] = ["not_observed"]
    tally = rp.PlannerTally(
        submitted=1,
        pattern_observations=[_observation(["t0", "t1", "t2"])],
        action_candidates=[action],
    )
    with pytest.raises(ValueError, match="patterns observed in this cohort"):
        rp.apply_cohort_analysis(
            agent=_agent(),
            rows=rows,
            evidence_lookup=_lookup(rows),
            cohort_id="cohort-1",
            existing_patterns=[],
            existing_actions=[],
            client=object(),
            config=_config(),
            tally=tally,
            created_at="now",
        )
    assert all(not values for values in captured.values())


def test_halo_evidence_span_resolves_to_parent_trace() -> None:
    trace_id = "8fc2bb5033a904cde824e5e9e542524b"
    evidence_span_id = "211a47921b17c504"
    rows = [
        _row(
            "rlm_review",
            trace_id,
            metadata={
                "verdict_json": json.dumps(
                    {"guideline_assessments": [{"evidence_span_ids": [evidence_span_id]}]}
                )
            },
        )
    ]
    assert rp._resolve_trace_ids({"source_trace_ids": [evidence_span_id]}, rows) == [trace_id]


def _patch_run_dependencies(
    monkeypatch: pytest.MonkeyPatch, *, eligible: list[str], evidence: list[AssessmentRow]
) -> dict[str, list[Any]]:
    calls: dict[str, list[Any]] = {"assigned": [], "finished": [], "watermark": []}
    monkeypatch.setattr(rp, "_ensure_state_tables", lambda *args: None)
    monkeypatch.setattr(rp, "_read_existing_actions", lambda *args: [])
    monkeypatch.setattr(rp, "_sync_queue_memory", lambda *args, **kwargs: [])
    monkeypatch.setattr(rp, "read_ingestion_watermark", lambda *args: "before")
    monkeypatch.setattr(rp, "read_assessments", lambda *args, **kwargs: [])
    monkeypatch.setattr(rp, "read_eligible_trace_ids", lambda *args, **kwargs: eligible)
    monkeypatch.setattr(
        rp,
        "read_evidence_for_traces",
        lambda *args, **kwargs: [(f"e{i}", row) for i, row in enumerate(evidence)],
    )
    monkeypatch.setattr(rp, "next_cohort_sequence", lambda *args: 1)
    monkeypatch.setattr(rp, "begin_cohort", lambda *args, **kwargs: None)
    monkeypatch.setattr(rp, "read_patterns", lambda *args: [])
    monkeypatch.setattr(
        rp,
        "assign_evidence_to_cohort",
        lambda *args, **kwargs: calls["assigned"].append((args, kwargs)),
    )
    monkeypatch.setattr(
        rp,
        "finish_cohort",
        lambda *args, **kwargs: calls["finished"].append(kwargs),
    )
    monkeypatch.setattr(rp, "apply_cohort_analysis", lambda **kwargs: "stored 0; queued 0")
    return calls


def test_nine_traces_buffer_without_calling_model(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = _cohort_rows(9)
    _patch_run_dependencies(
        monkeypatch, eligible=[row.trace_id for row in rows], evidence=rows
    )
    called = False

    def recommend(items: list[AssessmentRow], tally: rp.PlannerTally) -> None:
        nonlocal called
        called = True

    report = rp.run_for_agent(
        _agent(), _config(), deps=rp.PlannerDeps(client=object(), recommend=recommend)
    )
    assert called is False
    assert report.n_subject_traces == 9
    assert "9/10" in report.note


def test_tenth_trace_creates_exactly_one_committed_cohort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _cohort_rows()
    calls = _patch_run_dependencies(
        monkeypatch, eligible=[row.trace_id for row in rows], evidence=rows
    )

    def recommend(items: list[AssessmentRow], tally: rp.PlannerTally) -> None:
        assert len({row.trace_id for row in items}) == 10
        tally.submitted = 1
        tally.pattern_observations = []
        tally.action_candidates = []

    report = rp.run_for_agent(
        _agent(),
        _config(),
        deps=rp.PlannerDeps(client=object(), recommend=recommend, now=lambda: "now"),
    )
    assert report.cohort_id
    assert len(calls["assigned"]) == 1
    assert calls["finished"][-1]["status"] == "committed"


def test_model_must_submit_exactly_once_and_failed_cohort_stays_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _cohort_rows()
    calls = _patch_run_dependencies(
        monkeypatch, eligible=[row.trace_id for row in rows], evidence=rows
    )
    with pytest.raises(RuntimeError, match="exactly once"):
        rp.run_for_agent(
            _agent(),
            _config(),
            deps=rp.PlannerDeps(client=object(), recommend=lambda rows, tally: None),
        )
    assert calls["assigned"] == []
    assert calls["finished"][-1]["status"] == "failed"


def test_unchanged_queue_history_requires_no_delta_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    action = {
        "proposal_id": "p1",
        "status": "pending",
        "title": "Batch edits",
        "plan": "Canonical key: batch_file_edits\nPlan",
        "category": "tooling",
        "created_at": "before",
        "canonical_action_key": "batch_file_edits",
        "proof_proved_improvement": None,
    }
    action_id = rp.action_id_for(_agent(), "batch_file_edits")
    monkeypatch.setattr(
        rp,
        "read_action_index",
        lambda *args: {action_id: {"proposal_id": "p1", "status": "pending"}},
    )
    writes: list[list[dict[str, Any]]] = []
    monkeypatch.setattr(
        rp, "merge_actions", lambda *args: writes.append(list(args[-1]))
    )
    rp._sync_queue_memory(object(), _config(), _agent(), [action], now="now")
    assert writes == [[]]


def test_parse_defaults_enforce_robust_cohort_settings() -> None:
    args = rp._parse_args(["--warehouse-id=wh", "--catalog=cat", "--schema=sch"])
    assert args.min_traces == 10
    assert args.max_traces == 25
    assert args.max_recommendations == 3
    assert args.pattern_min_current_traces == 3
