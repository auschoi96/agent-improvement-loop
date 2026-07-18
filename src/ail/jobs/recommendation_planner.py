"""Hosted, cohort-level recommendation planner with governed decision memory.

The scheduled Databricks Job first persists assessment evidence, then invokes the
planner only when a configurable number of distinct RLM-reviewed subject traces is
ready.  Claude submits grounded pattern observations and bounded action candidates;
deterministic framework code owns identity, support thresholds, queue deduplication,
lineage, Delta writes, and human approval semantics.
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ail.jobs.multi_agent import (
    load_registered_agents,
    missing_registry_target,
    run_for_each_registered_agent,
)
from ail.loop.proposals import (
    ActionKind,
    ChangeKind,
    GateStatus,
    ProposalStatus,
    ProposedAction,
    ProposedChange,
    TriggerKind,
    TriggerSignal,
    default_risk_class,
)
from ail.loop.publish_proposals import PROPOSALS_TABLE, insert_proposal_if_absent
from ail.memory.assessments import (
    JUDGE_ASSESSMENT_NAMES,
    AssessmentRow,
    max_created_at,
    read_assessments,
)
from ail.memory.distiller import _claude_env
from ail.memory.provenance import ReservedPools, resolve_reserved_pools
from ail.publish import _execute, _lit
from ail.recommendations.schema import _ddl as recommendation_ddl
from ail.recommendations.state import (
    action_id_for,
    assign_evidence_to_cohort,
    begin_cohort,
    build_evidence_items,
    cohort_id_for,
    finish_cohort,
    merge_action,
    merge_action_pattern,
    merge_actions,
    merge_evidence,
    merge_outcome,
    merge_pattern,
    merge_pattern_event,
    next_cohort_sequence,
    pattern_id_for,
    read_action_index,
    read_eligible_trace_ids,
    read_evidence_for_traces,
    read_ingestion_watermark,
    read_pattern_event_trace_ids,
    read_patterns,
    stable_id,
    write_ingestion_watermark,
)
from ail.registry import Agent

DEFAULT_MODEL = "databricks-claude-opus-4-8"
PROMPT_VERSION = "recommendation-cohort/v2"
_TAG = "[ail.jobs.recommendation_planner]"
_CANONICAL_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{2,79}$")
_TREND_SCORES = {"falling": -1.0, "stable": 0.0, "new": 0.25, "rising": 1.0}
_ACTIVE_QUEUE_STATES = frozenset({"pending", "approved", "applied"})


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class PlannerConfig:
    warehouse_id: str
    catalog: str
    schema: str
    model: str = DEFAULT_MODEL
    max_turns: int = 20
    max_assessments: int = 500
    min_traces: int = 10
    max_traces: int = 25
    judge_grace_minutes: int = 30
    max_recommendations: int = 3
    pattern_min_current_traces: int = 3
    pattern_min_total_traces: int = 5
    pattern_min_cohorts: int = 2
    task_suite_version: str = "v1"
    groundtruth_root: str | None = None


@dataclass(slots=True)
class PlannerTally:
    submitted: int = 0
    written: int = 0
    proposals: list[ProposedAction] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    pattern_observations: list[dict[str, Any]] | None = None
    action_candidates: list[dict[str, Any]] | None = None


@dataclass(slots=True)
class PlannerReport:
    agent_name: str
    n_assessments: int
    n_subject_assessments: int
    n_subject_traces: int
    n_proposals: int
    watermark_before: str | None
    watermark_after: str | None
    cohort_id: str | None = None
    note: str = ""


@dataclass(slots=True)
class PlannerDeps:
    client: Any
    recommend: Callable[[list[AssessmentRow], PlannerTally], None] | None = None
    now: Callable[[], str] = field(default=_now)
    reserved: ReservedPools = field(default_factory=ReservedPools)


def _metadata(row: AssessmentRow) -> dict[str, Any]:
    if not row.metadata_json:
        return {}
    try:
        value = json.loads(row.metadata_json)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _trace_id_from_uri(value: Any) -> str | None:
    text = str(value or "").strip()
    return text.rsplit("/", 1)[-1] if text else None


def subject_assessments(rows: Iterable[AssessmentRow]) -> list[AssessmentRow]:
    """Remove assessments of HALO's own reviewer traces from subject evidence."""
    materialized = list(rows)
    reviewer_ids = {
        reviewer_id
        for row in materialized
        if row.name == "rlm_review"
        and (reviewer_id := _trace_id_from_uri(_metadata(row).get("reviewer_trace_id")))
    }
    return [row for row in materialized if row.trace_id not in reviewer_ids]


def _trim_verdict(metadata: dict[str, Any]) -> dict[str, Any] | None:
    raw = metadata.get("verdict_json")
    if not raw:
        return None
    try:
        verdict = json.loads(str(raw))
    except (TypeError, ValueError):
        return None
    if not isinstance(verdict, dict):
        return None
    return {
        key: verdict.get(key)
        for key in (
            "summary",
            "token_efficiency",
            "token_waste_score",
            "guideline_assessments",
            "recommended_assets",
            "failure_modes",
            "recommendations",
        )
        if verdict.get(key) not in (None, "", [], {})
    }


def _evidence_span_ids(metadata: dict[str, Any]) -> set[str]:
    raw = metadata.get("verdict_json")
    if not raw:
        return set()
    try:
        verdict = json.loads(str(raw))
    except (TypeError, ValueError):
        return set()
    found: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "evidence_span_ids" and isinstance(child, list):
                    found.update(str(item).strip() for item in child if str(item).strip())
                else:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(verdict)
    return found


def _judge_names_for_agent(agent: Agent) -> frozenset[str]:
    """Discover configured judge names while retaining the built-in scorer set."""
    names = set(JUDGE_ASSESSMENT_NAMES)
    config = agent.judge_config or {}
    for key in ("judges", "judge_names", "scorers"):
        value = config.get(key)
        if isinstance(value, dict):
            names.update(str(name) for name in value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and item.get("name"):
                    names.add(str(item["name"]))
                elif isinstance(item, str):
                    names.add(item)
    guardrails = (agent.goal_config or {}).get("guardrail_judge")
    if isinstance(guardrails, str) and guardrails.strip():
        names.add(guardrails.split(":", 1)[0].strip())
    elif isinstance(guardrails, list):
        for guardrail in guardrails:
            if isinstance(guardrail, str):
                names.add(guardrail.split(":", 1)[0].strip())
            elif isinstance(guardrail, dict) and guardrail.get("name"):
                names.add(str(guardrail["name"]))
    return frozenset(name for name in names if name and not name.startswith("rlm_"))


def build_recommendation_prompt(
    agent: Agent,
    rows: list[AssessmentRow],
    *,
    evidence_ids: dict[tuple[str, str, str], str] | None = None,
    patterns: list[dict[str, Any]] | None = None,
    existing_actions: list[dict[str, Any]] | None = None,
    max_recommendations: int = 3,
) -> str:
    """Render one frozen cohort plus queryable pattern, queue, and outcome memory."""
    evidence_lookup = evidence_ids or {}
    lines = [
        f"AGENT: {agent.agent_name}",
        f"EXPERIMENT: {agent.experiment_id}",
        "CONFIRMED GOAL CONFIG: " + json.dumps(agent.goal_config or {}, sort_keys=True),
        "",
        "FROZEN MULTI-TRACE COHORT",
    ]
    by_trace: dict[str, list[AssessmentRow]] = {}
    for row in rows:
        by_trace.setdefault(row.trace_id, []).append(row)
    for trace_id, items in by_trace.items():
        lines.append(f"\n### trace {trace_id}")
        for row in items:
            evidence_id = evidence_lookup.get((row.trace_id, row.name, row.created_at), "")
            comment = " ".join(row.comment.split())
            lines.append(
                f"- evidence_id={evidence_id or 'unknown'} [{row.source_signal}] "
                f"{row.name} = {row.value or 'n/a'}" + (f" — {comment}" if comment else "")
            )
            if row.name == "rlm_review" and (verdict := _trim_verdict(_metadata(row))):
                lines.append("  HALO_VERDICT: " + json.dumps(verdict, ensure_ascii=False))

    lines.extend(
        [
            "",
            "QUERYABLE PATTERN BANK",
            json.dumps(patterns or [], ensure_ascii=False, default=str),
            "",
            "APPROVAL/ACTION MEMORY (all statuses; human decisions are authoritative)",
            json.dumps(existing_actions or [], ensure_ascii=False, default=str),
            "",
            "Review the ENTIRE cohort before deciding. Consolidate recurring root causes across "
            "traces and prior cohorts. Do not emit one pattern or action per trace. Categories "
            "and change types are open-ended; examples are not an allowlist. Low-support and "
            "lower-priority signals must remain pattern observations instead of queue cards.",
            "",
            "Call submit_cohort_analysis exactly once with two JSON arrays. patterns_json contains "
            "objects with operation (create|reinforce|contradict|merge|split), canonical_key, "
            "category, title, root_cause, observation_summary, source_trace_ids, severity (0-1), "
            "confidence (0-1), and trend_label (new|rising|stable|falling). Reuse an existing "
            "pattern_id/canonical_key whenever the root cause matches memory. recommendations_json "
            "contains only the top distinct actions with canonical_key, category, title, "
            "rationale, implementation_plan, pattern_keys, and source_trace_ids. A pattern_key "
            "may be a pattern_id or canonical_key. Existing pending/approved/applied coverage "
            "means no new card. The framework—not you—enforces cross-trace support and "
            "deduplication. "
            f"Submit at most {max_recommendations} action candidates. Empty recommendations is a "
            "successful and common result. Do not reject or alter approval statuses.",
        ]
    )
    return "\n".join(lines)


def _json_object_list(raw: Any, *, name: str) -> list[dict[str, Any]]:
    try:
        value = json.loads(str(raw or "[]"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is not valid JSON: {exc}") from exc
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"{name} must be an array of objects")
    return value


def create_submit_cohort_analysis_tool(tally: PlannerTally) -> Any:
    from claude_agent_sdk import tool

    @tool(
        "submit_cohort_analysis",
        "Submit grounded pattern observations and the bounded top action candidates. No SQL and "
        "no replacement memory document; recommendation categories are open-ended.",
        {"patterns_json": str, "recommendations_json": str},
    )
    async def submit_cohort_analysis(args: dict[str, Any]) -> dict[str, Any]:
        try:
            if tally.submitted:
                raise ValueError("submit_cohort_analysis must be called exactly once")
            patterns = _json_object_list(args.get("patterns_json"), name="patterns_json")
            actions = _json_object_list(
                args.get("recommendations_json"), name="recommendations_json"
            )
            tally.pattern_observations = patterns
            tally.action_candidates = actions
            tally.submitted = 1
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"accepted {len(patterns)} pattern observation(s) and "
                        f"{len(actions)} candidate action(s) for deterministic validation",
                    }
                ]
            }
        except Exception as exc:  # noqa: BLE001 - surface a model tool error to the driver
            tally.errors.append(f"{type(exc).__name__}: {exc}")
            return {"content": [{"type": "text", "text": f"Error: {exc}"}], "is_error": True}

    return submit_cohort_analysis


_SYSTEM_PROMPT = (
    "You are the cohort-level recommendation planner in a human-gated agent improvement loop. "
    "Maintain durable cross-trace patterns, use RLM/HALO and judge rationales together, and "
    "compare against all queued and decided actions. Never plan per trace. Never reject an "
    "approval. Prefer zero or a few high-impact actions over redundant cards."
)


def _default_recommend(
    agent: Agent,
    rows: list[AssessmentRow],
    *,
    config: PlannerConfig,
    tally: PlannerTally,
    evidence_ids: dict[tuple[str, str, str], str],
    patterns: list[dict[str, Any]],
    existing_actions: list[dict[str, Any]],
) -> None:
    import nest_asyncio
    from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server, query

    nest_asyncio.apply()
    submit = create_submit_cohort_analysis_tool(tally)
    server = create_sdk_mcp_server(name="recommendation-tools", tools=[submit])
    project = Path(tempfile.mkdtemp(prefix="ail_recommendations_"))
    options = ClaudeAgentOptions(
        cwd=str(project),
        allowed_tools=["mcp__recommendation-tools__submit_cohort_analysis"],
        permission_mode="bypassPermissions",
        mcp_servers={"recommendation-tools": server},
        system_prompt=_SYSTEM_PROMPT,
        setting_sources=[],
        env=_claude_env(config.model),
        max_turns=config.max_turns,
    )

    async def run_agent() -> None:
        async for _message in query(
            prompt=build_recommendation_prompt(
                agent,
                rows,
                evidence_ids=evidence_ids,
                patterns=patterns,
                existing_actions=existing_actions,
                max_recommendations=config.max_recommendations,
            ),
            options=options,
        ):
            pass

    try:
        asyncio.run(run_agent())
    finally:
        shutil.rmtree(project, ignore_errors=True)


def _ensure_state_tables(client: Any, config: PlannerConfig) -> None:
    for statement in recommendation_ddl(config.catalog, config.schema):
        _execute(client, config.warehouse_id, statement)


def _canonical_key_from_plan(plan: Any, proposal_id: str) -> str:
    match = re.search(r"(?im)^Canonical key:\s*([a-z0-9][a-z0-9_-]{2,79})\s*$", str(plan or ""))
    return match.group(1).lower() if match else f"legacy_{proposal_id.lower()}"


def _read_existing_actions(
    client: Any, config: PlannerConfig, agent: Agent
) -> list[dict[str, Any]]:
    """Read the full human queue/history, not only pending actions."""
    from ail.jobs.bootstrap_tables import _read_rows

    table = f"`{config.catalog}`.`{config.schema}`.{PROPOSALS_TABLE}"
    rows = _read_rows(
        client,
        config.warehouse_id,
        f"""SELECT proposal_id, status, change_summary, change_plan,
       trigger_asset_type, trigger_trace_refs, created_at,
       proof_proved_improvement, proof_realized_savings_pct,
       proof_objective_metric, proof_n_promote
FROM {table}
WHERE agent_name = {_lit(agent.agent_name)}
  AND experiment_id = {_lit(agent.experiment_id)}
  AND action_kind = 'agent_task'
ORDER BY created_at DESC LIMIT 1000""",
    )
    out: list[dict[str, Any]] = []
    for row in rows:
        proposal_id = str(row.get("proposal_id") or "")
        if not proposal_id:
            continue
        out.append(
            {
                "proposal_id": proposal_id,
                "status": str(row.get("status") or ""),
                "title": str(row.get("change_summary") or ""),
                "plan": str(row.get("change_plan") or ""),
                "category": str(row.get("trigger_asset_type") or ""),
                "trace_refs": row.get("trigger_trace_refs") or [],
                "created_at": str(row.get("created_at") or ""),
                "canonical_action_key": _canonical_key_from_plan(
                    row.get("change_plan"), proposal_id
                ),
                "proof_proved_improvement": row.get("proof_proved_improvement"),
                "proof_realized_savings_pct": row.get("proof_realized_savings_pct"),
                "proof_objective_metric": row.get("proof_objective_metric"),
                "proof_n_promote": row.get("proof_n_promote"),
            }
        )
    return out


def _sync_queue_memory(
    client: Any,
    config: PlannerConfig,
    agent: Agent,
    actions: list[dict[str, Any]],
    *,
    now: str,
) -> list[dict[str, Any]]:
    """Mirror authoritative human status/proof into planner action/outcome memory."""
    known = read_action_index(
        client, config.warehouse_id, config.catalog, config.schema, agent
    )
    action_rows: list[dict[str, Any]] = []
    for action in actions:
        key = action["canonical_action_key"]
        action_id = action_id_for(agent, key)
        status = action["status"] or "queued"
        action_row = {
                "action_id": action_id,
                "canonical_action_key": key,
                "category": action["category"] or "uncategorized",
                "title": action["title"] or key,
                "plan": action["plan"] or action["title"] or key,
                "status": status,
                "proposal_id": action["proposal_id"],
                "first_proposed_cohort_id": None,
                "last_supported_cohort_id": None,
                "human_decided_at": now if status in {"approved", "rejected", "applied"} else None,
                "applied_at": now if status == "applied" else None,
                "created_at": action["created_at"] or now,
                "updated_at": now,
        }
        indexed = known.get(action_id)
        if indexed is None or indexed != {
            "proposal_id": action["proposal_id"],
            "status": status,
        }:
            action_rows.append(action_row)
        action["action_id"] = action_id
        if status in {"approved", "rejected", "applied"}:
            merge_outcome(
                client,
                config.warehouse_id,
                config.catalog,
                config.schema,
                agent,
                {
                    "outcome_id": stable_id(
                        "recommendation-human-outcome", action["proposal_id"], status
                    ),
                    "action_id": action_id,
                    "proposal_id": action["proposal_id"],
                    "observed_at": now,
                    "source": "human_decision",
                    "metric_name": None,
                    "result": "inconclusive",
                    "n_traces": 0,
                    "details_json": json.dumps({"status": status}),
                },
            )
        proved = action.get("proof_proved_improvement")
        if proved is not None:
            merge_outcome(
                client,
                config.warehouse_id,
                config.catalog,
                config.schema,
                agent,
                {
                    "outcome_id": stable_id(
                        "recommendation-tier2-outcome", action["proposal_id"], proved
                    ),
                    "action_id": action_id,
                    "proposal_id": action["proposal_id"],
                    "observed_at": now,
                    "source": "tier2_verification",
                    "metric_name": action.get("proof_objective_metric"),
                    "delta": action.get("proof_realized_savings_pct"),
                    "result": "improved" if bool(proved) else "regressed",
                    "n_traces": int(action.get("proof_n_promote") or 0),
                    "details_json": json.dumps({"proved_improvement": bool(proved)}),
                },
            )
    merge_actions(
        client,
        config.warehouse_id,
        config.catalog,
        config.schema,
        agent,
        action_rows,
    )
    return actions


def _float01(value: Any, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number from 0 to 1") from exc
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field_name} must be from 0 to 1")
    return result


def _required_text(candidate: dict[str, Any], names: Sequence[str]) -> dict[str, str]:
    values = {name: str(candidate.get(name, "")).strip() for name in names}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise ValueError(f"candidate is missing {', '.join(missing)}")
    return values


def _canonical_key(candidate: dict[str, Any], field_name: str = "canonical_key") -> str:
    value = str(candidate.get(field_name, "")).strip().lower()
    if not _CANONICAL_KEY_RE.fullmatch(value):
        raise ValueError(
            f"{field_name} must be 3-80 lowercase letters, digits, '_' or '-'"
        )
    return value


def _resolve_trace_ids(candidate: dict[str, Any], rows: Sequence[AssessmentRow]) -> list[str]:
    raw = candidate.get("source_trace_ids") or []
    if isinstance(raw, str):
        raw = [raw]
    requested = [trace_id for value in raw if (trace_id := _trace_id_from_uri(value))]
    readable = {row.trace_id for row in rows}
    evidence_parents: dict[str, set[str]] = {}
    for row in rows:
        for evidence_id in _evidence_span_ids(_metadata(row)):
            evidence_parents.setdefault(evidence_id, set()).add(row.trace_id)
    resolved: list[str] = []
    unread: list[str] = []
    for trace_id in requested:
        if trace_id in readable:
            resolved.append(trace_id)
            continue
        parents = evidence_parents.get(trace_id, set())
        if len(parents) == 1:
            resolved.append(next(iter(parents)))
            continue
        ranked = sorted(
            (
                (difflib.SequenceMatcher(a=trace_id, b=known).ratio(), known)
                for known in readable
                if known[:8] == trace_id[:8]
            ),
            reverse=True,
        )
        runner_up = ranked[1][0] if len(ranked) > 1 else 0.0
        if ranked and ranked[0][0] >= 0.85 and ranked[0][0] - runner_up >= 0.05:
            resolved.append(ranked[0][1])
        else:
            unread.append(trace_id)
    if not resolved:
        raise ValueError("candidate has no grounded source_trace_ids")
    if unread:
        raise ValueError(f"candidate cites unread trace id(s): {sorted(unread)}")
    return list(dict.fromkeys(resolved))


def _candidate_to_proposal(
    candidate: dict[str, Any],
    *,
    agent: Agent,
    rows: list[AssessmentRow],
    trace_ids: Sequence[str],
    pattern_ids: Sequence[str],
    cohort_id: str,
    created_at: str,
    planner_model: str,
) -> ProposedAction:
    values = _required_text(
        candidate, ("category", "title", "rationale", "implementation_plan")
    )
    canonical_key = _canonical_key(candidate)
    signals = sorted({row.source_signal for row in rows if row.trace_id in trace_ids})
    plan = (
        f"Canonical key: {canonical_key}\n"
        f"Recommendation: {values['title']}\n"
        f"Category: {values['category']}\n"
        f"Rationale: {values['rationale']}\n\n"
        f"Implementation plan:\n{values['implementation_plan']}\n\n"
        f"Patterns: {', '.join(pattern_ids)}\n"
        f"Evidence cohort: {cohort_id}\n"
        f"Evidence traces: {', '.join(trace_ids)}\n"
        f"Evidence signals: {', '.join(signals)}"
    )
    objective = str((agent.goal_config or {}).get("objective_metric") or "agent_quality")
    return ProposedAction(
        proposal_id=hashlib.sha256(
            f"{agent.agent_name}\0{ActionKind.AGENT_TASK.value}\0{canonical_key}".encode()
        ).hexdigest()[:16],
        agent_name=agent.agent_name,
        experiment_id=agent.experiment_id,
        action_kind=ActionKind.AGENT_TASK,
        risk_class=default_risk_class(ActionKind.AGENT_TASK),
        status=ProposalStatus.PENDING,
        objective_metric=objective,
        goal_cohort=agent.cohort().name,
        trigger=TriggerSignal(
            kind=TriggerKind.AGENT_PLANNER,
            summary=f"[{values['category']}] {values['rationale']}",
            metric=values["category"],
            n_traces=len(set(trace_ids)),
            trace_refs=list(trace_ids),
            asset_type=values["category"],
        ),
        change=ProposedChange(
            kind=ChangeKind.AGENT_TASK_PLAN,
            summary=values["title"],
            plan=plan,
        ),
        proof=None,
        gate_status=GateStatus(
            readiness_tier="human_review",
            gated=True,
            reasons=["Cross-trace RLM/judge evidence is present; the human is the decision gate."],
        ),
        created_at=created_at,
        notes=[
            f"canonical_key={canonical_key}",
            f"cohort_id={cohort_id}",
            f"pattern_ids={','.join(pattern_ids)}",
            f"source_signals={','.join(signals)}",
            f"planner_model={planner_model}",
        ],
    )


def apply_cohort_analysis(
    *,
    agent: Agent,
    rows: list[AssessmentRow],
    evidence_lookup: dict[tuple[str, str, str], str],
    cohort_id: str,
    existing_patterns: list[dict[str, Any]],
    existing_actions: list[dict[str, Any]],
    client: Any,
    config: PlannerConfig,
    tally: PlannerTally,
    created_at: str,
) -> str:
    """Validate a complete model submission, evolve patterns, and append distinct cards."""
    observations = tally.pattern_observations or []
    action_candidates = tally.action_candidates or []
    if len(action_candidates) > config.max_recommendations:
        raise ValueError(
            f"action batch has {len(action_candidates)} candidates; maximum is "
            f"{config.max_recommendations}"
        )
    patterns_by_key = {
        str(row.get("canonical_key") or ""): dict(row)
        for row in existing_patterns
        if row.get("canonical_key")
    }
    patterns_by_id = {
        str(row.get("pattern_id") or ""): dict(row)
        for row in existing_patterns
        if row.get("pattern_id")
    }
    resolved_patterns: dict[str, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []
    cohort_trace_count = max(1, len({row.trace_id for row in rows if row.name == "rlm_review"}))

    for index, observation in enumerate(observations):
        values = _required_text(
            observation,
            ("category", "title", "root_cause", "observation_summary"),
        )
        key = _canonical_key(observation)
        operation = str(observation.get("operation") or "create").strip().lower()
        if operation not in {"create", "reinforce", "contradict", "merge", "split"}:
            raise ValueError(f"pattern[{index}] has unsupported operation {operation!r}")
        requested_id = str(observation.get("pattern_id") or "").strip()
        existing = patterns_by_id.get(requested_id) or patterns_by_key.get(key)
        if requested_id and existing is None and operation not in {"create", "split"}:
            raise ValueError(f"pattern[{index}] references unknown pattern_id {requested_id!r}")
        pattern_id = str((existing or {}).get("pattern_id") or pattern_id_for(agent, key))
        trace_ids = _resolve_trace_ids(observation, rows)
        severity = _float01(observation.get("severity"), "severity")
        confidence = _float01(observation.get("confidence"), "confidence")
        trend_label = str(observation.get("trend_label") or "new").strip().lower()
        if trend_label not in _TREND_SCORES:
            raise ValueError(f"pattern[{index}] has invalid trend_label {trend_label!r}")
        prior_trace_ids = read_pattern_event_trace_ids(
            client,
            config.warehouse_id,
            config.catalog,
            config.schema,
            agent,
            pattern_id,
        )
        all_trace_ids = prior_trace_ids | set(trace_ids)
        previous_cohort_count = int((existing or {}).get("cohort_count") or 0)
        last_seen = str((existing or {}).get("last_seen_cohort_id") or "")
        cohort_count = previous_cohort_count + (0 if last_seen == cohort_id else 1)
        total_support = len(all_trace_ids)
        supported = len(set(trace_ids)) >= config.pattern_min_current_traces or (
            total_support >= config.pattern_min_total_traces
            and cohort_count >= config.pattern_min_cohorts
        )
        prior_status = str((existing or {}).get("status") or "")
        status = "active" if supported else "emerging"
        if prior_status in {"queued", "monitoring"} and operation != "contradict":
            status = prior_status
        if operation == "contradict" and trend_label == "falling":
            status = "dormant"
        pattern_row = {
            "pattern_id": pattern_id,
            "canonical_key": key,
            "category": values["category"],
            "title": values["title"],
            "root_cause": values["root_cause"],
            "status": status,
            "first_seen_cohort_id": str(
                (existing or {}).get("first_seen_cohort_id") or cohort_id
            ),
            "last_seen_cohort_id": cohort_id,
            "cohort_count": cohort_count,
            "distinct_trace_count": total_support,
            "recent_trace_count": len(set(trace_ids)),
            "recent_prevalence": len(set(trace_ids)) / cohort_trace_count,
            "severity": severity,
            "confidence": confidence,
            "trend_score": _TREND_SCORES[trend_label],
            "trend_label": trend_label,
            "current_action_id": (existing or {}).get("current_action_id"),
            "created_at": str((existing or {}).get("created_at") or created_at),
            "updated_at": created_at,
            "source_trace_ids": trace_ids,
        }
        resolved_patterns[key] = pattern_row
        resolved_patterns[pattern_id] = pattern_row
        patterns_by_key[key] = pattern_row
        patterns_by_id[pattern_id] = pattern_row
        evidence_ids = sorted(
            {
                evidence_id
                for row in rows
                if row.trace_id in trace_ids
                and (
                    evidence_id := evidence_lookup.get((row.trace_id, row.name, row.created_at))
                )
            }
        )
        event_type = "created" if existing is None else {
            "create": "reinforced",
            "reinforce": "reinforced",
            "contradict": "contradicted",
            "merge": "merged",
            "split": "split",
        }[operation]
        events.append(
            {
                "event_id": stable_id(
                    "recommendation-pattern-event",
                    cohort_id,
                    pattern_id,
                    event_type,
                    *sorted(evidence_ids),
                ),
                "pattern_id": pattern_id,
                "cohort_id": cohort_id,
                "event_type": event_type,
                "evidence_ids": evidence_ids,
                "source_trace_ids": trace_ids,
                "observation_summary": values["observation_summary"],
                "severity": severity,
                "confidence": confidence,
                "created_at": created_at,
            }
        )

    active_actions = [
        action for action in existing_actions if str(action.get("status")) in _ACTIVE_QUEUE_STATES
    ]
    exact_actions = {
        str(action.get("canonical_action_key") or ""): action for action in existing_actions
    }
    planned: list[tuple[dict[str, Any], list[dict[str, Any]], list[str], str]] = []
    covered: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for index, candidate in enumerate(action_candidates):
        _required_text(candidate, ("category", "title", "rationale", "implementation_plan"))
        key = _canonical_key(candidate)
        pattern_keys = candidate.get("pattern_keys") or []
        if isinstance(pattern_keys, str):
            pattern_keys = [pattern_keys]
        linked = [resolved_patterns.get(str(pattern_key)) for pattern_key in pattern_keys]
        if not linked or any(pattern is None for pattern in linked):
            raise ValueError(f"action[{index}] must reference patterns observed in this cohort")
        linked_patterns = [pattern for pattern in linked if pattern is not None]
        candidate_trace_ids = _resolve_trace_ids(candidate, rows)
        pattern_trace_ids = {
            trace_id
            for pattern in linked_patterns
            for trace_id in pattern["source_trace_ids"]
        }
        trace_ids = sorted(set(candidate_trace_ids) | pattern_trace_ids)
        current_support = len(set(trace_ids))
        total_support = max(int(pattern["distinct_trace_count"]) for pattern in linked_patterns)
        cohort_support = max(int(pattern["cohort_count"]) for pattern in linked_patterns)
        criticality = str(candidate.get("criticality") or "").lower()
        eligible = current_support >= config.pattern_min_current_traces or (
            total_support >= config.pattern_min_total_traces
            and cohort_support >= config.pattern_min_cohorts
        )
        if criticality in {"security", "privacy", "data_loss"}:
            eligible = True
        if not eligible:
            continue
        existing = exact_actions.get(key)
        if existing is None:
            title = str(candidate.get("title") or "")
            similar = [
                action
                for action in active_actions
                if difflib.SequenceMatcher(
                    a=title.casefold(), b=str(action.get("title") or "").casefold()
                ).ratio()
                >= 0.88
            ]
            existing = similar[0] if len(similar) == 1 else None
        if existing is not None:
            covered.append((existing, linked_patterns))
            continue
        planned.append((candidate, linked_patterns, trace_ids, key))

    if len(planned) > config.max_recommendations:
        raise ValueError(
            f"validated batch has {len(planned)} new actions; maximum is "
            f"{config.max_recommendations}"
        )

    # All model references and promotion rules are valid before the first write.
    for event in events:
        merge_pattern_event(
            client,
            config.warehouse_id,
            config.catalog,
            config.schema,
            agent,
            event,
        )
    unique_pattern_rows = {row["pattern_id"]: row for row in resolved_patterns.values()}
    for pattern in unique_pattern_rows.values():
        merge_pattern(
            client,
            config.warehouse_id,
            config.catalog,
            config.schema,
            agent,
            pattern,
        )

    for existing, linked_patterns in covered:
        action_id = str(existing.get("action_id") or action_id_for(
            agent, str(existing["canonical_action_key"])
        ))
        for pattern in linked_patterns:
            merge_action_pattern(
                client,
                config.warehouse_id,
                config.catalog,
                config.schema,
                agent,
                action_id=action_id,
                pattern_id=pattern["pattern_id"],
                relation="covered_by",
                cohort_id=cohort_id,
                now=created_at,
            )

    for candidate, linked_patterns, trace_ids, key in planned:
        pattern_ids = [pattern["pattern_id"] for pattern in linked_patterns]
        proposal = _candidate_to_proposal(
            candidate,
            agent=agent,
            rows=rows,
            trace_ids=trace_ids,
            pattern_ids=pattern_ids,
            cohort_id=cohort_id,
            created_at=created_at,
            planner_model=config.model,
        )
        inserted = insert_proposal_if_absent(
            proposal,
            client=client,
            warehouse_id=config.warehouse_id,
            catalog=config.catalog,
            schema=config.schema,
            generated_at=created_at,
        )
        action_id = action_id_for(agent, key)
        values = _required_text(candidate, ("category", "title", "implementation_plan"))
        merge_action(
            client,
            config.warehouse_id,
            config.catalog,
            config.schema,
            agent,
            {
                "action_id": action_id,
                "canonical_action_key": key,
                "category": values["category"],
                "title": values["title"],
                "plan": values["implementation_plan"],
                "status": "queued",
                "proposal_id": proposal.proposal_id,
                "first_proposed_cohort_id": cohort_id,
                "last_supported_cohort_id": cohort_id,
                "created_at": created_at,
                "updated_at": created_at,
            },
        )
        for pattern in linked_patterns:
            merge_action_pattern(
                client,
                config.warehouse_id,
                config.catalog,
                config.schema,
                agent,
                action_id=action_id,
                pattern_id=pattern["pattern_id"],
                relation="addresses",
                cohort_id=cohort_id,
                now=created_at,
            )
            pattern["current_action_id"] = action_id
            pattern["status"] = "queued"
            merge_pattern(
                client,
                config.warehouse_id,
                config.catalog,
                config.schema,
                agent,
                pattern,
            )
        if inserted:
            tally.written += 1
            tally.proposals.append(proposal)
    return (
        f"stored {len(unique_pattern_rows)} pattern(s); queued {tally.written} new "
        f"recommendation(s); linked {len(covered)} covered action(s)"
    )


def run_for_agent(agent: Agent, config: PlannerConfig, *, deps: PlannerDeps) -> PlannerReport:
    if not agent.annotations_table:
        raise ValueError(f"agent {agent.agent_name!r} has no annotations_table")
    _ensure_state_tables(deps.client, config)
    sync_at = deps.now()
    existing_actions = _sync_queue_memory(
        deps.client,
        config,
        agent,
        _read_existing_actions(deps.client, config, agent),
        now=sync_at,
    )
    before = read_ingestion_watermark(
        deps.client, config.warehouse_id, config.catalog, config.schema, agent
    )
    judge_names = _judge_names_for_agent(agent)
    raw_rows = read_assessments(
        deps.client,
        config.warehouse_id,
        annotations_table=agent.annotations_table,
        since_created_at=before,
        max_results=config.max_assessments,
        judge_names=judge_names,
        ascending=True,
    )
    after = before
    if raw_rows:
        ingested_at = deps.now()
        merge_evidence(
            deps.client,
            config.warehouse_id,
            config.catalog,
            config.schema,
            agent,
            build_evidence_items(
                agent,
                raw_rows,
                reserved=deps.reserved,
                ingested_at=ingested_at,
            ),
        )
        after = max_created_at(raw_rows) or before or ingested_at
        write_ingestion_watermark(
            deps.client,
            config.warehouse_id,
            config.catalog,
            config.schema,
            agent,
            last_created_at=after,
            updated_at=ingested_at,
            n_assessments=len(raw_rows),
        )

    trace_ids = read_eligible_trace_ids(
        deps.client,
        config.warehouse_id,
        config.catalog,
        config.schema,
        agent,
        judge_names=sorted(judge_names),
        judge_grace_minutes=config.judge_grace_minutes,
        max_traces=config.max_traces,
    )
    if len(trace_ids) < config.min_traces:
        return PlannerReport(
            agent.agent_name,
            len(raw_rows),
            len(subject_assessments(raw_rows)),
            len(trace_ids),
            0,
            before,
            after,
            note=f"buffering durable cohort: {len(trace_ids)}/{config.min_traces} completed traces",
        )

    evidence = read_evidence_for_traces(
        deps.client,
        config.warehouse_id,
        config.catalog,
        config.schema,
        agent,
        trace_ids,
    )
    evidence_ids = [evidence_id for evidence_id, _row in evidence]
    rows = [row for _evidence_id, row in evidence]
    if len({row.trace_id for row in rows if row.name == "rlm_review"}) < config.min_traces:
        raise RuntimeError("eligible cohort lost its RLM-complete trace floor during snapshot")
    cohort_id = cohort_id_for(agent, trace_ids, evidence_ids)
    sequence = next_cohort_sequence(
        deps.client, config.warehouse_id, config.catalog, config.schema, agent
    )
    started_at = deps.now()
    begin_cohort(
        deps.client,
        config.warehouse_id,
        config.catalog,
        config.schema,
        agent,
        cohort_id=cohort_id,
        sequence=sequence,
        min_traces=config.min_traces,
        trace_ids=trace_ids,
        evidence_ids=evidence_ids,
        evidence_cutoff_at=max_created_at(rows) or started_at,
        queue_snapshot_at=sync_at,
        planner_model=config.model,
        planner_prompt_version=PROMPT_VERSION,
        planner_run_id=stable_id("recommendation-run", cohort_id, started_at),
        created_at=started_at,
    )
    patterns = read_patterns(
        deps.client, config.warehouse_id, config.catalog, config.schema, agent
    )
    lookup = {
        (row.trace_id, row.name, row.created_at): evidence_id for evidence_id, row in evidence
    }
    tally = PlannerTally()
    try:
        if deps.recommend is None:
            _default_recommend(
                agent,
                rows,
                config=config,
                tally=tally,
                evidence_ids=lookup,
                patterns=patterns,
                existing_actions=existing_actions,
            )
        else:
            deps.recommend(rows, tally)
        if tally.errors:
            raise RuntimeError("recommendation submission failed: " + "; ".join(tally.errors))
        if tally.submitted != 1:
            raise RuntimeError("recommendation agent must call submit_cohort_analysis exactly once")
        summary = apply_cohort_analysis(
            agent=agent,
            rows=rows,
            evidence_lookup=lookup,
            cohort_id=cohort_id,
            existing_patterns=patterns,
            existing_actions=existing_actions,
            client=deps.client,
            config=config,
            tally=tally,
            created_at=deps.now(),
        )
        assign_evidence_to_cohort(
            deps.client,
            config.warehouse_id,
            config.catalog,
            config.schema,
            agent,
            cohort_id,
            evidence_ids,
        )
        finish_cohort(
            deps.client,
            config.warehouse_id,
            config.catalog,
            config.schema,
            agent,
            cohort_id,
            status="committed",
            completed_at=deps.now(),
        )
    except Exception as exc:
        finish_cohort(
            deps.client,
            config.warehouse_id,
            config.catalog,
            config.schema,
            agent,
            cohort_id,
            status="failed",
            completed_at=deps.now(),
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    return PlannerReport(
        agent.agent_name,
        len(raw_rows),
        len(rows),
        len(set(trace_ids)),
        tally.written,
        before,
        after,
        cohort_id=cohort_id,
        note=summary,
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan queue-aware recommendations from multi-trace RLM/judge cohorts"
    )
    parser.add_argument("--warehouse-id", default=os.environ.get("AIL_WAREHOUSE_ID", ""))
    parser.add_argument("--catalog", default=os.environ.get("AIL_CATALOG", ""))
    parser.add_argument("--schema", default=os.environ.get("AIL_SCHEMA", ""))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--max-assessments", type=int, default=500)
    parser.add_argument("--min-traces", type=int, default=10)
    parser.add_argument("--max-traces", type=int, default=25)
    parser.add_argument("--judge-grace-minutes", type=int, default=30)
    parser.add_argument("--max-recommendations", type=int, default=3)
    parser.add_argument("--pattern-min-current-traces", type=int, default=3)
    parser.add_argument("--pattern-min-total-traces", type=int, default=5)
    parser.add_argument("--pattern-min-cohorts", type=int, default=2)
    parser.add_argument("--task-suite-version", default="v1")
    parser.add_argument("--groundtruth-root", default="")
    parser.add_argument(
        "--token-secret-scope", default=os.environ.get("AIL_TOKEN_SECRET_SCOPE", "")
    )
    parser.add_argument("--token-secret-key", default=os.environ.get("AIL_TOKEN_SECRET_KEY", ""))
    args = parser.parse_args(argv)
    missing = [
        name
        for name, value in (
            ("--warehouse-id", args.warehouse_id),
            ("--catalog", args.catalog),
            ("--schema", args.schema),
        )
        if not value
    ]
    if missing:
        parser.error(f"missing required arg(s): {', '.join(missing)}")
    if args.min_traces < 1 or args.max_traces < args.min_traces:
        parser.error("--max-traces must be >= --min-traces >= 1")
    for name in (
        "max_recommendations",
        "pattern_min_current_traces",
        "pattern_min_total_traces",
        "pattern_min_cohorts",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    return args


def main(argv: list[str] | None = None) -> int:
    from ail.jobs.publish_job import resolve_job_auth
    from ail.publish import _build_workspace_client

    args = _parse_args(argv)
    missing = missing_registry_target(args.warehouse_id, args.catalog, args.schema)
    if missing:
        print(f"{_TAG} registry mode requires {', '.join(missing)}", file=sys.stderr)
        return 2
    auth = resolve_job_auth(
        token_secret_scope=args.token_secret_scope or None,
        token_secret_key=args.token_secret_key or None,
    )
    client = _build_workspace_client(None)
    reserved = resolve_reserved_pools(
        task_suite_version=args.task_suite_version,
        groundtruth_root=args.groundtruth_root or None,
    )
    print(
        f"{_TAG} auth={auth} model={args.model} min_traces={args.min_traces} "
        f"max_traces={args.max_traces} max_recommendations={args.max_recommendations}"
    )
    config = PlannerConfig(
        warehouse_id=args.warehouse_id,
        catalog=args.catalog,
        schema=args.schema,
        model=args.model,
        max_turns=args.max_turns,
        max_assessments=args.max_assessments,
        min_traces=args.min_traces,
        max_traces=args.max_traces,
        judge_grace_minutes=args.judge_grace_minutes,
        max_recommendations=args.max_recommendations,
        pattern_min_current_traces=args.pattern_min_current_traces,
        pattern_min_total_traces=args.pattern_min_total_traces,
        pattern_min_cohorts=args.pattern_min_cohorts,
        task_suite_version=args.task_suite_version,
        groundtruth_root=args.groundtruth_root or None,
    )
    agents = load_registered_agents(
        warehouse_id=args.warehouse_id,
        catalog=args.catalog,
        schema=args.schema,
        client=client,
    )

    def one(agent: Agent) -> int:
        report = run_for_agent(
            agent, config, deps=PlannerDeps(client=client, reserved=reserved)
        )
        print(f"{_TAG} {report}")
        return 0

    result = run_for_each_registered_agent(agents, one, job_name="ail.jobs.recommendation_planner")
    return result.worst_rc


if __name__ == "__main__":
    rc = main()
    if rc != 0:
        sys.exit(rc)
