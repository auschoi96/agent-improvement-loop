"""Unit tests for the agent registry (typed config + loader)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ail.registry import (
    CLAUDE_CODE_EXPERIMENT_ID,
    CLAUDE_CODE_REVIEWER_EXPERIMENT_ID,
    DEFAULT_REGISTRY,
    Agent,
    AgentRegistry,
    OptimizationTarget,
    load_registry,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_registry_seeds_claude_code() -> None:
    agent = DEFAULT_REGISTRY.get("claude_code")
    assert agent.experiment_id == CLAUDE_CODE_EXPERIMENT_ID == "1301765275062543"
    assert agent.reviewer_experiment_id == CLAUDE_CODE_REVIEWER_EXPERIMENT_ID
    assert DEFAULT_REGISTRY.names() == ["claude_code"]
    assert len(DEFAULT_REGISTRY) == 1
    assert [a.agent_name for a in DEFAULT_REGISTRY] == ["claude_code"]


def test_get_missing_raises_with_helpful_message() -> None:
    with pytest.raises(KeyError, match="no agent named 'nope'"):
        DEFAULT_REGISTRY.get("nope")


def test_duplicate_agent_name_is_loud() -> None:
    with pytest.raises(ValueError, match="duplicate agent_name"):
        AgentRegistry(
            agents=[
                Agent(agent_name="dup", experiment_id="1"),
                Agent(agent_name="dup", experiment_id="2"),
            ]
        )


def test_unknown_field_is_rejected() -> None:
    # extra='forbid' makes a config typo loud rather than silently dropped.
    with pytest.raises(ValueError):
        Agent.model_validate({"agent_name": "x", "experiment_id": "1", "experimnt_id": "typo"})


def test_agent_target_workspace_absent_is_none() -> None:
    # Optional at the model level: "not configured yet" is a clean None (a registry entry
    # is valid before the executor is wired).
    agent = Agent(agent_name="x", experiment_id="1")
    assert agent.target_workspace is None


def test_agent_target_workspace_present_is_carried() -> None:
    # User-provided (the target agent's own repo the executor edits + snapshots). L7b-1
    # only carries it; it neither runs the executor nor validates the path exists.
    agent = Agent(agent_name="x", experiment_id="1", target_workspace="/repos/my-agent")
    assert agent.target_workspace == "/repos/my-agent"


def test_agent_target_workspace_typo_is_loud() -> None:
    # extra='forbid': a near-miss field name (target_workspaces) fails rather than being
    # silently dropped, so a misconfigured executor target is caught at load time.
    with pytest.raises(ValueError):
        Agent.model_validate(
            {"agent_name": "x", "experiment_id": "1", "target_workspaces": "/repos/typo"}
        )


def test_agent_target_workspace_round_trips_json() -> None:
    agent = Agent(agent_name="x", experiment_id="1", target_workspace="/repos/my-agent")
    restored = Agent.model_validate_json(agent.model_dump_json())
    assert restored == agent


def test_optimization_target_is_relative_tokenized_and_round_trips() -> None:
    target = OptimizationTarget(
        path=".claude/skills/my-agent/SKILL.md",
        validation_command='python -m pytest -q "tests/test agent.py"',
    )
    assert target.validation_command == [
        "python",
        "-m",
        "pytest",
        "-q",
        "tests/test agent.py",
    ]
    agent = Agent(
        agent_name="x",
        experiment_id="1",
        target_workspace="/repos/x",
        optimization_target=target,
    )
    assert Agent.model_validate_json(agent.model_dump_json()) == agent


@pytest.mark.parametrize("path", ["/absolute/SKILL.md", "../escape.md", "a/../../escape.md"])
def test_optimization_target_rejects_unsafe_path(path: str) -> None:
    with pytest.raises(ValueError, match="project-relative"):
        OptimizationTarget(path=path, validation_command=["true"])


def test_agent_goal_config_and_annotations_table_absent_are_none() -> None:
    # Optional at the model level: a registry entry is valid before the continuous-RLM
    # goal or the distiller's annotations table are configured (Slice 4 populates them).
    agent = Agent(agent_name="x", experiment_id="1")
    assert agent.goal_config is None
    assert agent.annotations_table is None


def test_agent_goal_config_and_annotations_table_are_carried() -> None:
    # goal_config carries the continuous_rlm goal knobs (free-form, symmetric with
    # judge_config); annotations_table is the memory_distiller's UC table.
    goal = {
        "objective_metric": "total_tokens",
        "goal_direction": "decrease",
        "goal_target": 0.2,
        "goal_target_kind": "relative",
        "guardrail_judge": "correctness",
    }
    agent = Agent(
        agent_name="x",
        experiment_id="1",
        goal_config=goal,
        annotations_table="cat.sch.x_annotations",
    )
    assert agent.goal_config == goal
    assert agent.annotations_table == "cat.sch.x_annotations"


def test_agent_goal_config_and_annotations_table_round_trip_json() -> None:
    agent = Agent(
        agent_name="x",
        experiment_id="1",
        goal_config={"objective_metric": "total_tokens", "goal_direction": "decrease"},
        annotations_table="cat.sch.x_annotations",
    )
    restored = Agent.model_validate_json(agent.model_dump_json())
    assert restored == agent


def test_agent_annotations_table_typo_is_loud() -> None:
    # extra='forbid': a near-miss field name fails at load rather than being dropped.
    with pytest.raises(ValueError):
        Agent.model_validate(
            {"agent_name": "x", "experiment_id": "1", "annotation_table": "cat.sch.typo"}
        )


def test_agent_cohort_uses_whole_dedicated_experiment_by_default() -> None:
    cohort = DEFAULT_REGISTRY.get("claude_code").cohort()
    assert cohort.name == "claude_code"
    assert cohort.tag_filter.clauses == ()
    assert cohort.to_mlflow_filter() is None


def test_agent_cohort_uses_tag_filter_when_set() -> None:
    agent = Agent(
        agent_name="mas",
        experiment_id="42",
        tag_filter={"ail.agent": "mas", "ail.cohort": "nightly"},
    )
    cohort = agent.cohort()
    assert cohort.name == "mas"
    # both equality clauses are AND'd into the pushdown filter
    flt = cohort.to_mlflow_filter() or ""
    assert "tags.`ail.agent` = 'mas'" in flt
    assert "tags.`ail.cohort` = 'nightly'" in flt


def test_round_trip_json() -> None:
    restored = AgentRegistry.model_validate_json(DEFAULT_REGISTRY.model_dump_json())
    assert restored == DEFAULT_REGISTRY


def test_load_registry_none_returns_default() -> None:
    assert load_registry(None) == DEFAULT_REGISTRY


def test_committed_yaml_matches_default_seed() -> None:
    # config/agents.yaml is the operator-facing mirror of DEFAULT_REGISTRY; keep
    # them in sync (mirrors how the frozen Task Suite mirrors its seed).
    loaded = load_registry(REPO_ROOT / "config" / "agents.yaml")
    assert loaded == DEFAULT_REGISTRY
