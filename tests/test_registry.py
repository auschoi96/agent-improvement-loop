"""Unit tests for the agent registry (typed config + loader)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ail.cohorts import TAG_AGENT
from ail.registry import (
    CLAUDE_CODE_EXPERIMENT_ID,
    DEFAULT_REGISTRY,
    Agent,
    AgentRegistry,
    load_registry,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_registry_seeds_claude_code() -> None:
    agent = DEFAULT_REGISTRY.get("claude_code")
    assert agent.experiment_id == CLAUDE_CODE_EXPERIMENT_ID == "660599403165942"
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


def test_agent_cohort_uses_agent_tag_by_default() -> None:
    cohort = DEFAULT_REGISTRY.get("claude_code").cohort()
    assert cohort.name == "claude_code"
    # The cohort selects traces tagged ail.agent = claude_code.
    flt = cohort.to_mlflow_filter()
    assert flt == f"tags.`{TAG_AGENT}` = 'claude_code'"


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
