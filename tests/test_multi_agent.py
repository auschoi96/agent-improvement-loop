"""Load-bearing properties of the shared multi-agent fan-out (:mod:`ail.jobs.multi_agent`).

Offline and injectable throughout — no client, no workspace, no live call. These
cover the guarantees every registry-driven job inherits from the shared core:

* per-agent isolation (one agent's failure is recorded + logged, the rest STILL run);
* an empty registry is a clean no-op (worst_rc 0, no fabricated work);
* the worst-case return code is non-zero iff ANY agent failed (raise OR non-zero rc);
* each agent's OWN identity is threaded to its per-agent call.
"""

from __future__ import annotations

import io

from ail.jobs.multi_agent import (
    MultiAgentResult,
    missing_registry_target,
    run_for_each_registered_agent,
)
from ail.registry import Agent


def _agent(name: str, exp: str, **over: object) -> Agent:
    return Agent(agent_name=name, experiment_id=exp, **over)  # type: ignore[arg-type]


def test_per_agent_isolation_middle_agent_failure_does_not_abort_others() -> None:
    agents = [_agent("a1", "e1"), _agent("a2", "e2"), _agent("a3", "e3")]
    attempted: list[str] = []
    err = io.StringIO()

    def per_agent(agent: Agent) -> int:
        attempted.append(agent.agent_name)
        if agent.agent_name == "a2":
            raise RuntimeError("boom in a2")
        return 0

    result = run_for_each_registered_agent(agents, per_agent, job_name="test", stderr=err)

    # All three were ATTEMPTED — the middle failure never aborted the loop.
    assert attempted == ["a1", "a2", "a3"]
    assert result.attempted == ("a1", "a2", "a3")
    # The failure is recorded, and worst_rc is non-zero.
    assert result.n_failed == 1
    assert result.worst_rc == 1
    (failure,) = result.failures
    assert failure.agent_name == "a2"
    assert "boom in a2" in (failure.error or "")
    # Logged LOUDLY to stderr, naming the agent.
    logged = err.getvalue()
    assert "a2" in logged and "FAILED" in logged


def test_empty_registry_is_a_clean_no_op() -> None:
    result = run_for_each_registered_agent([], lambda a: 0, job_name="test")
    assert isinstance(result, MultiAgentResult)
    assert result.n_agents == 0
    assert result.attempted == ()
    assert result.worst_rc == 0  # a clean no-op, NOT an error


def test_none_return_is_treated_as_success() -> None:
    result = run_for_each_registered_agent([_agent("a1", "e1")], lambda a: None, job_name="test")
    assert result.worst_rc == 0
    assert result.n_failed == 0


def test_non_zero_rc_without_raising_is_a_failure() -> None:
    # auto-align's per-agent body returns 1 when a judge's cadence failed (it does
    # not raise). That must count as a failure and bump worst_rc.
    agents = [_agent("a1", "e1"), _agent("a2", "e2")]

    def per_agent(agent: Agent) -> int:
        return 1 if agent.agent_name == "a2" else 0

    result = run_for_each_registered_agent(agents, per_agent, job_name="test", stderr=io.StringIO())
    assert result.worst_rc == 1
    assert result.n_failed == 1
    assert result.attempted == ("a1", "a2")


def test_each_agent_own_identity_is_threaded() -> None:
    agents = [
        _agent("a1", "e1", goal_config={"objective_metric": "total_tokens"}),
        _agent("a2", "e2", annotations_table="cat.sch.a2"),
    ]
    seen: list[tuple[str, str]] = []

    def per_agent(agent: Agent) -> int:
        seen.append((agent.agent_name, agent.experiment_id))
        return 0

    run_for_each_registered_agent(agents, per_agent, job_name="test")
    # Agent a1 got e1 (not a shared/global experiment); a2 got e2.
    assert seen == [("a1", "e1"), ("a2", "e2")]


def test_all_failing_still_attempts_all_and_worst_rc_nonzero() -> None:
    agents = [_agent("a1", "e1"), _agent("a2", "e2"), _agent("a3", "e3")]
    attempted: list[str] = []

    def per_agent(agent: Agent) -> int:
        attempted.append(agent.agent_name)
        raise ValueError(f"fail {agent.agent_name}")

    result = run_for_each_registered_agent(agents, per_agent, job_name="test", stderr=io.StringIO())
    assert attempted == ["a1", "a2", "a3"]
    assert result.n_failed == 3
    assert result.worst_rc == 1


def test_missing_registry_target_flags_empties() -> None:
    assert missing_registry_target("wh", "cat", "sch") == []
    assert missing_registry_target("", "cat", "sch") == ["--warehouse-id"]
    assert missing_registry_target(None, "", None) == ["--warehouse-id", "--catalog", "--schema"]
