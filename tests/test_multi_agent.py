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

import pytest
from databricks.sdk.service.sql import StatementState

from ail.jobs.multi_agent import (
    MultiAgentResult,
    missing_registry_target,
    resolve_registered_agent,
    run_for_each_registered_agent,
)
from ail.publish_versions import REGISTRY_COLUMNS, _registry_row
from ail.registry import Agent


def _agent(name: str, exp: str, **over: object) -> Agent:
    return Agent(agent_name=name, experiment_id=exp, **over)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# A stub UC client: serves the SAME serialization ``_registry_row`` writes, so the
# resolver is exercised through the REAL read path (load_registered_agents_full ->
# _query_registry_rows) with no live warehouse. Mirrors test_publish_versions' fake.
# ---------------------------------------------------------------------------


class _Col:
    def __init__(self, name: str) -> None:
        self.name = name


class _Schema:
    def __init__(self, cols: list[str]) -> None:
        self.columns = [_Col(c) for c in cols]


class _Manifest:
    def __init__(self, cols: list[str]) -> None:
        self.schema = _Schema(cols)


class _ResultData:
    def __init__(self, data: list[list]) -> None:  # type: ignore[type-arg]
        self.data_array = data


class _Err:
    def __init__(self, message: str) -> None:
        self.message = message


class _Status:
    def __init__(self, state: StatementState, err: _Err | None = None) -> None:
        self.state = state
        self.error = err


class _Resp:
    def __init__(
        self,
        state: StatementState,
        *,
        cols: list[str] | None = None,
        data: list[list] | None = None,  # type: ignore[type-arg]
        err: _Err | None = None,
    ) -> None:
        self.statement_id = "stmt"
        self.status = _Status(state, err)
        self.manifest = _Manifest(cols) if cols else None
        self.result = _ResultData(data or []) if cols else None


class _StmtExec:
    def __init__(self, resp: _Resp) -> None:
        self._resp = resp

    def execute_statement(self, *, warehouse_id, statement, wait_timeout=None):  # type: ignore[no-untyped-def]
        return self._resp

    def get_statement(self, statement_id):  # type: ignore[no-untyped-def]
        return self._resp


class _StubUcClient:
    """A workspace client whose registry read returns exactly the given rows."""

    def __init__(self, resp: _Resp) -> None:
        self.statement_execution = _StmtExec(resp)


def _registry_client(*agents: Agent) -> _StubUcClient:
    rows = [_registry_row(a, generated_at="t") for a in agents]
    return _StubUcClient(_Resp(StatementState.SUCCEEDED, cols=list(REGISTRY_COLUMNS), data=rows))


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


# ---------------------------------------------------------------------------
# resolve_registered_agent: the shared UC resolver the companion (planner + executor)
# and the scheduled jobs read the single source of truth through.
# ---------------------------------------------------------------------------


def test_resolve_ui_onboarded_agent_from_uc_only() -> None:
    # A UI-onboarded agent present ONLY in UC (never in any YAML): resolvable by name,
    # with experiment_id + target_workspace + goal_config coming straight from UC.
    ui_agent = _agent(
        "ui_onboarded",
        "exp-999",
        target_workspace="/repos/ui_onboarded",
        goal_config={"objective_metric": "answer_quality", "goal_direction": "maximize"},
    )
    client = _registry_client(_agent("other", "exp-1"), ui_agent)

    got = resolve_registered_agent(
        "ui_onboarded", warehouse_id="wh", catalog="cat", schema="sch", client=client
    )
    # experiment_id + target_workspace resolved from UC (not a local YAML / guessed value)...
    assert got.experiment_id == "exp-999"
    assert got.target_workspace == "/repos/ui_onboarded"
    # ...and the whole entry round-trips, generic to any goal dimension.
    assert got.goal_config == {"objective_metric": "answer_quality", "goal_direction": "maximize"}
    assert got == ui_agent


def test_resolve_absent_agent_in_present_registry_fails_closed() -> None:
    # The registry is present and has agents, but not the requested one -> clear raise,
    # never a fabricated Agent with a guessed experiment.
    client = _registry_client(_agent("a1", "e1"), _agent("a2", "e2"))
    with pytest.raises(KeyError) as exc:
        resolve_registered_agent(
            "not_registered", warehouse_id="wh", catalog="cat", schema="sch", client=client
        )
    msg = str(exc.value)
    assert "not_registered" in msg
    assert "a1" in msg and "a2" in msg  # names what IS registered
    assert "cat.sch.agent_registry" in msg


def test_resolve_empty_registry_fails_closed() -> None:
    # A not-yet-created registry table reads back as empty -> resolving anything raises
    # (nothing to resolve; never fabricated). Distinct message from the not-in-table case.
    absent = _Resp(
        StatementState.FAILED,
        err=_Err(
            "[TABLE_OR_VIEW_NOT_FOUND] The table or view `cat`.`sch`.agent_registry "
            "cannot be found."
        ),
    )
    client = _StubUcClient(absent)
    with pytest.raises(KeyError) as exc:
        resolve_registered_agent(
            "claude_code", warehouse_id="wh", catalog="cat", schema="sch", client=client
        )
    assert "registry table absent or empty" in str(exc.value)


def test_resolve_infra_error_propagates_not_swallowed() -> None:
    # A real permission / warehouse error must PROPAGATE (never read as "no such agent").
    broken = _Resp(StatementState.FAILED, err=_Err("PERMISSION_DENIED on warehouse abc"))
    client = _StubUcClient(broken)
    with pytest.raises(RuntimeError):
        resolve_registered_agent(
            "claude_code", warehouse_id="wh", catalog="cat", schema="sch", client=client
        )
