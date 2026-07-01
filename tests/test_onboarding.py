"""Tests for the in-app onboarding wizard's server side (:mod:`ail.onboarding`).

All offline: the two live dependencies — MLflow (experiment validate/create) and
the SQL warehouse (registry read/write) — are injected away with fakes, mirroring
the injectable-client seams in ``ail.jobs.readiness_preflight`` /
``ail.publish_versions``. The headline invariants under test are the slice's
contract: the goal→scorer mapping and the real readiness floors are surfaced from
the Python source of truth (never re-derived), and every write is **fail-closed** —
a non-fresh experiment is rejected, a denied create is an honest error, and a
registry write failure never reports a registered agent.
"""

from __future__ import annotations

import pytest
from databricks.sdk.service.sql import StatementState

from ail.onboarding.experiment import (
    ExperimentCreation,
    ExperimentInfo,
    ExperimentPermissionError,
    create_experiment,
    validate_experiment,
)
from ail.onboarding.goals import (
    GOAL_CATALOG,
    GateName,
    GoalKey,
    UnknownGoalError,
    build_judge_config,
    build_requirements,
)
from ail.onboarding.service import (
    ErrorResult,
    OnboardingOutcome,
    RegisterResult,
    load_registered_agents,
    register_agent,
    run_action,
    run_create,
    run_register,
    run_requirements,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeExperimentClient:
    """A canned :class:`~ail.onboarding.experiment.ExperimentClient`.

    ``by_id`` / ``by_name`` are the visible experiments; ``traces`` maps an
    experiment id to its trace count; ``created`` records create calls. Any method
    can be told to ``raise`` to exercise the fail-closed error paths.
    """

    def __init__(
        self,
        *,
        by_id: dict[str, ExperimentInfo] | None = None,
        by_name: dict[str, ExperimentInfo] | None = None,
        traces: dict[str, int] | None = None,
        create_returns: str | None = "exp-new",
    ) -> None:
        self.by_id = by_id or {}
        self.by_name = by_name or {}
        self.traces = traces or {}
        self.create_returns = create_returns
        self.created: list[str] = []

    def get_experiment(self, experiment_id):  # type: ignore[no-untyped-def]
        return self.by_id.get(experiment_id)

    def get_experiment_by_name(self, name):  # type: ignore[no-untyped-def]
        return self.by_name.get(name)

    def create_experiment(self, name):  # type: ignore[no-untyped-def]
        self.created.append(name)
        return self.create_returns

    def count_traces(self, experiment_id, *, limit):  # type: ignore[no-untyped-def]
        return min(self.traces.get(experiment_id, 0), limit)


class _RecordingStmtExec:
    """A fake statement-execution that records writes and returns SUCCEEDED."""

    def __init__(self) -> None:
        self.statements: list[str] = []

    def execute_statement(self, *, warehouse_id, statement, wait_timeout=None):  # type: ignore[no-untyped-def]
        self.statements.append(statement)
        return _Resp(StatementState.SUCCEEDED)

    def get_statement(self, statement_id):  # type: ignore[no-untyped-def]
        return _Resp(StatementState.SUCCEEDED)


class _RecordingClient:
    def __init__(self) -> None:
        self.statement_execution = _RecordingStmtExec()


class _RaisingStmtExec:
    def execute_statement(self, *, warehouse_id, statement, wait_timeout=None):  # type: ignore[no-untyped-def]
        raise RuntimeError("warehouse unavailable")

    def get_statement(self, statement_id):  # type: ignore[no-untyped-def]
        raise RuntimeError("warehouse unavailable")


class _RaisingClient:
    """A warehouse client whose first write blows up — the no-fabrication case."""

    def __init__(self) -> None:
        self.statement_execution = _RaisingStmtExec()


# -- read fakes for load_registered_agents ---------------------------------


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
    def __init__(self, data: list[list[str]]) -> None:
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
        data: list[list[str]] | None = None,
        err: _Err | None = None,
    ) -> None:
        self.statement_id = "stmt-1"
        self.status = _Status(state, err)
        self.manifest = _Manifest(cols) if cols else None
        self.result = _ResultData(data or []) if cols else None


class _ReadStmtExec:
    def __init__(self, resp: _Resp) -> None:
        self._resp = resp

    def execute_statement(self, *, warehouse_id, statement, wait_timeout=None):  # type: ignore[no-untyped-def]
        return self._resp

    def get_statement(self, statement_id):  # type: ignore[no-untyped-def]
        return self._resp


class _ReadClient:
    def __init__(self, resp: _Resp) -> None:
        self.statement_execution = _ReadStmtExec(resp)


# ---------------------------------------------------------------------------
# goals: the fixed catalog, the goal→scorer mapping, the real floors
# ---------------------------------------------------------------------------


def test_catalog_is_the_four_fixed_goals() -> None:
    assert set(GOAL_CATALOG) == {
        GoalKey.TOKEN_EFFICIENCY,
        GoalKey.LATENCY,
        GoalKey.ACCURACY,
        GoalKey.COST,
    }
    # Accuracy is the only judged goal; latency/cost are deterministic; token
    # efficiency is deterministic-base with an optional quality judge (hybrid).
    assert GOAL_CATALOG[GoalKey.ACCURACY].requires_quality is True
    assert GOAL_CATALOG[GoalKey.ACCURACY].guardrail_names == ("correctness",)
    assert GOAL_CATALOG[GoalKey.LATENCY].requires_quality is False
    assert GOAL_CATALOG[GoalKey.COST].requires_quality is False
    assert GOAL_CATALOG[GoalKey.TOKEN_EFFICIENCY].optional_quality_judge == "token_efficiency"


def test_requirements_surface_the_real_floors() -> None:
    result = build_requirements(None)
    # Floors come straight from ReadinessThresholds — never invented here.
    assert result.thresholds.prove_min_traces == 50
    assert result.thresholds.quality_min_labels == 20
    assert result.thresholds.scored_coverage_floor == 0.5
    assert len(result.catalog) == 4


def test_deterministic_goal_needs_traces_not_labels() -> None:
    result = build_requirements(["cost"])
    (cost,) = result.selected
    assert cost.requires_labels is False
    gate_names = {g.name for g in cost.gates}
    assert GateName.TRACE_PROVE.value in gate_names
    assert GateName.HUMAN_LABELS.value not in gate_names
    assert result.requires_labels is False


def test_judged_goal_needs_the_20_labels() -> None:
    result = build_requirements(["accuracy"])
    (acc,) = result.selected
    assert acc.requires_labels is True
    labels_gate = next(g for g in acc.gates if g.name == GateName.HUMAN_LABELS.value)
    # The threshold + the "need N" copy are the readiness module's own output.
    assert labels_gate.threshold == 20
    assert "20" in labels_gate.needed
    assert result.requires_labels is True


def test_union_requires_labels_when_any_goal_is_judged() -> None:
    result = build_requirements(["cost", "accuracy"])
    assert result.requires_labels is True
    union_names = {g.name for g in result.union_gates}
    assert GateName.HUMAN_LABELS.value in union_names  # from accuracy
    assert GateName.TRACE_PROVE.value in union_names  # from both


def test_unknown_goal_is_rejected() -> None:
    with pytest.raises(UnknownGoalError):
        build_requirements(["turbo"])
    with pytest.raises(ValueError):
        build_judge_config([])


def test_judge_config_records_the_resolved_mapping() -> None:
    cfg = build_judge_config(["accuracy", "cost"])
    assert cfg["goals"] == ["accuracy", "cost"]
    assert cfg["scorers"]["accuracy"]["guardrail_judges"] == ["correctness"]
    assert cfg["scorers"]["cost"]["scorer_kind"] == "deterministic_l0"


# ---------------------------------------------------------------------------
# experiment: fresh validation + creation (fail-closed)
# ---------------------------------------------------------------------------


def test_fresh_experiment_validates() -> None:
    client = _FakeExperimentClient(
        by_id={"exp-1": ExperimentInfo("exp-1", "/Users/a/agent")}, traces={"exp-1": 0}
    )
    v = validate_experiment("exp-1", client=client, claimed_experiment_ids={})
    assert v.exists is True
    assert v.fresh is True
    assert v.reasons == []


def test_non_fresh_experiment_with_traces_is_rejected() -> None:
    client = _FakeExperimentClient(
        by_id={"exp-1": ExperimentInfo("exp-1", "/Users/a/agent")}, traces={"exp-1": 7}
    )
    v = validate_experiment("exp-1", client=client, claimed_experiment_ids={})
    assert v.fresh is False
    assert v.trace_count == 7
    assert any("trace" in r for r in v.reasons)


def test_already_registered_experiment_is_rejected() -> None:
    client = _FakeExperimentClient(
        by_id={"exp-1": ExperimentInfo("exp-1", "/Users/a/agent")}, traces={"exp-1": 0}
    )
    v = validate_experiment("exp-1", client=client, claimed_experiment_ids={"exp-1": "claude_code"})
    assert v.fresh is False
    assert v.already_registered is True
    assert v.registered_as == "claude_code"


def test_missing_experiment_is_not_fresh() -> None:
    client = _FakeExperimentClient(by_id={})
    v = validate_experiment("nope", client=client, claimed_experiment_ids={})
    assert v.exists is False
    assert v.fresh is False


def test_create_experiment_returns_id() -> None:
    client = _FakeExperimentClient(create_returns="exp-42")
    creation = create_experiment("Fresh agent", client=client)
    assert isinstance(creation, ExperimentCreation)
    assert creation.experiment_id == "exp-42"
    assert client.created == ["Fresh agent"]


def test_create_experiment_refuses_existing_name() -> None:
    client = _FakeExperimentClient(by_name={"Taken": ExperimentInfo("exp-9", "Taken")})
    with pytest.raises(ValueError):
        create_experiment("Taken", client=client)
    assert client.created == []  # never attempted


def test_create_experiment_failclosed_on_empty_id() -> None:
    # A client that returns no id must never be reported as a creation.
    client = _FakeExperimentClient(create_returns="")
    with pytest.raises(ExperimentPermissionError):
        create_experiment("Fresh agent", client=client)


# ---------------------------------------------------------------------------
# register_agent: reuse publish_registry, fail-closed
# ---------------------------------------------------------------------------


def _register(client, existing=None, **overrides):  # type: ignore[no-untyped-def]
    kwargs = {
        "agent_name": "my_agent",
        "experiment_id": "exp-1",
        "goal_keys": ["accuracy", "cost"],
        "actor": "dev@databricks.com",
        "client": client,
        "warehouse_id": "wh",
        "existing_agents": existing or {},
        "catalog": "cat",
        "schema": "sch",
    }
    kwargs.update(overrides)
    return register_agent(**kwargs)


def test_register_writes_via_publish_registry() -> None:
    client = _RecordingClient()
    result = _register(client)
    assert result.outcome is OnboardingOutcome.REGISTERED
    assert result.agent_name == "my_agent"
    assert result.judge_config is not None
    # It reused the registry writer: exactly one composite agent_name REPLACE.
    swaps = [s for s in client.statement_execution.statements if "REPLACE WHERE" in s]
    assert len(swaps) == 1
    assert "agent_name = 'my_agent'" in swaps[0]


def test_register_refuses_anonymous_actor() -> None:
    result = _register(_RecordingClient(), actor="  ")
    assert result.outcome is OnboardingOutcome.REFUSED
    assert "anonymous" in (result.refused_reason or "")


def test_register_refuses_duplicate_name() -> None:
    result = _register(_RecordingClient(), existing={"my_agent": "exp-other"})
    assert result.outcome is OnboardingOutcome.REFUSED
    assert "already registered" in (result.refused_reason or "")


def test_register_refuses_claimed_experiment() -> None:
    result = _register(_RecordingClient(), existing={"other_agent": "exp-1"})
    assert result.outcome is OnboardingOutcome.REFUSED
    assert "one agent per experiment" in (result.refused_reason or "")


def test_register_refuses_unknown_goal() -> None:
    result = _register(_RecordingClient(), goal_keys=["turbo"])
    assert result.outcome is OnboardingOutcome.REFUSED


def test_register_does_not_fabricate_success_on_write_failure() -> None:
    # A warehouse write failure must propagate — never a fabricated REGISTERED.
    with pytest.raises(RuntimeError):
        _register(_RaisingClient())


# ---------------------------------------------------------------------------
# load_registered_agents: authoritative read, tolerant of a fresh table
# ---------------------------------------------------------------------------


def test_load_registered_agents_reads_rows() -> None:
    resp = _Resp(
        StatementState.SUCCEEDED,
        cols=["agent_name", "experiment_id"],
        data=[["claude_code", "660599403165942"]],
    )
    agents = load_registered_agents(
        client=_ReadClient(resp), warehouse_id="wh", catalog="cat", schema="sch"
    )
    assert agents == {"claude_code": "660599403165942"}


def test_load_registered_agents_tolerates_missing_table() -> None:
    resp = _Resp(StatementState.FAILED, err=_Err("TABLE_OR_VIEW_NOT_FOUND: agent_registry"))
    agents = load_registered_agents(
        client=_ReadClient(resp), warehouse_id="wh", catalog="cat", schema="sch"
    )
    assert agents == {}


def test_load_registered_agents_failclosed_on_other_error() -> None:
    resp = _Resp(StatementState.FAILED, err=_Err("PERMISSION_DENIED on warehouse"))
    with pytest.raises(RuntimeError):
        load_registered_agents(
            client=_ReadClient(resp), warehouse_id="wh", catalog="cat", schema="sch"
        )


# ---------------------------------------------------------------------------
# dispatch + live-wiring guards
# ---------------------------------------------------------------------------


def test_run_action_unknown_is_error() -> None:
    result = run_action({"action": "nuke", "actor": "a@b.com"})
    assert isinstance(result, ErrorResult)
    assert result.outcome is OnboardingOutcome.ERROR


def test_run_action_requirements_dispatches() -> None:
    result = run_action({"action": "requirements", "goals": ["cost"], "actor": "a@b.com"})
    assert result.outcome == "requirements"


def test_run_requirements_is_pure() -> None:
    result = run_requirements(["accuracy"])
    assert result.outcome == "requirements"


def test_run_register_refuses_anonymous_without_touching_warehouse() -> None:
    result = run_register(
        agent_name="a",
        experiment_id="e",
        goal_keys=["cost"],
        actor="",
    )
    assert isinstance(result, RegisterResult)
    assert result.outcome is OnboardingOutcome.REFUSED


def test_run_register_errors_without_warehouse(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("DATABRICKS_WAREHOUSE_ID", raising=False)
    result = run_register(
        agent_name="a",
        experiment_id="e",
        goal_keys=["cost"],
        actor="dev@databricks.com",
    )
    assert result.outcome is OnboardingOutcome.ERROR
    assert "warehouse" in (result.error or "")


def test_run_create_refuses_anonymous() -> None:
    result = run_create("Fresh", actor="")
    assert result.outcome is OnboardingOutcome.REFUSED
