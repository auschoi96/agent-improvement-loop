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

import json
from typing import Any

import pytest
from databricks.sdk.service.sql import StatementState

from ail.onboarding.experiment import (
    ExperimentAccessError,
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
    RequirementsConfirmResult,
    RequirementsPreviewResult,
    compiled_goal_to_goal_config,
    load_registered_agents,
    register_agent,
    run_action,
    run_confirm_requirements,
    run_create,
    run_preview_requirements,
    run_register,
    run_requirements,
    run_validate,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


# A fixed workspace home the fake returns so tests exercise the bare-name → absolute
# path build without a live SDK call (the live impl resolves it via WorkspaceClient).
_FAKE_HOME = "/Users/dev@example.com"
# An arbitrary host a TS/Python constant could not plausibly be (proves the URL is
# resolved LIVE and relayed, not fabricated).
_FAKE_HOST = "https://arbitrary-workspace-9f3.cloud.databricks.example"


class _FakeExperimentClient:
    """A canned :class:`~ail.onboarding.experiment.ExperimentClient`.

    ``by_id`` / ``by_name`` are the visible experiments; ``traces`` maps an
    experiment id to its trace count; ``created`` records create calls. ``home`` is
    the workspace home the bare-name path build uses (``None`` exercises the
    fail-closed path); ``host`` feeds the convenience URL (``""`` exercises the
    fail-soft path). Any method can be told to ``raise`` to exercise fail-closed.
    """

    def __init__(
        self,
        *,
        by_id: dict[str, ExperimentInfo] | None = None,
        by_name: dict[str, ExperimentInfo] | None = None,
        traces: dict[str, int] | None = None,
        create_returns: str | None = "exp-new",
        home: str | None = _FAKE_HOME,
        host: str = _FAKE_HOST,
    ) -> None:
        self.by_id = by_id or {}
        self.by_name = by_name or {}
        self.traces = traces or {}
        self.create_returns = create_returns
        self.home = home
        self.host = host
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

    def workspace_home(self):  # type: ignore[no-untyped-def]
        return self.home

    def workspace_host(self):  # type: ignore[no-untyped-def]
        return self.host


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


def test_create_experiment_bare_name_builds_absolute_path() -> None:
    # (a) A BARE name is placed under the caller's resolved workspace home — the
    # created experiment carries the ABSOLUTE path, and that is what MLflow was asked
    # to create (Databricks-backed MLflow rejects a bare name).
    client = _FakeExperimentClient(create_returns="exp-42")
    creation = create_experiment("my-agent-exp", client=client)
    assert isinstance(creation, ExperimentCreation)
    assert creation.experiment_id == "exp-42"
    assert creation.name == f"{_FAKE_HOME}/my-agent-exp"
    assert client.created == [f"{_FAKE_HOME}/my-agent-exp"]


def test_create_experiment_absolute_name_used_as_is() -> None:
    # (b) An already-absolute name is used verbatim (back-compat: the wizard may pass
    # one) — the home is NOT consulted, nothing is prefixed.
    client = _FakeExperimentClient(create_returns="exp-7", home=None)
    creation = create_experiment("/Users/someone/explicit-path", client=client)
    assert creation.experiment_id == "exp-7"
    assert creation.name == "/Users/someone/explicit-path"
    assert client.created == ["/Users/someone/explicit-path"]


def test_create_experiment_bare_name_failclosed_when_home_unresolvable() -> None:
    # (c) A BARE name whose workspace home cannot be resolved is fail-closed: an honest
    # ExperimentAccessError, and NOTHING is created at a guessed path.
    client = _FakeExperimentClient(create_returns="exp-x", home=None)
    with pytest.raises(ExperimentAccessError):
        create_experiment("my-agent-exp", client=client)
    assert client.created == []  # never attempted


@pytest.mark.parametrize("slashes_only", ["/", "//", "///"])
def test_create_experiment_failclosed_on_slashes_only_name(slashes_only: str) -> None:
    # A slashes-only name starts with '/' but has NO leaf segment — it is not a valid
    # absolute workspace path. Fail closed with an honest ExperimentAccessError BEFORE
    # any create; nothing is created at an invalid path.
    client = _FakeExperimentClient(create_returns="exp-x")
    with pytest.raises(ExperimentAccessError):
        create_experiment(slashes_only, client=client)
    assert client.created == []  # never attempted


def test_create_experiment_refuses_existing_name() -> None:
    # (f) The 'already exists' refusal is preserved, now against the FINAL absolute
    # name — never silently reuse another agent's experiment.
    target = f"{_FAKE_HOME}/Taken"
    client = _FakeExperimentClient(by_name={target: ExperimentInfo("exp-9", target)})
    with pytest.raises(ValueError):
        create_experiment("Taken", client=client)
    assert client.created == []  # never attempted


def test_create_experiment_failclosed_on_empty_id() -> None:
    # A client that returns no id must never be reported as a creation.
    client = _FakeExperimentClient(create_returns="")
    with pytest.raises(ExperimentPermissionError):
        create_experiment("my-agent-exp", client=client)


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
# register_agent: the EXTENDED registry fields (goal_config / annotations_table /
# target_workspace) — what makes a wizard-registered agent fully functional across
# the loop (Slice 4). Capture the Agent handed to publish_registry (no live write).
# ---------------------------------------------------------------------------


class _CapturePublishRegistry:
    """Capture the ``AgentRegistry`` publish_registry was called with (no write)."""

    def __init__(self) -> None:
        self.registries: list[Any] = []

    def __call__(self, registry: Any, **_kwargs: Any) -> None:
        self.registries.append(registry)


_SLICE4_GOAL_CONFIG = {
    "objective_metric": "duration_seconds",
    "goal_direction": "minimize",
    "goal_target": -0.30,
    "goal_target_kind": "relative",
    "guardrail_judge": ["correctness:4.0"],
}


def test_register_persists_all_three_extended_fields(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    cap = _CapturePublishRegistry()
    monkeypatch.setattr("ail.onboarding.service.publish_registry", cap)
    result = _register(
        _RecordingClient(),
        goal_config=_SLICE4_GOAL_CONFIG,
        annotations_table="cat.sch.otel_annotations",
        target_workspace="/Workspace/Repos/me/my_agent",
    )
    assert result.outcome is OnboardingOutcome.REGISTERED
    assert len(cap.registries) == 1
    agent = cap.registries[0].agents[0]
    # ALL THREE new carriers land on the persisted Agent...
    assert agent.goal_config == _SLICE4_GOAL_CONFIG
    assert agent.annotations_table == "cat.sch.otel_annotations"
    assert agent.target_workspace == "/Workspace/Repos/me/my_agent"
    # ...alongside the fields register always carried (no regression).
    assert agent.agent_name == "my_agent"
    assert agent.experiment_id == "exp-1"
    assert agent.judge_config is not None


def test_register_backcompat_extended_fields_default_to_none(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # Called exactly as today (none of the three supplied) => all three persist as
    # None — a registered-but-not-fully-functional agent, honest, no regression.
    cap = _CapturePublishRegistry()
    monkeypatch.setattr("ail.onboarding.service.publish_registry", cap)
    result = _register(_RecordingClient())
    assert result.outcome is OnboardingOutcome.REGISTERED
    agent = cap.registries[0].agents[0]
    assert agent.goal_config is None
    assert agent.annotations_table is None
    assert agent.target_workspace is None


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


def test_run_create_populates_url_and_tracing_hint_on_created() -> None:
    # (d) A real creation hands the experiment back ready to use: the deep-link URL
    # (host resolved LIVE via the seam) and a copy-paste tracing snippet built from id.
    client = _FakeExperimentClient(create_returns="exp-42")
    result = run_create("my-agent-exp", actor="dev@databricks.com", experiment_client=client)
    assert result.outcome is OnboardingOutcome.CREATED
    assert result.experiment_id == "exp-42"
    assert result.name == f"{_FAKE_HOME}/my-agent-exp"
    assert result.experiment_url == f"{_FAKE_HOST}/ml/experiments/exp-42"
    assert "exp-42" in result.tracing_hint
    assert "set_experiment" in result.tracing_hint


def test_run_create_still_created_when_url_host_unresolvable() -> None:
    # (e) The URL is a convenience: an unresolvable host is fail-soft (url='') — the
    # creation still SUCCEEDS (never fail a real create just because the URL can't build).
    client = _FakeExperimentClient(create_returns="exp-42", host="")
    result = run_create("my-agent-exp", actor="dev@databricks.com", experiment_client=client)
    assert result.outcome is OnboardingOutcome.CREATED
    assert result.experiment_id == "exp-42"
    assert result.experiment_url == ""
    assert "exp-42" in result.tracing_hint  # the hint needs no host, still present


# ---------------------------------------------------------------------------
# register_agent dispatch: the extended fields are parsed fail-closed at the payload
# boundary (a bad type is a REFUSED, nothing written — never a crash or silent drop).
# ---------------------------------------------------------------------------


def _register_action(**fields: Any) -> Any:
    payload = {
        "action": "register_agent",
        "actor": "dev@databricks.com",
        "agent_name": "x",
        "experiment_id": "e",
        "goals": ["cost"],
    }
    payload.update(fields)
    return run_action(payload)


@pytest.mark.parametrize(
    "bad_goal_config",
    [
        "not-json-{[",  # a string that does not parse as JSON
        "[1, 2, 3]",  # parses, but a JSON array is not a mapping
        "42",  # parses, but a scalar is not a mapping
        [1, 2, 3],  # a native list is not a mapping
        42,  # a native scalar is not a mapping
        True,  # a bool is not a mapping (and never a stray 1.0-style coercion)
    ],
)
def test_register_dispatch_refuses_non_dict_goal_config(bad_goal_config: Any) -> None:
    result = _register_action(goal_config=bad_goal_config)
    assert isinstance(result, RegisterResult)
    assert result.outcome is OnboardingOutcome.REFUSED
    assert "goal_config" in (result.refused_reason or "")


@pytest.mark.parametrize("field", ["annotations_table", "target_workspace"])
def test_register_dispatch_refuses_non_string_field(field: str) -> None:
    result = _register_action(**{field: {"not": "a string"}})
    assert isinstance(result, RegisterResult)
    assert result.outcome is OnboardingOutcome.REFUSED
    assert field in (result.refused_reason or "")


def test_register_dispatch_accepts_json_string_goal_config(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # A well-formed goal_config (as a JSON-object string) + string annotations_table /
    # target_workspace pass type validation and reach run_register — proven by the
    # honest no-warehouse ERROR (NOT a type REFUSED), with no live write attempted.
    monkeypatch.delenv("DATABRICKS_WAREHOUSE_ID", raising=False)
    result = _register_action(
        goal_config='{"objective_metric": "duration_seconds", "goal_direction": "minimize"}',
        annotations_table="cat.sch.otel_annotations",
        target_workspace="/Workspace/Repos/me/my_agent",
    )
    assert isinstance(result, RegisterResult)
    assert result.outcome is OnboardingOutcome.ERROR
    assert "warehouse" in (result.error or "")


# ---------------------------------------------------------------------------
# run_validate — freshness is FULLY fail-closed: BOTH checks must run (BLOCKING 1)
# ---------------------------------------------------------------------------


def test_validate_fresh_only_when_both_checks_run() -> None:
    # Happy path: the MLflow-traces check (fake: exists + 0 traces) AND the
    # registry-claims check (explicit empty dict = it ran, found none) both ran.
    client = _FakeExperimentClient(
        by_id={"exp-1": ExperimentInfo("exp-1", "/Users/a/agent")}, traces={"exp-1": 0}
    )
    result = run_validate(
        "exp-1", actor="dev@databricks.com", experiment_client=client, claimed_experiment_ids={}
    )
    assert result.outcome is OnboardingOutcome.VALIDATED
    assert result.fresh is True


def test_validate_failclosed_when_claims_check_cannot_run(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # No warehouse configured => the registry-claims check cannot run. Even though
    # the experiment exists with zero traces, we must NOT report fresh (that would
    # fabricate an unverified 'not claimed' state). Honest ERROR naming the prereq.
    monkeypatch.delenv("DATABRICKS_WAREHOUSE_ID", raising=False)
    client = _FakeExperimentClient(
        by_id={"exp-1": ExperimentInfo("exp-1", "/Users/a/agent")}, traces={"exp-1": 0}
    )
    result = run_validate(
        "exp-1", actor="dev@databricks.com", experiment_client=client, claimed_experiment_ids=None
    )
    assert result.outcome is OnboardingOutcome.ERROR
    assert result.fresh is False
    assert "unclaimed" in (result.error or "") and "DATABRICKS_WAREHOUSE_ID" in (result.error or "")


def test_validate_failclosed_when_registry_read_raises(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # A warehouse IS configured but the registry read fails (permission/authority) —
    # the claims check raised, so we fail closed with an honest error, never fresh.
    monkeypatch.setenv("DATABRICKS_WAREHOUSE_ID", "wh")

    def _boom(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("PERMISSION_DENIED reading agent_registry")

    monkeypatch.setattr("ail.onboarding.service._claimed_experiments", _boom)
    client = _FakeExperimentClient(
        by_id={"exp-1": ExperimentInfo("exp-1", "/Users/a/agent")}, traces={"exp-1": 0}
    )
    result = run_validate(
        "exp-1", actor="dev@databricks.com", experiment_client=client, claimed_experiment_ids=None
    )
    assert result.outcome is OnboardingOutcome.ERROR
    assert result.fresh is False
    assert "PERMISSION_DENIED" in (result.error or "")


# ---------------------------------------------------------------------------
# requirements — gate descriptions come from Python with the real floors (BLOCKING 2)
# ---------------------------------------------------------------------------


def test_python_owns_the_gate_descriptions_with_real_floors() -> None:
    # The per-goal + overall summaries are composed in Python from the computed gate
    # set + ReadinessThresholds, so the client renders them verbatim (no TS numbers).
    judged = build_requirements(["accuracy"])
    (acc,) = judged.selected
    assert "20" in acc.summary  # quality_min_labels, from ReadinessThresholds
    assert "50" in acc.summary  # prove_min_traces
    assert "20" in judged.summary  # overall note names the label floor
    assert judged.summary  # non-empty when goals selected

    deterministic = build_requirements(["cost"])
    (cost,) = deterministic.selected
    assert "50" in cost.summary  # needs traces to prove
    assert "20" not in cost.summary  # a deterministic goal never needs the 20 labels
    assert "no human labels" in deterministic.summary

    # No goals selected => no fabricated summary.
    assert build_requirements(None).summary == ""


# ---------------------------------------------------------------------------
# preview_requirements / confirm_requirements — the free-form intake actions.
# All offline: the LLM is a canned mock (the compile_goal seam), and judge
# authoring / goal persistence are injected spies — no live model / MLflow /
# warehouse call. The headline invariants: a PREVIEW authors + persists NOTHING,
# and a CONFIRM needs an explicit human target and only then authors + persists.
# ---------------------------------------------------------------------------


def _mock_llm(payload: Any):  # type: ignore[no-untyped-def]
    """A canned extractor LLM (the ail.goals.compiler.GoalProposerLLM seam)."""
    text = json.dumps(payload)

    def _llm(*, system: str, user: str) -> str:
        return text

    return _llm


# priority 1 = a quality dimension (judge, maximize); priority 3 = a deterministic
# L0 metric (latency, no judge). Deterministic given the same text (temp-0 LLM).
_THREE_DIMS = [
    {
        "name": "no hallucinated tool calls",
        "description": "never invent tool calls the user did not enable",
        "user_priority": 1,
        "metric": None,
    },
    {
        "name": "response conciseness",
        "description": "answers should be brief",
        "user_priority": 2,
        "metric": None,
    },
    {
        "name": "latency",
        "description": "responses should be fast",
        "user_priority": 3,
        "metric": "duration_seconds",
    },
]


def _spy_author(log: list[tuple[str, str]]):  # type: ignore[no-untyped-def]
    def _author(name: str, description: str, *, experiment_id: str) -> Any:
        log.append(("author", name))
        return object()

    return _author


def test_preview_returns_structured_plan_and_authors_persists_nothing(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # execute_plan is the ONLY author/persist path; a preview must never reach it.
    def _boom(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("preview must not author or persist (execute_plan called)")

    monkeypatch.setattr("ail.onboarding.service.execute_plan", _boom)

    result = run_preview_requirements(
        "correctness matters most; never hallucinate a tool call; keep latency low",
        cohort="claude_code",
        actor="dev@databricks.com",
        llm=_mock_llm(_THREE_DIMS),
    )
    assert isinstance(result, RequirementsPreviewResult)
    assert result.outcome is OnboardingOutcome.REQUIREMENTS_PREVIEW

    # Machine fields per dimension — routing owned by Python, surfaced structurally.
    by_name = {d.name: d for d in result.dimensions}
    assert set(by_name) == {"no hallucinated tool calls", "response conciseness", "latency"}
    obj = by_name["no hallucinated tool calls"]
    assert obj.role == "objective" and obj.kind == "memalign_judge"
    assert obj.judge_name == "no_hallucinated_tool_calls" and obj.direction == "maximize"
    lat = by_name["latency"]
    assert lat.role == "guardrail" and lat.kind == "deterministic_l0"
    assert lat.metric == "duration_seconds" and lat.direction == "minimize"

    # The objective + the routed split are surfaced; and the target is a SUGGESTION.
    assert result.objective_metric == "no_hallucinated_tool_calls"
    assert sorted(result.judges_to_author) == [
        "no_hallucinated_tool_calls",
        "response_conciseness",
    ]
    assert result.deterministic_metrics == ["duration_seconds"]
    assert result.suggested_target is not None
    assert result.suggested_target.is_suggestion is True
    assert result.suggested_target.value == 0.10  # maximize default, from the composer
    assert "confirmed=False" in result.describe  # a proposal, not confirmed


def test_preview_failclosed_on_garbage_never_fabricates() -> None:
    # An LLM that ignores the JSON-array contract fails closed as an honest error.
    result = run_preview_requirements(
        "make it good", cohort="claude_code", actor="dev@databricks.com", llm=_mock_llm("nope")
    )
    assert isinstance(result, ErrorResult)
    assert result.action == "preview_requirements"


def test_confirm_requires_explicit_target_and_touches_nothing() -> None:
    log: list[tuple[str, str]] = []
    persisted: list[Any] = []
    result = run_confirm_requirements(
        "never hallucinate a tool call",
        objective_target=None,  # the human has not set/acknowledged a target
        experiment_id="exp-1",
        agent_name="claude_code",
        actor="dev@databricks.com",
        llm=_mock_llm(_THREE_DIMS),
        author=_spy_author(log),
        persist=lambda g: persisted.append(g),
    )
    assert isinstance(result, RequirementsConfirmResult)
    assert result.outcome is OnboardingOutcome.REFUSED
    assert "target" in (result.refused_reason or "")
    assert log == [] and persisted == []  # nothing authored / persisted


def test_confirm_refuses_anonymous_actor() -> None:
    result = run_confirm_requirements(
        "never hallucinate a tool call",
        objective_target=0.25,
        experiment_id="exp-1",
        agent_name="claude_code",
        actor="   ",
        llm=_mock_llm(_THREE_DIMS),
    )
    assert result.outcome is OnboardingOutcome.REFUSED
    assert "anonymous" in (result.refused_reason or "")


def test_confirm_authors_then_persists_with_human_target() -> None:
    order: list[str] = []
    log: list[tuple[str, str]] = []

    def _persist(goal: Any) -> None:
        order.append("persist")

    def _author(name: str, description: str, *, experiment_id: str) -> Any:
        assert experiment_id == "exp-1"
        order.append("author")
        log.append(("author", name))
        return object()

    result = run_confirm_requirements(
        "correctness matters most; never hallucinate a tool call; keep latency low",
        objective_target=0.25,  # a maximize objective => positive relative target
        experiment_id="exp-1",
        agent_name="claude_code",
        actor="dev@databricks.com",
        llm=_mock_llm(_THREE_DIMS),
        author=_author,
        persist=_persist,
    )
    assert result.outcome is OnboardingOutcome.REQUIREMENTS_CONFIRMED
    assert result.objective_metric == "no_hallucinated_tool_calls"
    assert result.objective_target == 0.25
    # both quality dimensions authored (in plan order), then the goal persisted.
    assert result.authored_judges == ["no_hallucinated_tool_calls", "response_conciseness"]
    assert result.persisted is True
    assert order == ["author", "author", "persist"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_confirm_rejects_non_finite_target_failclosed(bad: float) -> None:
    # A non-finite target must fail closed: the CompiledGoal sign checks (value < 0 /
    # value > 0) are BOTH False for NaN, so without this guard a NaN would slip past
    # plan.confirm()/execute_plan() and author judges + persist a meaningless goal.
    log: list[tuple[str, str]] = []
    persisted: list[Any] = []
    result = run_confirm_requirements(
        "never hallucinate a tool call",
        objective_target=bad,
        experiment_id="exp-1",
        agent_name="claude_code",
        actor="dev@databricks.com",
        llm=_mock_llm(_THREE_DIMS),
        author=_spy_author(log),
        persist=lambda g: persisted.append(g),
    )
    assert result.outcome is OnboardingOutcome.REFUSED
    assert "finite" in (result.refused_reason or "")
    assert log == [] and persisted == []  # nothing authored / persisted
    # the non-finite value is NOT echoed back, and the result stays valid JSON.
    assert result.objective_target is None
    json.loads(result.model_dump_json())  # must not raise (no NaN/Infinity token)


def test_run_action_confirm_drops_non_finite_target() -> None:
    # Defense-in-depth at the dispatcher: json.loads admits Infinity, so run_action's
    # coercion must drop a non-finite numeric to None => an honest refusal, no crash.
    result = run_action(
        {
            "action": "confirm_requirements",
            "requirements_text": "be fast",
            "agent_name": "claude_code",
            "experiment_id": "exp-1",
            "objective_target": float("inf"),
            "actor": "a@b.com",
        }
    )
    assert result.outcome is OnboardingOutcome.REFUSED


def test_confirm_wrong_sign_target_fails_closed_as_error() -> None:
    # The objective is an authored judge (maximize); a negative relative target
    # contradicts the derived direction and must be an honest error, not a bad goal.
    log: list[tuple[str, str]] = []
    persisted: list[Any] = []
    result = run_confirm_requirements(
        "never hallucinate a tool call",
        objective_target=-0.5,
        experiment_id="exp-1",
        agent_name="claude_code",
        actor="dev@databricks.com",
        llm=_mock_llm(_THREE_DIMS),
        author=_spy_author(log),
        persist=lambda g: persisted.append(g),
    )
    assert result.outcome is OnboardingOutcome.ERROR
    assert result.error  # the engine's own contract message
    assert log == [] and persisted == []


def test_run_action_dispatches_preview_and_confirm() -> None:
    # preview_requirements dispatches (an unconfigured LLM endpoint => honest error,
    # never a crash or a fabricated plan).
    preview = run_action(
        {"action": "preview_requirements", "requirements_text": "be fast", "actor": "a@b.com"}
    )
    assert preview.outcome in (
        OnboardingOutcome.REQUIREMENTS_PREVIEW,
        OnboardingOutcome.ERROR,
    )
    # confirm_requirements with no target => refused before any live seam is touched.
    confirm = run_action(
        {
            "action": "confirm_requirements",
            "requirements_text": "be fast",
            "agent_name": "claude_code",
            "experiment_id": "exp-1",
            "actor": "a@b.com",
        }
    )
    assert confirm.outcome is OnboardingOutcome.REFUSED


# ---------------------------------------------------------------------------
# goal_config wiring: confirm -> register -> RLM actually STEERS (Slice 4). The
# helper serializes a CompiledGoal to the exact keys the continuous-RLM lane reads,
# and the confirmed-requirements result surfaces that goal_config for the wizard to
# thread onto the register payload.
# ---------------------------------------------------------------------------


def test_compiled_goal_to_goal_config_round_trips_through_rlm_knobs() -> None:
    from ail.goals.compiler import CompiledGoal, GoalTarget, Guardrail
    from ail.jobs.continuous_rlm import _knobs_from_goal_config

    goal = CompiledGoal(
        objective_metric="duration_seconds",  # a generic L0 objective (not judged)
        direction="minimize",
        target=GoalTarget(value=-0.30, kind="relative"),
        guardrails=(Guardrail(name="correctness", kind="judge", threshold=4.0),),
        cohort="my_agent",
    )
    gc = compiled_goal_to_goal_config(goal)
    # JSON-clean: the registry stores it as goal_config_json (json.dumps must not raise).
    assert json.loads(json.dumps(gc)) == gc
    # Read back by the RLM lane => the SAME goal knobs (proves the steering wires up).
    knobs = _knobs_from_goal_config(gc, cohort="my_agent")
    assert knobs["objective_metric"] == "duration_seconds"
    assert knobs["goal_direction"] == "minimize"
    assert knobs["goal_target"] == -0.30
    assert knobs["goal_target_kind"] == "relative"
    assert knobs["guardrail_specs"] == ["correctness:4.0"]
    # And the judge-guardrail spec decodes back to its name + threshold, exactly as
    # ail.jobs.continuous_rlm._build_rubric reconstructs it.
    name, _, threshold = knobs["guardrail_specs"][0].partition(":")
    assert name == "correctness" and float(threshold) == 4.0


def test_confirm_surfaces_goal_config_that_steers_the_rlm() -> None:
    from ail.jobs.continuous_rlm import _knobs_from_goal_config

    result = run_confirm_requirements(
        "correctness matters most; never hallucinate a tool call; keep latency low",
        objective_target=0.25,  # a maximize objective => positive relative target
        experiment_id="exp-1",
        agent_name="claude_code",
        actor="dev@databricks.com",
        llm=_mock_llm(_THREE_DIMS),
        author=_spy_author([]),
        persist=lambda _g: None,
    )
    assert result.outcome is OnboardingOutcome.REQUIREMENTS_CONFIRMED
    # The confirmed goal is surfaced in the registry goal_config shape...
    assert result.goal_config is not None
    assert result.goal_config["objective_metric"] == "no_hallucinated_tool_calls"
    assert result.goal_config["goal_direction"] == "maximize"
    assert result.goal_config["goal_target"] == 0.25
    # ...and the RLM lane reads it back to the same objective (confirm -> register -> RLM).
    knobs = _knobs_from_goal_config(result.goal_config, cohort="claude_code")
    assert knobs["objective_metric"] == "no_hallucinated_tool_calls"
    assert knobs["goal_direction"] == "maximize"
    assert knobs["goal_target"] == 0.25
