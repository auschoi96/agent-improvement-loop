"""Unit tests for the post-deploy bootstrap (:mod:`ail.jobs.bootstrap_grants`).

These prove the three idempotent behaviours without any workspace access: the
``WorkspaceClient`` and ``MlflowClient`` are faked. Coverage:

* provide-or-create warehouse (use existing / reuse by name / create serverless),
* the grant uses ``update_permissions`` (merge) with ``CAN_USE``,
* the experiment monitoring tag is set via the existing helper, and
* the CLI ``main`` wiring collapses empty strings to "not provided".
"""

from __future__ import annotations

import pytest
from databricks.sdk.service.catalog import Privilege, SecurableType
from databricks.sdk.service.sql import (
    CreateWarehouseRequestWarehouseType,
    StatementState,
    WarehousePermissionLevel,
)

from ail.compare.monitoring import MONITORING_WAREHOUSE_TAG, TRACING_WAREHOUSE_ENV
from ail.jobs import bootstrap_grants
from ail.jobs.bootstrap_grants import (
    DEFAULT_AUTO_STOP_MINS,
    DEFAULT_CLUSTER_SIZE,
    DEFAULT_WAREHOUSE_NAME,
    REFERENCE_WORKSPACE_DEFAULTS,
    bootstrap,
    ensure_warehouse,
    grant_framework_schema_access,
    grant_warehouse_can_use,
    main,
    validate_workspace_values,
)

# -- fakes -----------------------------------------------------------------


class _FakeWarehouse:
    def __init__(self, id: str, name: str) -> None:
        self.id = id
        self.name = name


class _FakeCreateResponse:
    def __init__(self, id: str | None) -> None:
        self.id = id


class _FakeWaiter:
    """Mirrors the SDK ``Wait`` object: ``.response`` holds the create response."""

    def __init__(self, response: _FakeCreateResponse) -> None:
        self.response = response


class _FakeWarehousesAPI:
    def __init__(
        self, listing: list[_FakeWarehouse] | None = None, new_id: str | None = "new-wh-id"
    ) -> None:
        self._listing = listing or []
        self._new_id = new_id
        self.create_calls: list[dict[str, object]] = []
        self.update_perm_calls: list[tuple[str, list[object]]] = []
        self.listed = 0

    def list(self) -> list[_FakeWarehouse]:
        self.listed += 1
        return self._listing

    def create(self, **kwargs: object) -> _FakeWaiter:
        self.create_calls.append(kwargs)
        return _FakeWaiter(_FakeCreateResponse(self._new_id))

    def update_permissions(
        self, warehouse_id: str, *, access_control_list: list[object] | None = None
    ) -> object:
        self.update_perm_calls.append((warehouse_id, access_control_list or []))
        return object()


class _FakeStatementStatus:
    def __init__(self, state: StatementState) -> None:
        self.state = state
        self.error = None


class _FakeStatementResponse:
    def __init__(self, statement_id: str, state: StatementState) -> None:
        self.statement_id = statement_id
        self.status = _FakeStatementStatus(state)


class _FakeStatementExecutionAPI:
    """Records executed statements and always succeeds immediately."""

    def __init__(self) -> None:
        self.statements: list[str] = []

    def execute_statement(
        self, *, warehouse_id: str, statement: str, wait_timeout: str = "50s"
    ) -> _FakeStatementResponse:
        self.statements.append(statement)
        return _FakeStatementResponse("stmt-1", StatementState.SUCCEEDED)

    def get_statement(self, statement_id: str) -> _FakeStatementResponse:
        return _FakeStatementResponse(statement_id, StatementState.SUCCEEDED)


class _FakeGrantsAPI:
    def __init__(self) -> None:
        self.update_calls: list[tuple[object, str, list[object]]] = []

    def update(
        self, securable_type: object, full_name: str, *, changes: list[object]
    ) -> object:
        self.update_calls.append((securable_type, full_name, changes))
        return object()


class _FakeClient:
    def __init__(
        self,
        warehouses: _FakeWarehousesAPI,
        statement_execution: _FakeStatementExecutionAPI | None = None,
        grants: _FakeGrantsAPI | None = None,
    ) -> None:
        self.warehouses = warehouses
        self.statement_execution = statement_execution or _FakeStatementExecutionAPI()
        self.grants = grants or _FakeGrantsAPI()


class _FakeMlflowClient:
    def __init__(self) -> None:
        self.tag_calls: list[tuple[str, str, str]] = []

    def set_experiment_tag(self, experiment_id: str, key: str, value: str) -> None:
        self.tag_calls.append((experiment_id, key, value))


# -- ensure_warehouse: provide-or-create -----------------------------------


def test_provided_warehouse_id_is_used_without_listing_or_creating() -> None:
    api = _FakeWarehousesAPI(listing=[_FakeWarehouse("other", "something-else")])
    wid, created = ensure_warehouse(_FakeClient(api), warehouse_id="wh-existing")
    assert (wid, created) == ("wh-existing", False)
    assert api.listed == 0  # short-circuits — no lookup, no create
    assert api.create_calls == []


def test_existing_warehouse_found_by_name_is_reused_idempotently() -> None:
    api = _FakeWarehousesAPI(listing=[_FakeWarehouse("wh-42", DEFAULT_WAREHOUSE_NAME)])
    wid, created = ensure_warehouse(_FakeClient(api), warehouse_id=None)
    # Re-running finds the same warehouse by name -> never a second create.
    assert (wid, created) == ("wh-42", False)
    assert api.create_calls == []


def test_creates_small_serverless_warehouse_when_absent() -> None:
    api = _FakeWarehousesAPI(listing=[_FakeWarehouse("x", "unrelated")], new_id="brand-new")
    wid, created = ensure_warehouse(_FakeClient(api), warehouse_id="   ")  # blank == not provided
    assert (wid, created) == ("brand-new", True)
    assert len(api.create_calls) == 1
    kwargs = api.create_calls[0]
    assert kwargs["enable_serverless_compute"] is True
    assert kwargs["warehouse_type"] == CreateWarehouseRequestWarehouseType.PRO
    assert kwargs["cluster_size"] == DEFAULT_CLUSTER_SIZE
    assert kwargs["auto_stop_mins"] == DEFAULT_AUTO_STOP_MINS
    assert kwargs["name"] == DEFAULT_WAREHOUSE_NAME
    assert kwargs["max_num_clusters"] == 1


def test_create_without_id_raises() -> None:
    api = _FakeWarehousesAPI(listing=[], new_id=None)
    with pytest.raises(RuntimeError, match="no id"):
        ensure_warehouse(_FakeClient(api), warehouse_id=None)


# -- grant -----------------------------------------------------------------


def test_grant_uses_merge_update_with_can_use() -> None:
    api = _FakeWarehousesAPI()
    grant_warehouse_can_use(_FakeClient(api), "wh-1", "sp-app-id")
    assert len(api.update_perm_calls) == 1
    warehouse_id, acl = api.update_perm_calls[0]
    assert warehouse_id == "wh-1"
    assert len(acl) == 1
    entry = acl[0]
    assert entry.service_principal_name == "sp-app-id"
    assert entry.permission_level == WarehousePermissionLevel.CAN_USE


def test_grant_schema_access_merges_only_required_framework_privileges() -> None:
    grants = _FakeGrantsAPI()
    grant_framework_schema_access(
        _FakeClient(_FakeWarehousesAPI(), grants=grants),
        catalog="cat",
        schema="sch",
        sp_id="sp-app-id",
    )
    assert [(kind, name) for kind, name, _ in grants.update_calls] == [
        (SecurableType.CATALOG, "cat"),
        (SecurableType.SCHEMA, "cat.sch"),
    ]
    catalog_change = grants.update_calls[0][2][0]
    schema_change = grants.update_calls[1][2][0]
    assert catalog_change.principal == "sp-app-id"
    assert catalog_change.add == [Privilege.USE_CATALOG]
    assert set(schema_change.add) == {
        Privilege.USE_SCHEMA,
        Privilege.SELECT,
        Privilege.MODIFY,
        Privilege.CREATE_TABLE,
    }


# -- bootstrap orchestration ----------------------------------------------


def test_bootstrap_creates_grants_and_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TRACING_WAREHOUSE_ENV, raising=False)
    api = _FakeWarehousesAPI(listing=[], new_id="fresh-wh")
    stmts = _FakeStatementExecutionAPI()
    mlflow = _FakeMlflowClient()

    grants = _FakeGrantsAPI()
    result = bootstrap(
        experiment_id="EXP-9",
        warehouse_id="wh-explicit",
        framework_sp_id="sp-9",
        catalog="prod_catalog",
        schema="prod_schema",
        client=_FakeClient(api, statement_execution=stmts, grants=grants),
        mlflow_client=mlflow,
    )

    assert result.warehouse_id == "wh-explicit"
    assert result.warehouse_created is False
    assert result.granted_sp_id == "sp-9"
    # table-ensure ran against the resolved warehouse; every CREATE is idempotent.
    create_stmts = [s for s in stmts.statements if s.strip().upper().startswith("CREATE")]
    assert create_stmts, "bootstrap must issue the table-ensure DDL"
    assert all("IF NOT EXISTS" in s for s in create_stmts)
    # The additive-reconcile step probes live columns via information_schema; this
    # fake returns no rows, so every table looks absent -> no ALTER is emitted, and
    # bootstrap issues only CREATE + SELECT (never a destructive/ALTER statement).
    assert all(s.strip().upper().startswith(("CREATE", "SELECT")) for s in stmts.statements)
    assert result.columns_reconciled == []
    assert result.tables_ensured  # the app-read table set was covered
    # grant landed on the explicit warehouse
    assert api.update_perm_calls[0][0] == "wh-explicit"
    assert [(kind, name) for kind, name, _ in grants.update_calls] == [
        (SecurableType.CATALOG, "prod_catalog"),
        (SecurableType.SCHEMA, "prod_catalog.prod_schema"),
    ]
    # monitoring tag set on the experiment with the resolved warehouse
    assert mlflow.tag_calls == [("EXP-9", MONITORING_WAREHOUSE_TAG, "wh-explicit")]
    assert result.monitoring is not None
    assert result.monitoring.warehouse_id == "wh-explicit"
    # set_env=False: bootstrap must not mutate the process environment
    import os

    assert TRACING_WAREHOUSE_ENV not in os.environ


def test_bootstrap_reconciles_columns_after_creates(monkeypatch: pytest.MonkeyPatch) -> None:
    # The additive-reconcile step is wired into bootstrap and runs in the same
    # pre-app-build step, AFTER every CREATE: so a just-created fresh table is a
    # trivial no-op, and reconcile can never run before its table exists.
    monkeypatch.delenv(TRACING_WAREHOUSE_ENV, raising=False)
    api = _FakeWarehousesAPI(listing=[], new_id="fresh-wh")
    stmts = _FakeStatementExecutionAPI()

    bootstrap(
        experiment_id="EXP-order",
        warehouse_id="wh-order",
        framework_sp_id=None,
        catalog="prod_catalog",
        schema="prod_schema",
        client=_FakeClient(api, statement_execution=stmts),
        mlflow_client=_FakeMlflowClient(),
    )

    kinds = [s.strip().upper().split(None, 1)[0] for s in stmts.statements]
    assert "CREATE" in kinds and "SELECT" in kinds
    # Every information_schema probe (SELECT) comes after every CREATE.
    assert max(i for i, k in enumerate(kinds) if k == "CREATE") < min(
        i for i, k in enumerate(kinds) if k == "SELECT"
    )


def test_bootstrap_skips_grant_when_no_sp() -> None:
    api = _FakeWarehousesAPI(listing=[_FakeWarehouse("wh-prov", DEFAULT_WAREHOUSE_NAME)])
    mlflow = _FakeMlflowClient()

    result = bootstrap(
        experiment_id="EXP-1",
        warehouse_id="wh-given",
        framework_sp_id="   ",  # blank == not provided
        catalog="prod_catalog",
        schema="prod_schema",
        client=_FakeClient(api),
        mlflow_client=mlflow,
    )

    assert result.warehouse_id == "wh-given"
    assert result.warehouse_created is False
    assert result.granted_sp_id is None
    assert api.update_perm_calls == []  # no grant attempted
    # the tag is still set against the provided warehouse
    assert mlflow.tag_calls == [("EXP-1", MONITORING_WAREHOUSE_TAG, "wh-given")]


def test_bootstrap_create_warehouse_reaches_create_path() -> None:
    api = _FakeWarehousesAPI(listing=[], new_id="created-wh")
    mlflow = _FakeMlflowClient()

    result = bootstrap(
        experiment_id="EXP-1",
        warehouse_id=None,
        framework_sp_id=None,
        catalog="prod_catalog",
        schema="prod_schema",
        create_warehouse=True,
        client=_FakeClient(api),
        mlflow_client=mlflow,
    )

    assert result.warehouse_id == "created-wh"
    assert result.warehouse_created is True
    assert len(api.create_calls) == 1
    assert mlflow.tag_calls == [("EXP-1", MONITORING_WAREHOUSE_TAG, "created-wh")]


def test_bootstrap_rejects_blank_experiment() -> None:
    api = _FakeWarehousesAPI()
    with pytest.raises(SystemExit, match="experiment_id is empty"):
        bootstrap(
            experiment_id="  ",
            warehouse_id="wh",
            framework_sp_id=None,
            catalog="prod_catalog",
            schema="prod_schema",
            client=_FakeClient(api),
            mlflow_client=_FakeMlflowClient(),
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"experiment_id": "", "warehouse_id": "wh", "catalog": "cat", "schema": "sch"},
            "experiment_id is empty",
        ),
        (
            {
                "experiment_id": "660599403165942",
                "warehouse_id": "wh",
                "catalog": "cat",
                "schema": "sch",
            },
            "experiment_id is a reference workspace default",
        ),
        (
            {"experiment_id": "exp", "warehouse_id": "", "catalog": "cat", "schema": "sch"},
            "warehouse_id is empty",
        ),
        (
            {
                "experiment_id": "exp",
                "warehouse_id": "7d1d3dbb3ba65f2a",
                "catalog": "cat",
                "schema": "sch",
            },
            "warehouse_id is a reference workspace default",
        ),
        (
            {
                "experiment_id": "exp",
                "warehouse_id": "wh",
                "catalog": "austin_choi_omni_agent_catalog",
                "schema": "sch",
            },
            "catalog is a reference workspace default",
        ),
        (
            {
                "experiment_id": "exp",
                "warehouse_id": "wh",
                "catalog": "cat",
                "schema": "agent_improvement_loop",
            },
            "schema is a reference workspace default",
        ),
        (
            {
                "experiment_id": "exp",
                "warehouse_id": "wh",
                "catalog": "cat",
                "schema": "REPLACE_ME",
            },
            "schema is a placeholder",
        ),
        (
            {
                "experiment_id": "exp",
                "warehouse_id": "wh",
                "catalog": "${var.catalog}",
                "schema": "sch",
            },
            "catalog is an unresolved bundle reference",
        ),
    ],
)
def test_workspace_value_guard_rejects_empty_placeholder_and_reference_values(
    kwargs: dict[str, str], message: str
) -> None:
    with pytest.raises(SystemExit, match=message):
        validate_workspace_values(**kwargs)


def test_workspace_value_guard_accepts_distinct_explicit_values() -> None:
    validate_workspace_values(
        experiment_id="exp-prod-123",
        warehouse_id="wh-prod-123",
        catalog="prod_ail_catalog",
        schema="prod_ail_schema",
    )


def test_workspace_value_guard_rejects_case_variant_reference_values() -> None:
    with pytest.raises(SystemExit, match="catalog is a reference workspace default"):
        validate_workspace_values(
            experiment_id="exp-prod-123",
            warehouse_id="wh-prod-123",
            catalog="Austin_Choi_Omni_Agent_Catalog",
            schema="prod_ail_schema",
        )


def test_reference_defaults_use_default_schema_constant() -> None:
    assert REFERENCE_WORKSPACE_DEFAULTS["schema"] == frozenset({bootstrap_grants.DEFAULT_SCHEMA})


def test_create_warehouse_allows_missing_warehouse_only() -> None:
    validate_workspace_values(
        experiment_id="exp-prod-123",
        warehouse_id=None,
        catalog="prod_ail_catalog",
        schema="prod_ail_schema",
        warehouse_required=False,
    )
    with pytest.raises(SystemExit, match="warehouse_id is a placeholder"):
        validate_workspace_values(
            experiment_id="exp-prod-123",
            warehouse_id="REPLACE_ME",
            catalog="prod_ail_catalog",
            schema="prod_ail_schema",
            warehouse_required=False,
        )


def test_allow_reference_workspace_bypasses_only_reference_defaults() -> None:
    validate_workspace_values(
        experiment_id="660599403165942",
        warehouse_id="7D1D3DBB3BA65F2A",
        catalog="Austin_Choi_Omni_Agent_Catalog",
        schema="Agent_Improvement_Loop",
        allow_reference_workspace=True,
    )

    with pytest.raises(SystemExit, match="warehouse_id is empty"):
        validate_workspace_values(
            experiment_id="660599403165942",
            warehouse_id="",
            catalog="Austin_Choi_Omni_Agent_Catalog",
            schema="Agent_Improvement_Loop",
            allow_reference_workspace=True,
        )

    with pytest.raises(SystemExit, match="catalog is a placeholder"):
        validate_workspace_values(
            experiment_id="660599403165942",
            warehouse_id="7D1D3DBB3BA65F2A",
            catalog="REPLACE_ME",
            schema="Agent_Improvement_Loop",
            allow_reference_workspace=True,
        )

    with pytest.raises(SystemExit, match="schema is an unresolved bundle reference"):
        validate_workspace_values(
            experiment_id="660599403165942",
            warehouse_id="7D1D3DBB3BA65F2A",
            catalog="Austin_Choi_Omni_Agent_Catalog",
            schema="${var.schema}",
            allow_reference_workspace=True,
        )


# -- main wiring -----------------------------------------------------------


def test_main_collapses_empty_strings_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def fake_bootstrap(**kwargs: object) -> bootstrap_grants.BootstrapResult:
        calls.update(kwargs)
        return bootstrap_grants.BootstrapResult(
            warehouse_id="wh",
            warehouse_created=False,
            granted_sp_id=None,
            tables_ensured=[],
            monitoring=None,
        )

    monkeypatch.setattr(bootstrap_grants, "bootstrap", fake_bootstrap)

    rc = main(
        [
            "--experiment=EXP",
            "--warehouse-id=",
            "--framework-sp-id=",
        ]
    )

    assert rc == 0
    assert calls["experiment_id"] == "EXP"
    # Empty CLI strings collapse to None -> find-or-create / skip-grant.
    assert calls["warehouse_id"] is None
    assert calls["framework_sp_id"] is None
    assert calls["catalog"] == ""
    assert calls["schema"] == ""
    assert calls["create_warehouse"] is False
    assert calls["allow_reference_workspace"] is False


def test_main_passes_through_provided_values(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def fake_bootstrap(**kwargs: object) -> bootstrap_grants.BootstrapResult:
        calls.update(kwargs)
        return bootstrap_grants.BootstrapResult(
            warehouse_id="wh-1",
            warehouse_created=True,
            granted_sp_id="sp-1",
            tables_ensured=[],
            monitoring=None,
        )

    monkeypatch.setattr(bootstrap_grants, "bootstrap", fake_bootstrap)

    rc = main(
        [
            "--experiment=EXP2",
            "--warehouse-id=wh-1",
            "--framework-sp-id=sp-1",
            "--cluster-size=Small",
            "--auto-stop-mins=15",
            "--catalog=cat-1",
            "--schema=sch-1",
            "--allow-reference-workspace",
        ]
    )

    assert rc == 0
    assert calls["warehouse_id"] == "wh-1"
    assert calls["framework_sp_id"] == "sp-1"
    assert calls["cluster_size"] == "Small"
    assert calls["auto_stop_mins"] == 15
    assert calls["catalog"] == "cat-1"
    assert calls["schema"] == "sch-1"
    assert calls["create_warehouse"] is False
    assert calls["allow_reference_workspace"] is True


def test_main_create_warehouse_prints_resolved_id(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    api = _FakeWarehousesAPI(listing=[], new_id="printed-wh")
    mlflow = _FakeMlflowClient()

    monkeypatch.setattr(bootstrap_grants, "_default_workspace_client", lambda: _FakeClient(api))
    monkeypatch.setattr(
        bootstrap_grants,
        "configure_monitoring_warehouse",
        lambda *args, **kwargs: None,
    )

    rc = main(
        [
            "--experiment=EXP2",
            "--create-warehouse",
            "--catalog=cat-1",
            "--schema=sch-1",
        ]
    )

    assert rc == 0
    assert len(api.create_calls) == 1
    assert mlflow.tag_calls == []
    out = capsys.readouterr().out
    assert "warehouse=printed-wh (created)" in out
    assert "warehouse_id=printed-wh" in out


# -- deploy heal: reconcile the RLM trigger over the whole registry ---------


class _HealTableUpdate:
    def __init__(self, table_names: list[str]) -> None:
        self.table_names = list(table_names)


class _HealTrigger:
    def __init__(self, table_names: list[str]) -> None:
        self.table_update = _HealTableUpdate(table_names)


class _HealSettings:
    def __init__(self, table_names: list[str]) -> None:
        self.trigger = _HealTrigger(table_names)


class _HealJob:
    def __init__(self, settings: object) -> None:
        self.settings = settings


class _HealJobsAPI:
    def __init__(self, table_names: list[str]) -> None:
        self._settings = _HealSettings(table_names)
        self.update_calls: list[object] = []

    def get(self, job_id: int) -> _HealJob:
        return _HealJob(self._settings)

    def update(self, job_id: int, *, new_settings: object) -> None:
        self.update_calls.append((job_id, new_settings))


class _ClientWithJobs(_FakeClient):
    def __init__(self, warehouses: _FakeWarehousesAPI, jobs: _HealJobsAPI) -> None:
        super().__init__(warehouses)
        self.jobs = jobs


def _seed_registry(monkeypatch: pytest.MonkeyPatch, agents: list[object]) -> None:
    monkeypatch.setattr(
        "ail.publish_versions.load_registered_agents_full",
        lambda **_kwargs: agents,
    )


def test_bootstrap_heals_rlm_trigger_for_registered_agents(monkeypatch: pytest.MonkeyPatch) -> None:
    from ail.registry import Agent

    monkeypatch.delenv(TRACING_WAREHOUSE_ENV, raising=False)
    _seed_registry(
        monkeypatch,
        [
            Agent(
                agent_name="claude_code",
                experiment_id="e1",
                annotations_table="cat.mlflow_traces.claude_code_otel_annotations",
            ),
            Agent(
                agent_name="newbot",
                experiment_id="e2",
                annotations_table="cat.mlflow_traces.newbot_otel_annotations",
            ),
        ],
    )
    jobs = _HealJobsAPI(["cat.mlflow_traces.claude_code_otel_spans"])
    client = _ClientWithJobs(_FakeWarehousesAPI(listing=[]), jobs)

    result = bootstrap(
        experiment_id="EXP-heal",
        warehouse_id="wh-heal",
        framework_sp_id=None,
        catalog="cat",
        schema="sch",
        rlm_job_id=643188029858547,
        client=client,
        mlflow_client=_FakeMlflowClient(),
    )

    # The heal added ONLY the missing agent's spans table (add-only) and issued one update.
    assert result.rlm_trigger_tables_added == ["cat.mlflow_traces.newbot_otel_spans"]
    assert len(jobs.update_calls) == 1


def test_bootstrap_skips_heal_when_no_rlm_job_id(monkeypatch: pytest.MonkeyPatch) -> None:
    # No rlm_job_id => the heal is a quiet no-op (never reads the registry, never writes).
    def _boom(**_kwargs: object) -> object:
        raise AssertionError("registry must not be read when rlm_job_id is None")

    monkeypatch.setattr("ail.publish_versions.load_registered_agents_full", _boom)
    result = bootstrap(
        experiment_id="EXP-noheal",
        warehouse_id="wh-noheal",
        framework_sp_id=None,
        catalog="cat",
        schema="sch",
        client=_FakeClient(_FakeWarehousesAPI(listing=[])),
        mlflow_client=_FakeMlflowClient(),
    )
    assert result.rlm_trigger_tables_added == []


def test_bootstrap_heal_is_fail_soft(monkeypatch: pytest.MonkeyPatch) -> None:
    # A registry read failure during the heal must not fail an otherwise-successful
    # bootstrap (warehouse/tables/grants already succeeded).
    def _boom(**_kwargs: object) -> object:
        raise RuntimeError("warehouse down mid-heal")

    monkeypatch.setattr("ail.publish_versions.load_registered_agents_full", _boom)
    result = bootstrap(
        experiment_id="EXP-soft",
        warehouse_id="wh-soft",
        framework_sp_id=None,
        catalog="cat",
        schema="sch",
        rlm_job_id=1,
        client=_FakeClient(_FakeWarehousesAPI(listing=[])),
        mlflow_client=_FakeMlflowClient(),
    )
    assert result.rlm_trigger_tables_added == []
    assert result.warehouse_id == "wh-soft"  # bootstrap still succeeded
