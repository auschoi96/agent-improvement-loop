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
    bootstrap,
    ensure_warehouse,
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


class _FakeClient:
    def __init__(
        self,
        warehouses: _FakeWarehousesAPI,
        statement_execution: _FakeStatementExecutionAPI | None = None,
    ) -> None:
        self.warehouses = warehouses
        self.statement_execution = statement_execution or _FakeStatementExecutionAPI()


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


# -- bootstrap orchestration ----------------------------------------------


def test_bootstrap_creates_grants_and_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TRACING_WAREHOUSE_ENV, raising=False)
    api = _FakeWarehousesAPI(listing=[], new_id="fresh-wh")
    stmts = _FakeStatementExecutionAPI()
    mlflow = _FakeMlflowClient()

    result = bootstrap(
        experiment_id="EXP-9",
        warehouse_id="wh-explicit",
        framework_sp_id="sp-9",
        catalog="prod_catalog",
        schema="prod_schema",
        client=_FakeClient(api, statement_execution=stmts),
        mlflow_client=mlflow,
    )

    assert result.warehouse_id == "wh-explicit"
    assert result.warehouse_created is False
    assert result.granted_sp_id == "sp-9"
    # table-ensure ran against the resolved warehouse, only CREATE ... IF NOT EXISTS
    assert stmts.statements, "bootstrap must issue the table-ensure DDL"
    assert all("IF NOT EXISTS" in s for s in stmts.statements)
    assert result.tables_ensured  # the app-read table set was covered
    # grant landed on the explicit warehouse
    assert api.update_perm_calls[0][0] == "wh-explicit"
    # monitoring tag set on the experiment with the resolved warehouse
    assert mlflow.tag_calls == [("EXP-9", MONITORING_WAREHOUSE_TAG, "wh-explicit")]
    assert result.monitoring is not None
    assert result.monitoring.warehouse_id == "wh-explicit"
    # set_env=False: bootstrap must not mutate the process environment
    import os

    assert TRACING_WAREHOUSE_ENV not in os.environ


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
        ]
    )

    assert rc == 0
    assert calls["warehouse_id"] == "wh-1"
    assert calls["framework_sp_id"] == "sp-1"
    assert calls["cluster_size"] == "Small"
    assert calls["auto_stop_mins"] == 15
