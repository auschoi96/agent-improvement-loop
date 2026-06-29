"""Unit tests for the scheduled-Job entrypoint (:mod:`ail.jobs.publish_job`).

These cover the one job-specific concern — resolving a v4-acceptable bearer for
the run-as identity — plus the arg wiring into ``ail.publish.publish``. No
network/warehouse access: the WorkspaceClient and the publish call are faked.
"""

from __future__ import annotations

import pytest

from ail.jobs import publish_job
from ail.jobs.publish_job import _bearer_from_config, main, resolve_job_auth

# -- fakes -----------------------------------------------------------------


class _FakeConfig:
    def __init__(self, host: str, auth_header: str | None) -> None:
        self.host = host
        self._auth_header = auth_header

    def authenticate(self) -> dict[str, str]:
        return {} if self._auth_header is None else {"Authorization": self._auth_header}


class _FakeSecrets:
    def __init__(self, value: str) -> None:
        self._value = value
        self.calls: list[tuple[str, str]] = []

    def get(self, scope: str, key: str) -> str:
        self.calls.append((scope, key))
        return self._value


class _FakeDbutils:
    def __init__(self, secret_value: str) -> None:
        self.secrets = _FakeSecrets(secret_value)


class _FakeClient:
    def __init__(
        self, host: str = "https://ws.example.com", auth_header: str = "Bearer minted-tok-123"
    ) -> None:
        self.config = _FakeConfig(host, auth_header)
        self.dbutils = _FakeDbutils("secret-tok-456")


# -- resolve_job_auth ------------------------------------------------------


def test_preset_env_short_circuits_without_client() -> None:
    env = {"DATABRICKS_HOST": "https://h", "DATABRICKS_TOKEN": "t"}

    def _no_client() -> object:
        raise AssertionError("factory must not be called when env is pre-set")

    assert resolve_job_auth(workspace_client_factory=_no_client, env=env) == "env"
    # untouched
    assert env == {"DATABRICKS_HOST": "https://h", "DATABRICKS_TOKEN": "t"}


def test_minted_path_sets_explicit_bearer_and_drops_profile() -> None:
    env: dict[str, str] = {"DATABRICKS_CONFIG_PROFILE": "dais-demo"}
    client = _FakeClient(host="https://ws.example.com", auth_header="Bearer minted-tok-123")

    path = resolve_job_auth(workspace_client_factory=lambda: client, env=env)

    assert path == "minted"
    assert env["DATABRICKS_HOST"] == "https://ws.example.com"
    assert env["DATABRICKS_TOKEN"] == "minted-tok-123"
    # The ambient profile is dropped so MLflow uses the explicit bearer uniformly.
    assert "DATABRICKS_CONFIG_PROFILE" not in env


def test_secret_scope_path_reads_token_from_scope() -> None:
    env: dict[str, str] = {}
    client = _FakeClient()

    path = resolve_job_auth(
        token_secret_scope="ail",
        token_secret_key="sp_token",
        workspace_client_factory=lambda: client,
        env=env,
    )

    assert path == "secret-scope"
    assert env["DATABRICKS_TOKEN"] == "secret-tok-456"
    assert client.dbutils.secrets.calls == [("ail", "sp_token")]


def test_missing_host_raises() -> None:
    client = _FakeClient(host="")
    with pytest.raises(RuntimeError, match="workspace host"):
        resolve_job_auth(workspace_client_factory=lambda: client, env={})


def test_bearer_from_config_rejects_non_bearer() -> None:
    with pytest.raises(RuntimeError, match="bearer token"):
        _bearer_from_config(_FakeConfig("https://h", "Basic abc"))
    with pytest.raises(RuntimeError, match="bearer token"):
        _bearer_from_config(_FakeConfig("https://h", None))


def test_bearer_from_config_extracts_token() -> None:
    assert _bearer_from_config(_FakeConfig("https://h", "Bearer tok-xyz")) == "tok-xyz"


# -- main wiring -----------------------------------------------------------


def test_main_resolves_auth_then_calls_publish(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def fake_resolve(**kwargs: object) -> str:
        calls["resolve"] = kwargs
        return "minted"

    def fake_publish(**kwargs: object) -> object:
        calls["publish"] = kwargs
        return object()

    monkeypatch.setattr(publish_job, "resolve_job_auth", fake_resolve)
    monkeypatch.setattr(publish_job, "publish", fake_publish)

    rc = main(
        [
            "--experiment=EXP1",
            "--warehouse-id=wh-1",
            "--catalog=cat",
            "--schema=sch",
            "--max-results=5",
            # bundle passes empty strings when no secret scope is configured
            "--token-secret-scope=",
            "--token-secret-key=",
        ]
    )

    assert rc == 0
    # Empty secret args collapse to None -> mint path.
    assert calls["resolve"] == {"token_secret_scope": None, "token_secret_key": None}
    assert calls["publish"] == {
        "experiment_id": "EXP1",
        "warehouse_id": "wh-1",
        "catalog": "cat",
        "schema": "sch",
        "max_results": 5,
    }


def test_main_requires_warehouse_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AIL_WAREHOUSE_ID", raising=False)
    with pytest.raises(SystemExit):
        main(["--experiment=EXP1"])
