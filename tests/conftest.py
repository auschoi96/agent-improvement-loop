"""Shared test fixtures."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# A fake WorkspaceClient whose statement_execution answers SELECTs from a
# responder and records every executed statement — enough for the shared
# ``_execute`` / ``_read_rows`` seams (SUCCEEDED immediately, never polled).
# ---------------------------------------------------------------------------


def _fake_statement_response(columns: list[str] | None, data: list[list[Any]] | None) -> Any:
    from databricks.sdk.service.sql import StatementState

    status = SimpleNamespace(state=StatementState.SUCCEEDED, error=None)
    manifest = None
    result = None
    if columns is not None:
        manifest = SimpleNamespace(
            schema=SimpleNamespace(columns=[SimpleNamespace(name=c) for c in columns])
        )
        result = SimpleNamespace(data_array=data or [])
    return SimpleNamespace(statement_id="stmt-1", status=status, manifest=manifest, result=result)


class _FakeStatementExecution:
    def __init__(
        self, responder: Callable[[str], tuple[list[str], list[list[Any]]] | None]
    ) -> None:
        self.executed: list[str] = []
        self._responder = responder

    def execute_statement(
        self, *, warehouse_id: str, statement: str, wait_timeout: str = "50s"
    ) -> Any:
        self.executed.append(statement)
        return _fake_statement_response(*(self._responder(statement) or (None, None)))

    def get_statement(self, statement_id: str) -> Any:  # pragma: no cover - never polled
        raise AssertionError("fake returns SUCCEEDED immediately; get_statement must not be called")


class _FakeSqlClient:
    def __init__(
        self, responder: Callable[[str], tuple[list[str], list[list[Any]]] | None]
    ) -> None:
        self.statement_execution = _FakeStatementExecution(responder)


def _make_fake_sql_client(
    matchers: dict[str, tuple[list[str], list[list[Any]]]]
    | Callable[[str], tuple[list[str], list[list[Any]]] | None],
) -> _FakeSqlClient:
    """Build a fake SQL client from a substring->(columns, rows) map or a responder.

    A statement matching no key (e.g. an INSERT) gets a SUCCEEDED response with no
    result set, so ``_execute`` succeeds and ``_read_rows`` returns ``[]``.
    """
    if callable(matchers):
        return _FakeSqlClient(matchers)

    def responder(statement: str) -> tuple[list[str], list[list[Any]]] | None:
        for substr, cols_rows in matchers.items():
            if substr in statement:
                return cols_rows
        return None

    return _FakeSqlClient(responder)


@pytest.fixture
def fake_sql_client() -> Callable[..., _FakeSqlClient]:
    """Factory: ``fake_sql_client({"SELECT ...": (cols, rows)})`` or a responder callable."""
    return _make_fake_sql_client


@pytest.fixture
def synthetic_trace() -> Any:
    """A synthetic MLflow ``Trace`` reconstructed from a recorded schema.

    The JSON matches the real ``Trace.to_dict()`` shape observed against
    experiment 660599403165942 (AGENT + LLM + two TOOL spans, one of which
    errored), but carries no real session content. Reconstructing via
    ``Trace.from_dict`` exercises the exact normalization path used for live
    traces.
    """
    from mlflow.entities import Trace

    data = json.loads((FIXTURE_DIR / "synthetic_trace.json").read_text())
    return Trace.from_dict(data)
