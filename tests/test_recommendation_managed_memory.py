from __future__ import annotations

from typing import Any

import pytest

from ail.recommendations.managed_memory import (
    MEMORY_ROOT,
    ManagedMemoryClient,
    agent_memory_scope,
    cohort_memory_path,
    resolve_memory_store_name,
)


class ApiError(RuntimeError):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


class FakeApi:
    def __init__(self, outcomes: list[Any] | None = None):
        self.outcomes = list(outcomes or [])
        self.calls: list[dict[str, Any]] = []

    def do(
        self,
        method: str,
        path: str | None = None,
        *,
        query: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any] | list[Any]:
        self.calls.append({"method": method, "path": path, "query": query, "body": body})
        outcome = self.outcomes.pop(0) if self.outcomes else {}
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_scope_is_stable_agent_isolated_and_not_model_selected() -> None:
    first = agent_memory_scope("claude_code", "experiment-1")
    assert first == agent_memory_scope("claude_code", "experiment-1")
    assert first != agent_memory_scope("claude_code", "experiment-2")
    assert first.startswith("ail-agent-")
    assert "claude_code" not in first


def test_resolve_store_accepts_short_or_fully_qualified_name() -> None:
    assert resolve_memory_store_name("history", catalog="cat", schema="sch") == "cat.sch.history"
    assert (
        resolve_memory_store_name("other.team.history", catalog="cat", schema="sch")
        == "other.team.history"
    )
    with pytest.raises(ValueError, match="short name"):
        resolve_memory_store_name("bad.name", catalog="cat", schema="sch")


def test_ensure_store_is_idempotent_and_creates_only_on_not_found() -> None:
    existing = FakeApi([{"full_name": "cat.sch.history"}])
    ManagedMemoryClient(existing, "cat.sch.history").ensure_store(description="AIL memory")
    assert [call["method"] for call in existing.calls] == ["GET"]

    missing = FakeApi([ApiError("missing", 404), {"full_name": "cat.sch.history"}])
    ManagedMemoryClient(missing, "cat.sch.history").ensure_store(description="AIL memory")
    assert [call["method"] for call in missing.calls] == ["GET", "POST"]
    assert missing.calls[1]["body"] == {
        "name": "history",
        "catalog_name": "cat",
        "schema_name": "sch",
        "description": "AIL memory",
    }


def test_list_entries_uses_fixed_scope_prefix_paginates_and_parses_metadata() -> None:
    api = FakeApi(
        [
            {
                "entries": [
                    {
                        "path": f"{MEMORY_ROOT}/cohorts/c1.md",
                        "description": "cohort one",
                        "scope": "scope-1",
                        "update_time": "2026-07-21T00:00:00Z",
                    }
                ],
                "next_page_token": "next-1",
            },
            {
                "entries": [
                    {
                        "path": f"{MEMORY_ROOT}/cohorts/c2.md",
                        "description": "cohort two",
                        "scope": "scope-1",
                    }
                ]
            },
        ]
    )
    found = ManagedMemoryClient(api, "cat.sch.history").list_entries(scope="scope-1", page_size=7)
    assert [entry.description for entry in found] == ["cohort one", "cohort two"]
    assert found[0].contents == ""
    assert api.calls[0]["method"] == "GET"
    assert api.calls[0]["query"] == {
        "scope": "scope-1",
        "path_prefix": MEMORY_ROOT,
        "page_size": 7,
    }
    assert api.calls[1]["query"]["page_token"] == "next-1"


def test_upsert_creates_then_replaces_an_existing_entry() -> None:
    created = FakeApi([{}])
    ManagedMemoryClient(created, "cat.sch.history").upsert_entry(
        scope="scope-1",
        path=f"{MEMORY_ROOT}/state.md",
        contents="v1",
        description="state",
    )
    assert [call["method"] for call in created.calls] == ["POST"]
    assert created.calls[0]["query"] == {"scope": "scope-1"}

    existing = FakeApi([ApiError("exists", 409), {}])
    ManagedMemoryClient(existing, "cat.sch.history").upsert_entry(
        scope="scope-1",
        path=f"{MEMORY_ROOT}/state.md",
        contents="v2",
        description="updated state",
    )
    assert [call["method"] for call in existing.calls] == ["POST", "PATCH"]
    assert existing.calls[1]["body"] == {
        "scope": "scope-1",
        "path": f"{MEMORY_ROOT}/state.md",
        "replace_all": {"contents": "v2"},
        "description": "updated state",
    }


def test_memory_paths_are_constrained_to_documented_root() -> None:
    client = ManagedMemoryClient(FakeApi(), "cat.sch.history")
    with pytest.raises(ValueError, match="/memories/"):
        client.upsert_entry(scope="scope-1", path="/tmp/state", contents="x", description="bad")
    assert cohort_memory_path("cohort:1") == f"{MEMORY_ROOT}/cohorts/cohort_1.md"
