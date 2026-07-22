"""Unity Catalog Managed Memory support for recommendation planning.

Managed Memory is a supplemental retrieval layer.  The recommendation Delta tables
remain authoritative for evidence, queue state, human decisions, and outcomes; a
memory API outage must never erase or replace that governed state.

The public Databricks SDK does not currently expose typed Managed Memory methods, so
this module uses the SDK's authenticated ``ApiClient.do`` seam against the documented
Unity Catalog REST endpoints.  The seam is deliberately tiny and injectable so every
request shape is covered offline.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import quote

MEMORY_ROOT = "/memories/recommendations"
STATE_PATH = f"{MEMORY_ROOT}/state.md"
_STORE_PART_RE = re.compile(r"^[A-Za-z0-9_-]{1,255}$")


class MemoryApi(Protocol):
    """Authenticated subset of :class:`databricks.sdk.core.ApiClient`."""

    def do(
        self,
        method: str,
        path: str | None = None,
        *,
        query: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any] | list[Any]: ...


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    path: str
    contents: str
    description: str = ""
    scope: str = ""
    update_time: str = ""


def agent_memory_scope(agent_name: str, experiment_id: str) -> str:
    """Return the trusted per-agent isolation scope used for every API request.

    Scope is an authorization boundary in Managed Memory.  It is derived solely from
    the registered agent identity in framework code; neither model output nor request
    content can select it.  The hash avoids leaking agent/user identifiers into a
    partition key while remaining stable across planner runs.
    """

    digest = hashlib.sha256(f"{agent_name}\0{experiment_id}".encode()).hexdigest()
    return f"ail-agent-{digest[:32]}"


def cohort_memory_path(cohort_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", cohort_id).strip("_")
    if not safe:
        raise ValueError("cohort_id must contain at least one path-safe character")
    return f"{MEMORY_ROOT}/cohorts/{safe}.md"


def resolve_memory_store_name(value: str, *, catalog: str, schema: str) -> str:
    """Resolve a short or three-level memory-store name and validate every part."""

    raw = value.strip()
    parts = raw.split(".") if "." in raw else [catalog.strip(), schema.strip(), raw]
    if len(parts) != 3 or any(not _STORE_PART_RE.fullmatch(part) for part in parts):
        raise ValueError(
            "managed memory store must be a short name or catalog.schema.name; "
            "each part may contain only letters, digits, '_' or '-'"
        )
    return ".".join(parts)


def _status_code(exc: Exception) -> int | None:
    direct = getattr(exc, "status_code", None) or getattr(exc, "http_status_code", None)
    if isinstance(direct, int):
        return direct
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def _error_code(exc: Exception) -> str:
    return str(getattr(exc, "error_code", "") or "").upper()


def _is_not_found(exc: Exception) -> bool:
    return (
        _status_code(exc) == 404
        or exc.__class__.__name__.lower() in {"notfound", "resourcedoesnotexist"}
        or _error_code(exc) in {"NOT_FOUND", "RESOURCE_DOES_NOT_EXIST"}
    )


def _is_already_exists(exc: Exception) -> bool:
    return (
        _status_code(exc) == 409
        or exc.__class__.__name__.lower() in {"alreadyexists", "resourcealreadyexists"}
        or _error_code(exc) in {"ALREADY_EXISTS", "RESOURCE_ALREADY_EXISTS"}
    )


def _entry(payload: Any) -> MemoryEntry | None:
    if not isinstance(payload, dict):
        return None
    nested = payload.get("entry") or payload.get("memory_entry")
    row = nested if isinstance(nested, dict) else payload
    path = str(row.get("path") or "").strip()
    if not path:
        return None
    return MemoryEntry(
        path=path,
        contents=str(row.get("contents") or ""),
        description=str(row.get("description") or ""),
        scope=str(row.get("scope") or ""),
        update_time=str(row.get("update_time") or ""),
    )


class ManagedMemoryClient:
    """Minimal REST client for one governed Unity Catalog memory store."""

    def __init__(self, api_client: MemoryApi, store_name: str):
        self._api = api_client
        # Require a fully-qualified store at the transport boundary.
        self.store_name = resolve_memory_store_name(store_name, catalog="", schema="")
        encoded = quote(self.store_name, safe=".")
        self._store_path = f"/api/2.1/unity-catalog/memory-stores/{encoded}"

    def ensure_store(self, *, description: str) -> None:
        """Get the store, creating it only when the REST API returns not-found."""

        try:
            self._api.do("GET", self._store_path)
            return
        except Exception as exc:  # noqa: BLE001 - SDK error types vary by version
            if not _is_not_found(exc):
                raise
        catalog, schema, name = self.store_name.split(".", 2)
        self._api.do(
            "POST",
            "/api/2.1/unity-catalog/memory-stores",
            body={
                "name": name,
                "catalog_name": catalog,
                "schema_name": schema,
                "description": description,
            },
        )

    def get_entry(self, *, scope: str, path: str) -> MemoryEntry | None:
        _validate_scope_path(scope, path)
        try:
            raw = self._api.do(
                "GET",
                f"{self._store_path}/entries:get",
                query={"scope": scope, "path": path},
            )
        except Exception as exc:  # noqa: BLE001 - normalize only a documented absence
            if _is_not_found(exc):
                return None
            raise
        return _entry(raw)

    def list_entries(
        self,
        *,
        scope: str,
        path_prefix: str = MEMORY_ROOT,
        page_size: int = 100,
        max_entries: int = 500,
    ) -> list[MemoryEntry]:
        """List bounded entry metadata across pages.

        Managed Memory list responses intentionally omit ``contents``. Callers can
        inspect stable paths, descriptions, and update times before fetching only the
        bounded entries they plan to place in an LLM context. This is the recommended
        retrieval pattern while semantic search remains in Beta.
        """

        if not scope.strip():
            raise ValueError("managed memory scope must be non-empty")
        if not path_prefix.startswith("/memories/"):
            raise ValueError("managed memory path prefixes must start with /memories/")
        if not 1 <= page_size <= 1_000:
            raise ValueError("managed memory page_size must be from 1 to 1000")
        if not 1 <= max_entries <= 10_000:
            raise ValueError("managed memory max_entries must be from 1 to 10000")

        entries: list[MemoryEntry] = []
        page_token = ""
        seen_tokens: set[str] = set()
        while len(entries) < max_entries:
            request_query: dict[str, Any] = {
                "scope": scope,
                "path_prefix": path_prefix,
                "page_size": min(page_size, max_entries - len(entries)),
            }
            if page_token:
                request_query["page_token"] = page_token
            raw = self._api.do(
                "GET",
                f"{self._store_path}/entries",
                query=request_query,
            )
            if isinstance(raw, list):
                rows = raw
                next_page_token = ""
            elif isinstance(raw, dict):
                value = raw.get("entries") or raw.get("memory_entries") or []
                rows = value if isinstance(value, list) else []
                next_page_token = str(raw.get("next_page_token") or "")
            else:
                rows = []
                next_page_token = ""
            entries.extend(item for row in rows if (item := _entry(row)) is not None)
            if not next_page_token or next_page_token in seen_tokens:
                break
            seen_tokens.add(next_page_token)
            page_token = next_page_token
        return entries[:max_entries]

    def upsert_entry(
        self,
        *,
        scope: str,
        path: str,
        contents: str,
        description: str,
    ) -> None:
        """Create an entry, replacing its contents when the path already exists."""

        _validate_scope_path(scope, path)
        body = {"path": path, "contents": contents, "description": description}
        try:
            self._api.do(
                "POST",
                f"{self._store_path}/entries",
                query={"scope": scope},
                body=body,
            )
            return
        except Exception as exc:  # noqa: BLE001 - SDK error types vary by version
            if not _is_already_exists(exc):
                raise
        self._api.do(
            "PATCH",
            f"{self._store_path}/entries",
            body={
                "scope": scope,
                "path": path,
                "replace_all": {"contents": contents},
                "description": description,
            },
        )


def _validate_scope_path(scope: str, path: str) -> None:
    if not scope.strip():
        raise ValueError("managed memory scope must be non-empty")
    if not path.startswith("/memories/"):
        raise ValueError("managed memory paths must start with /memories/")
