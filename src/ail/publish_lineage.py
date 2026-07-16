"""Tier A — publish the **prompt-lineage / audit timeline** table.

This is the publish behind the observability app's Phase-C *lineage / audit /
revert* surface (``docs/OBSERVABILITY_APP.md`` Phase C): a per-agent **version
timeline sourced from the prompt registry** that makes every promotion auditable
— *what changed → from which optimization run → with what proven held-out delta →
is it the champion* — and the trail that lets a change which did **not** actually
improve things be reverted.

Like :mod:`ail.publish_versions`, it federates at publish time and presents a
single pane at query time:

* The provenance is **not recomputed here**. It was stamped onto each registered
  prompt *version* as ``ail.prompt.*`` tags by the human-gated promote step
  (:mod:`ail.optimize.prompt_registry`). This module only *reads* those versions
  back via the prompt-registry seam, parses the tags into flat rows, and writes
  one unified UC table. The app (Tier B) only ``SELECT``s from it.
* It is **honest by construction.** A version that was force-registered despite
  *not* beating its seed on the held-out split carries ``ail.prompt.forced=true``
  + the recorded reason; that is surfaced as :attr:`PromptLineageRow.is_forced_non_improving`
  (+ ``registration_reason``) so the audit trail can flag it — it must never read
  as a genuine improvement. The champion is whichever version the
  ``champion``/``production`` alias points at, resolved authoritatively from the
  registry (not inferred from a number).

Unified table written to ``<catalog>.<schema>`` (keyed by ``agent_name`` +
``version`` — one table for all agents, segmented in SQL):

* ``agent_prompt_lineage`` — one row per (agent, prompt version): source
  (seed/gepa-evolved), ``changed``, the GEPA scores, the held-out
  evolved/seed/delta savings, the candidate artifact pointer, the suite version,
  ``is_champion``, ``is_forced_non_improving`` (+ recorded reason), and when the
  version was registered.

Writes reuse :mod:`ail.publish`'s atomic, idempotent staging→``REPLACE WHERE``
swap, scoped by an ``agent_name`` predicate: re-publishing one agent replaces that
agent's whole slice (so a version removed upstream is dropped) and never disturbs
another agent's rows.

Run (publish the Claude Code agent's prompt lineage)::

    python -m ail.publish_lineage \\
        --registry config/agents.yaml \\
        --warehouse-id <SQL_WAREHOUSE_ID> --profile dais-demo
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from ail.optimize.prompt_registry import (
    CHAMPION_ALIASES,
    DEFAULT_CATALOG,
    DEFAULT_PROMPT_NAME,
    DEFAULT_SCHEMA,
    PROMPT_TAG_PREFIX,
    REGISTRY_URI,
    TRACKING_URI,
    PromptSource,
    resolve_prompt_name,
)
from ail.publish import _atomic_replace_table, _build_workspace_client, _execute, _lit
from ail.registry import Agent, AgentRegistry, load_registry

__all__ = [
    "SCHEMA_VERSION",
    "LINEAGE_TABLE",
    "LINEAGE_COLUMNS",
    "PromptLineageRow",
    "LineageRegistryClient",
    "build_lineage_rows",
    "publish_agent_lineage",
    "publish_lineage",
    "new_lineage_client",
    "main",
]

SCHEMA_VERSION = "ail.observability/v1"

#: Unified, agent-keyed lineage table the app reads (one table for all agents).
LINEAGE_TABLE = "agent_prompt_lineage"

# The provenance tag prefix (``ail.prompt.*``) and the champion alias names are owned
# by :mod:`ail.optimize.prompt_registry` (imported above): the promote step writes the
# tags / sets the alias under those names, and this module reads them back under the
# same names — a single source of truth, not a duplicated literal.


# ---------------------------------------------------------------------------
# Output contract (typed; the app reads the flat table below)
# ---------------------------------------------------------------------------


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PromptLineageRow(_Contract):
    """One registered prompt version's lineage, for the audit timeline.

    Every field is read straight off the version's ``ail.prompt.*`` provenance tags
    (or the version object itself) — nothing is recomputed. The two honesty fields
    are :attr:`is_champion` (the champion alias points here) and
    :attr:`is_forced_non_improving` (the version was force-registered despite not
    beating its seed; :attr:`registration_reason` records *why*) — the app MUST
    flag the latter and never style it as a proven improvement.
    """

    agent_name: str
    experiment_id: str
    prompt_name: str
    version: int
    uri: str | None = None
    source: str
    changed: bool | None = None
    gepa_best_val_score: float | None = None
    gepa_num_candidates: int | None = None
    holdout_evolved_savings_pct: float | None = None
    holdout_seed_savings_pct: float | None = None
    holdout_savings_delta_pct: float | None = None
    candidate_artifact: str | None = None
    suite_version: str | None = None
    is_champion: bool = False
    is_forced_non_improving: bool = False
    registration_reason: str | None = None
    registered_at: str | None = None
    generated_at: str | None = None


# ---------------------------------------------------------------------------
# Registry read seam (version-level; no live MLflow on import or in tests)
# ---------------------------------------------------------------------------


class LineageRegistryClient(Protocol):
    """The slice of the MLflow prompt-registry API the lineage + revert lane needs.

    ``ail.optimize.prompt_registry``'s :class:`~ail.optimize.prompt_registry.PromptRegistryClient`
    covers the *write* (register / set-alias) path; reading the version history and
    resolving the champion needs the version-level calls below (``mlflow.genai`` has
    no version listing — these live on ``MlflowClient``). The default implementation
    delegates to ``MlflowClient`` against the configured Databricks-UC registry;
    tests inject a fake exposing these three methods so no live call is ever made.
    """

    def search_prompt_versions(self, name: str) -> list[Any]:
        """All versions of prompt ``name`` (each has ``.version``/``.tags``/``.uri``)."""
        ...

    def get_prompt_version_by_alias(self, name: str, alias: str) -> Any | None:
        """The version ``alias`` points at, or ``None`` if the alias is unset."""
        ...

    def set_prompt_alias(self, name: str, alias: str, version: int) -> None:
        """Point ``alias`` at ``version`` of prompt ``name`` (the revert primitive)."""
        ...


# ---------------------------------------------------------------------------
# Provenance tags -> rows (pure; no I/O)
# ---------------------------------------------------------------------------


def _tag(tags: dict[str, str], field: str) -> str | None:
    """Read one ``ail.prompt.<field>`` tag (``None`` when absent)."""
    return tags.get(f"{PROMPT_TAG_PREFIX}.{field}")


def _bool_tag(tags: dict[str, str], field: str) -> bool | None:
    """Parse a ``true``/``false`` tag (the form :meth:`PromptProvenance.as_tags` writes)."""
    raw = _tag(tags, field)
    if raw is None:
        return None
    return raw == "true"


def _float_tag(tags: dict[str, str], field: str) -> float | None:
    """Parse a numeric tag as a float, or ``None`` if absent/unparseable."""
    raw = _tag(tags, field)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _int_tag(tags: dict[str, str], field: str) -> int | None:
    """Parse a numeric tag as an int (via float so ``"4.0"`` works), else ``None``."""
    raw = _tag(tags, field)
    if raw is None:
        return None
    try:
        return int(float(raw))
    except ValueError:
        return None


def _iso_from_ms(value: Any) -> str | None:
    """Render a registry creation timestamp (ms epoch) as an ISO-8601 UTC string.

    ``PromptVersion.creation_timestamp`` is a milliseconds-since-epoch int; a string
    is passed through (already formatted), and ``None``/garbage yields ``None`` so a
    missing timestamp is honest rather than a fabricated epoch.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return datetime.fromtimestamp(float(value) / 1000.0, tz=UTC).isoformat()
    except (TypeError, ValueError, OSError):
        return None


def build_lineage_rows(
    agent_name: str,
    prompt_name: str,
    versions: list[Any],
    *,
    experiment_id: str,
    champion_versions: set[int],
    generated_at: str | None,
) -> list[PromptLineageRow]:
    """Map a prompt's registered ``versions`` to :class:`PromptLineageRow`, newest first.

    ``champion_versions`` is the set of version numbers any champion alias points at
    (resolved authoritatively from the registry) — a version is the champion iff its
    number is in that set. ``is_forced_non_improving`` is read straight from the
    ``ail.prompt.forced`` tag the promote step stamps **only** on a force-registered
    non-improving candidate, so a legitimate seed (``changed=false`` but never
    forced) is correctly *not* flagged.
    """
    rows: list[PromptLineageRow] = []
    for v in versions:
        tags = dict(getattr(v, "tags", None) or {})
        version_num = int(v.version)
        source = _tag(tags, "source") or PromptSource.SEED.value
        rows.append(
            PromptLineageRow(
                agent_name=agent_name,
                experiment_id=experiment_id,
                prompt_name=prompt_name,
                version=version_num,
                uri=getattr(v, "uri", None),
                source=source,
                changed=_bool_tag(tags, "changed"),
                gepa_best_val_score=_float_tag(tags, "gepa_best_val_score"),
                gepa_num_candidates=_int_tag(tags, "gepa_num_candidates"),
                holdout_evolved_savings_pct=_float_tag(tags, "holdout_evolved_savings_pct"),
                holdout_seed_savings_pct=_float_tag(tags, "holdout_seed_savings_pct"),
                holdout_savings_delta_pct=_float_tag(tags, "holdout_savings_delta_pct"),
                candidate_artifact=_tag(tags, "candidate_artifact"),
                suite_version=_tag(tags, "suite_version"),
                is_champion=version_num in champion_versions,
                is_forced_non_improving=bool(_bool_tag(tags, "forced")),
                registration_reason=_tag(tags, "registration_reason"),
                registered_at=_iso_from_ms(getattr(v, "creation_timestamp", None)),
                generated_at=generated_at,
            )
        )
    rows.sort(key=lambda r: r.version, reverse=True)
    return rows


def champion_versions(client: LineageRegistryClient, full_name: str) -> set[int]:
    """The set of version numbers the champion/production alias(es) point at.

    Resolved authoritatively per alias via ``get_prompt_version_by_alias`` (an unset
    alias yields ``None`` and is skipped) — never inferred from version numbers, so a
    revert that re-points the alias is reflected the next publish.
    """
    found: set[int] = set()
    for alias in CHAMPION_ALIASES:
        version = client.get_prompt_version_by_alias(full_name, alias)
        if version is not None:
            found.add(int(version.version))
    return found


# ---------------------------------------------------------------------------
# Flat rows / DDL (column order declared once; reused by DDL + INSERTs)
# ---------------------------------------------------------------------------

LINEAGE_COLUMNS: list[str] = [
    "agent_name",
    "experiment_id",
    "prompt_name",
    "version",
    "uri",
    "source",
    "changed",
    "gepa_best_val_score",
    "gepa_num_candidates",
    "holdout_evolved_savings_pct",
    "holdout_seed_savings_pct",
    "holdout_savings_delta_pct",
    "candidate_artifact",
    "suite_version",
    "is_champion",
    "is_forced_non_improving",
    "registration_reason",
    "registered_at",
    "generated_at",
]


def _lineage_row(r: PromptLineageRow) -> list[Any]:
    return [
        r.agent_name,
        r.experiment_id,
        r.prompt_name,
        r.version,
        r.uri,
        r.source,
        r.changed,
        r.gepa_best_val_score,
        r.gepa_num_candidates,
        r.holdout_evolved_savings_pct,
        r.holdout_seed_savings_pct,
        r.holdout_savings_delta_pct,
        r.candidate_artifact,
        r.suite_version,
        r.is_champion,
        r.is_forced_non_improving,
        r.registration_reason,
        r.registered_at,
        r.generated_at,
    ]


def _ddl(catalog: str, schema: str) -> list[str]:
    fqn = f"`{catalog}`.`{schema}`"
    return [
        f"CREATE SCHEMA IF NOT EXISTS {fqn} "
        "COMMENT 'Agent self-optimization loop: L0 deterministic metrics (Tier A).'",
        f"""CREATE TABLE IF NOT EXISTS {fqn}.{LINEAGE_TABLE} (
            agent_name STRING,
            experiment_id STRING,
            prompt_name STRING,
            version INT,
            uri STRING,
            source STRING,
            changed BOOLEAN,
            gepa_best_val_score DOUBLE,
            gepa_num_candidates INT,
            holdout_evolved_savings_pct DOUBLE,
            holdout_seed_savings_pct DOUBLE,
            holdout_savings_delta_pct DOUBLE,
            candidate_artifact STRING,
            suite_version STRING,
            is_champion BOOLEAN,
            is_forced_non_improving BOOLEAN,
            registration_reason STRING,
            registered_at STRING,
            generated_at STRING
        ) USING DELTA
        COMMENT 'Per (agent, prompt version) lineage: provenance + champion + forced flag.'""",
    ]


# ---------------------------------------------------------------------------
# Publish orchestration
# ---------------------------------------------------------------------------


def _default_prompt_name_for_agent(agent: Agent) -> str:
    """The UC prompt leaf an agent's lineage is read from.

    Today every registered agent shares the single token-efficiency skill
    (:data:`~ail.optimize.prompt_registry.DEFAULT_PROMPT_NAME`). This is the one
    seam to extend when agents carry their own prompt name.
    """
    return DEFAULT_PROMPT_NAME


def publish_agent_lineage(
    agent: Agent,
    *,
    prompt_name: str,
    registry_client: LineageRegistryClient,
    warehouse_client: Any,
    warehouse_id: str,
    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
    generated_at: str | None = None,
) -> list[PromptLineageRow]:
    """Read one agent's prompt versions and write its slice of ``agent_prompt_lineage``.

    The whole agent slice is swapped in one atomic ``REPLACE WHERE agent_name = …``
    so a re-publish drops versions that no longer exist and never disturbs another
    agent. Empty (no registered version) is handled correctly — the swap clears any
    prior rows for the agent and writes nothing (the app's honest empty state).
    Returns the rows written (newest version first).
    """
    stamp = generated_at or datetime.now(UTC).isoformat()
    full_name = resolve_prompt_name(prompt_name, catalog=catalog, schema=schema)
    versions = list(registry_client.search_prompt_versions(full_name))
    rows = build_lineage_rows(
        agent.agent_name,
        full_name,
        versions,
        experiment_id=agent.experiment_id,
        champion_versions=champion_versions(registry_client, full_name),
        generated_at=stamp,
    )

    fqn = f"`{catalog}`.`{schema}`"
    for ddl in _ddl(catalog, schema):
        _execute(warehouse_client, warehouse_id, ddl)
    _atomic_replace_table(
        warehouse_client,
        warehouse_id,
        fqn,
        LINEAGE_TABLE,
        LINEAGE_COLUMNS,
        [_lineage_row(r) for r in rows],
        f"agent_name = {_lit(agent.agent_name)} AND experiment_id = {_lit(agent.experiment_id)}",
    )
    return rows


def publish_lineage(
    registry: AgentRegistry,
    *,
    registry_client: LineageRegistryClient,
    warehouse_client: Any,
    warehouse_id: str,
    prompt_name_for: Callable[[Agent], str] = _default_prompt_name_for_agent,
    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
    generated_at: str | None = None,
) -> dict[str, list[PromptLineageRow]]:
    """Publish the prompt lineage for every registered agent. Returns rows per agent."""
    stamp = generated_at or datetime.now(UTC).isoformat()
    out: dict[str, list[PromptLineageRow]] = {}
    for agent in registry.agents:
        out[agent.agent_name] = publish_agent_lineage(
            agent,
            prompt_name=prompt_name_for(agent),
            registry_client=registry_client,
            warehouse_client=warehouse_client,
            warehouse_id=warehouse_id,
            catalog=catalog,
            schema=schema,
            generated_at=stamp,
        )
    return out


def new_lineage_client(profile: str | None = None) -> LineageRegistryClient:
    """Build a live :class:`LineageRegistryClient` against the Databricks-UC registry.

    Mirrors :func:`ail.optimize.prompt_registry._new_prompt_client`: resolve the
    workspace host from the CLI profile, set the tracking + ``databricks-uc``
    registry URIs, and return a thin ``MlflowClient``-backed implementation. Built
    **only** when a human invokes a publish/revert with no injected client — the one
    place a live MLflow call originates; never touched on import or in tests.
    """
    try:
        import mlflow
        from mlflow import MlflowClient
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "the prompt lineage requires mlflow. Install it with: pip install 'mlflow>=3.14,<4'"
        ) from exc

    from ail.ingest.mlflow_source import MLflowTraceSource

    if profile:
        os.environ.setdefault("DATABRICKS_CONFIG_PROFILE", profile)
    MLflowTraceSource._resolve_workspace_host()
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(REGISTRY_URI)
    return _MlflowLineageClient(MlflowClient(tracking_uri=TRACKING_URI, registry_uri=REGISTRY_URI))


class _MlflowLineageClient:
    """Default :class:`LineageRegistryClient` delegating to a live ``MlflowClient``."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def search_prompt_versions(self, name: str) -> list[Any]:
        return list(self._client.search_prompt_versions(name))

    def get_prompt_version_by_alias(self, name: str, alias: str) -> Any | None:
        # An unset alias raises on the UC store; "no champion yet" is a legitimate
        # state (the version list is still the audit trail), so fail soft to None.
        try:
            return self._client.get_prompt_version_by_alias(name, alias)
        except Exception:
            return None

    def set_prompt_alias(self, name: str, alias: str, version: int) -> None:
        self._client.set_prompt_alias(name, alias, version)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Publish the unified per-agent prompt-lineage / audit-timeline table (Tier A)."
    )
    parser.add_argument(
        "--registry",
        default="config/agents.yaml",
        help="Agent registry YAML (default: config/agents.yaml; falls back to the in-code seed).",
    )
    parser.add_argument(
        "--prompt-name",
        default=DEFAULT_PROMPT_NAME,
        help="UC prompt leaf each agent's lineage is read from (default: token-efficiency skill).",
    )
    parser.add_argument(
        "--warehouse-id",
        default=os.environ.get("AIL_WAREHOUSE_ID"),
        help="SQL warehouse id used to create and populate the Delta table.",
    )
    parser.add_argument(
        "--profile",
        default=os.environ.get("AIL_DATABRICKS_PROFILE", "dais-demo"),
        help="Databricks CLI profile (ignored if DATABRICKS_HOST/DATABRICKS_TOKEN are set).",
    )
    parser.add_argument("--catalog", default=DEFAULT_CATALOG)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    args = parser.parse_args(argv)

    if not args.warehouse_id:
        parser.error("--warehouse-id is required (or set AIL_WAREHOUSE_ID)")

    from pathlib import Path

    registry_path = args.registry if Path(args.registry).exists() else None
    registry = load_registry(registry_path)

    registry_client = new_lineage_client(args.profile)
    warehouse_client = _build_workspace_client(args.profile)
    published = publish_lineage(
        registry,
        registry_client=registry_client,
        warehouse_client=warehouse_client,
        warehouse_id=args.warehouse_id,
        prompt_name_for=lambda _agent: args.prompt_name,
        catalog=args.catalog,
        schema=args.schema,
    )
    for agent_name, rows in published.items():
        champ = next((r.version for r in rows if r.is_champion), None)
        forced = sum(1 for r in rows if r.is_forced_non_improving)
        print(
            f"published agent={agent_name}: {len(rows)} version(s) -> "
            f"{args.catalog}.{args.schema}.{LINEAGE_TABLE}; "
            f"champion=v{champ if champ is not None else '?'}; forced_non_improving={forced}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
