"""``ail-revert`` — the guarded champion-revert CLI (Phase-C lineage lane).

Revert = re-point a prompt's **champion alias** to a prior registered version. This
is the *undo* half of the lineage/audit surface (``docs/OBSERVABILITY_APP.md`` Phase
C): the timeline shows a change that did not actually improve things; this CLI rolls
the champion back to the version before it.

It is deliberately a **guarded CLI, not an in-app write button**:

* **Fail-closed.** Refuses an unknown agent and an unknown target version — it never
  points the champion at a version that does not exist.
* **Explicit audit.** Prints what the champion **WAS** (version + uri) and what it
  **BECOMES** (version + uri) before doing anything.
* **Dry-run by default.** Prints the planned change and writes nothing unless
  ``--yes`` is passed; the alias write is the only side effect, and only with
  ``--yes``.
* **No auto-publish.** Re-pointing the alias does not refresh the lineage table —
  the CLI reminds the operator to re-run ``python -m ail.publish_lineage`` so the
  app reflects the new champion.

Reuses the lineage lane's registry seam (:class:`ail.publish_lineage.LineageRegistryClient`
+ :func:`ail.publish_lineage.new_lineage_client`) and the prompt-registry's name
resolution / champion-alias convention — no MLflow access or alias logic is
reimplemented here.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from pathlib import Path

from ail.optimize.prompt_registry import (
    CHAMPION_ALIASES,
    DEFAULT_CATALOG,
    DEFAULT_PROMPT_NAME,
    DEFAULT_SCHEMA,
    resolve_prompt_name,
)
from ail.publish_lineage import (
    LineageRegistryClient,
    new_lineage_client,
)
from ail.registry import load_registry

#: The alias re-pointed by default. ``champion`` is the canonical production alias
#: (``docs/PROMPT_REGISTRY.md``); the synonyms in :data:`CHAMPION_ALIASES` are read
#: when reporting the current champion but the write targets exactly one alias.
DEFAULT_ALIAS = CHAMPION_ALIASES[0]

#: Exit code for a fail-closed refusal (unknown agent / unknown version). Distinct
#: from a usage error (argparse exits 2 too, but via SystemExit before we run).
EXIT_REFUSED = 2


def _describe(version: int | None, uri: str | None) -> str:
    """Render a champion pointer for the audit line."""
    if version is None:
        return "(no champion alias set)"
    return f"v{version}" + (f" ({uri})" if uri else "")


def revert_champion(
    *,
    agent_name: str,
    to_version: int,
    client: LineageRegistryClient,
    prompt_name: str = DEFAULT_PROMPT_NAME,
    alias: str = DEFAULT_ALIAS,
    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
    apply: bool = False,
    out: Callable[[str], None] = print,
) -> int:
    """Re-point ``alias`` to ``to_version`` of the agent's prompt (guarded).

    Returns ``0`` on success (dry-run or applied) and :data:`EXIT_REFUSED` on a
    fail-closed refusal. ``apply=False`` (the default) only prints the planned
    change. ``out`` is injectable so tests can capture the audit lines.
    """
    full_name = resolve_prompt_name(prompt_name, catalog=catalog, schema=schema)

    versions = list(client.search_prompt_versions(full_name))
    by_number = {int(v.version): v for v in versions}

    # Fail closed: never point the champion at a version that does not exist.
    if to_version not in by_number:
        have = ", ".join(f"v{n}" for n in sorted(by_number)) or "<none registered>"
        out(
            f"REFUSED: prompt {full_name!r} has no version {to_version} "
            f"(registered versions: {have}). Champion alias left unchanged."
        )
        return EXIT_REFUSED
    target = by_number[to_version]

    current = client.get_prompt_version_by_alias(full_name, alias)
    current_version = int(current.version) if current is not None else None

    out(f"Revert champion for agent {agent_name!r} (prompt {full_name})")
    out(f"  alias        : {alias}")
    out(f"  champion WAS : {_describe(current_version, getattr(current, 'uri', None))}")
    out(f"  champion BECOMES: {_describe(to_version, getattr(target, 'uri', None))}")

    if current_version == to_version:
        out(f"No change: champion alias {alias!r} already points at v{to_version}.")
        return 0

    if not apply:
        out("DRY RUN — no alias written. Re-run with --yes to apply this change.")
        return 0

    client.set_prompt_alias(full_name, alias, to_version)
    out(f"APPLIED: champion alias {alias!r} now points at v{to_version}.")
    out(
        "REMINDER: this did NOT refresh the lineage table. Re-run "
        "`python -m ail.publish_lineage` so the app reflects the reverted champion."
    )
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ail-revert",
        description="Guarded champion revert: re-point a champion alias to a prior version.",
    )
    parser.add_argument("agent_name", help="Registered agent whose prompt champion to revert.")
    parser.add_argument(
        "--to-version",
        type=int,
        required=True,
        help="The prior prompt version number to make the champion.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually write the alias. Without it the CLI is a dry run that only prints the plan.",
    )
    parser.add_argument(
        "--alias",
        default=DEFAULT_ALIAS,
        help=f"Alias to re-point (default: {DEFAULT_ALIAS}).",
    )
    parser.add_argument(
        "--prompt-name",
        default=DEFAULT_PROMPT_NAME,
        help="UC prompt leaf the alias lives on (default: the token-efficiency skill).",
    )
    parser.add_argument(
        "--registry",
        default="config/agents.yaml",
        help="Agent registry YAML (default: config/agents.yaml; falls back to the in-code seed).",
    )
    parser.add_argument(
        "--profile",
        default=os.environ.get("AIL_DATABRICKS_PROFILE", "dais-demo"),
        help="Databricks CLI profile (ignored if DATABRICKS_HOST/DATABRICKS_TOKEN are set).",
    )
    parser.add_argument("--catalog", default=DEFAULT_CATALOG)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    registry_path = args.registry if Path(args.registry).exists() else None
    registry = load_registry(registry_path)
    try:
        registry.get(args.agent_name)
    except KeyError as exc:
        # Fail closed on an unknown agent before any registry call.
        print(f"REFUSED: {exc}")
        return EXIT_REFUSED

    client = new_lineage_client(args.profile)
    return revert_champion(
        agent_name=args.agent_name,
        to_version=args.to_version,
        client=client,
        prompt_name=args.prompt_name,
        alias=args.alias,
        catalog=args.catalog,
        schema=args.schema,
        apply=args.yes,
    )


if __name__ == "__main__":
    raise SystemExit(main())
