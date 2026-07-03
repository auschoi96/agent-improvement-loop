"""Workspace-specific write-path configuration.

The reference catalog/schema constants remain documented defaults for the
maintainer workspace, but live write paths must resolve a deployer's explicit
workspace configuration instead of silently using those literals.
"""

from __future__ import annotations

import os

from ail.jobs.bootstrap_grants import _workspace_value_error

CATALOG_ENV = "AIL_CATALOG"
SCHEMA_ENV = "AIL_SCHEMA"
ALLOW_REFERENCE_ENV = "AIL_ALLOW_REFERENCE_WORKSPACE"


def resolve_catalog_schema(
    catalog: str | None = None,
    schema: str | None = None,
) -> tuple[str, str]:
    """Resolve and validate the UC catalog/schema for live writes.

    Values come from explicit arguments first, then ``AIL_CATALOG`` /
    ``AIL_SCHEMA``. Empty, placeholder, unresolved bundle-reference, and reference
    workspace values fail closed unless ``AIL_ALLOW_REFERENCE_WORKSPACE=1`` is set.
    """
    resolved_catalog = catalog if catalog is not None else os.environ.get(CATALOG_ENV)
    resolved_schema = schema if schema is not None else os.environ.get(SCHEMA_ENV)
    allow_reference = os.environ.get(ALLOW_REFERENCE_ENV) == "1"
    errors = [
        error
        for error in (
            _workspace_value_error(
                "catalog",
                resolved_catalog,
                allow_reference_workspace=allow_reference,
            ),
            _workspace_value_error(
                "schema",
                resolved_schema,
                allow_reference_workspace=allow_reference,
            ),
        )
        if error
    ]
    if errors:
        details = "; ".join(errors)
        raise RuntimeError(
            "Refusing to use empty, placeholder, unresolved, or reference workspace "
            f"catalog/schema for the write path ({details}); set {CATALOG_ENV} and "
            f"{SCHEMA_ENV} to your deployment values, pass catalog/schema explicitly, "
            f"or set {ALLOW_REFERENCE_ENV}=1 for the reference demo workspace."
        )
    return str(resolved_catalog).strip(), str(resolved_schema).strip()
