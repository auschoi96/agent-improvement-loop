"""Leaf workspace-value guards shared by deploy/bootstrap and write paths."""

from __future__ import annotations

REFERENCE_WORKSPACE_DEFAULTS: dict[str, frozenset[str]] = {
    "experiment_id": frozenset({"660599403165942"}),
    "warehouse_id": frozenset({"7d1d3dbb3ba65f2a"}),
    "catalog": frozenset({"austin_choi_omni_agent_catalog"}),
    "schema": frozenset({"agent_improvement_loop"}),
}

PLACEHOLDER_VALUES = frozenset({"REPLACE_ME", "CHANGE_ME", "TODO", "TBD", "NONE", "NULL"})


def _workspace_value_error(
    var_name: str,
    value: str | None,
    *,
    required: bool = True,
    allow_reference_workspace: bool = False,
) -> str | None:
    cleaned = (value or "").strip()
    if not cleaned:
        if required:
            return f"{var_name} is empty"
        return None

    upper = cleaned.upper()
    if upper in PLACEHOLDER_VALUES or (cleaned.startswith("<") and cleaned.endswith(">")):
        return f"{var_name} is a placeholder"
    if cleaned.startswith("${") and cleaned.endswith("}"):
        return f"{var_name} is an unresolved bundle reference"
    reference_values = REFERENCE_WORKSPACE_DEFAULTS.get(var_name, frozenset())
    if not allow_reference_workspace and cleaned.casefold() in {
        reference.casefold() for reference in reference_values
    }:
        return f"{var_name} is a reference workspace default"
    return None
