"""Fresh-experiment **validation** and **creation** for the onboarding wizard.

Wizard page 1 (``docs/ONBOARDING_WIZARD.md`` §32): point an agent at a **fresh**
MLflow experiment (one agent per experiment), or **create** one from the app. Both
are handled here behind a narrow, injectable :class:`ExperimentClient` seam so the
orchestration is unit-testable with no live MLflow (mirroring the injectable client
seams in :mod:`ail.jobs.readiness_preflight` and :mod:`ail.judges.registration`).

**Honest / fail-closed, always.** "Fresh" means *empty of prior AIL state*: the
experiment exists, carries **no traces**, and is **not already claimed** by a
registered agent. If the identity cannot read the experiment or its traces
(auth / permission), validation raises :class:`ExperimentAccessError` — it never
reports "fresh" it could not verify. Creation is permission-sensitive: the app
service principal needs **experiment-create** authority; when it is missing the
MLflow call fails and :func:`create_experiment` surfaces an
:class:`ExperimentPermissionError` naming the prerequisite — it never reports a
created experiment that did not actually get created.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "ExperimentInfo",
    "ExperimentClient",
    "ExperimentValidation",
    "ExperimentCreation",
    "ExperimentAccessError",
    "ExperimentPermissionError",
    "validate_experiment",
    "create_experiment",
    "MlflowExperimentClient",
    "build_experiment_client",
    "FRESHNESS_TRACE_PROBE",
]

#: How many traces to probe when checking freshness. We only need to know whether
#: *any* prior traces exist; a small cap bounds the search latency and the count is
#: reported honestly (``trace_count_capped`` when the probe is saturated).
FRESHNESS_TRACE_PROBE = 25


class ExperimentAccessError(RuntimeError):
    """Could not read the experiment / its traces (auth or permission).

    Carries an actionable message (which profile, that the identity needs
    ``CAN_VIEW`` on the experiment and ``CAN_USE`` on the trace-store warehouse).
    The wizard surfaces it as an honest error — never a fabricated "fresh".
    """


class ExperimentPermissionError(RuntimeError):
    """The service principal lacks authority to CREATE an MLflow experiment.

    The documented prerequisite for the deploy: the app SP needs experiment-create
    authority in the workspace (same discipline as the warehouse ``CAN_USE`` grant).
    Raised instead of ever reporting a created experiment that did not get created.
    """


@dataclass(frozen=True, slots=True)
class ExperimentInfo:
    """The minimal experiment identity the wizard needs."""

    experiment_id: str
    name: str


@runtime_checkable
class ExperimentClient(Protocol):
    """The narrow MLflow surface the wizard needs — injectable/fakeable in tests.

    A live implementation (:class:`MlflowExperimentClient`) adapts an
    ``mlflow.MlflowClient``; a fake in tests returns canned info / raises, so no
    test touches a live workspace. Every method must be **honest**: a permission
    failure raises, a genuinely-absent experiment returns ``None`` — the two are
    never conflated (fail-closed distinguishes "not there" from "cannot tell").
    """

    def get_experiment(self, experiment_id: str) -> ExperimentInfo | None:
        """The experiment, or ``None`` if it genuinely does not exist. Raises on auth."""
        ...

    def get_experiment_by_name(self, name: str) -> ExperimentInfo | None:
        """The experiment named ``name``, or ``None`` if none. Raises on auth."""
        ...

    def create_experiment(self, name: str) -> str:
        """Create an experiment and return its id. Raises when creation is denied."""
        ...

    def count_traces(self, experiment_id: str, *, limit: int) -> int:
        """Count up to ``limit`` traces in the experiment. Raises on read failure."""
        ...


@dataclass(frozen=True, slots=True)
class ExperimentValidation:
    """The result of validating an experiment for freshness (page 1).

    ``fresh`` is the single honest verdict: the experiment exists, has zero traces,
    and no registered agent already claims it. ``reasons`` enumerate exactly why a
    non-fresh experiment was rejected (so the UI can say precisely what is wrong).
    """

    experiment_id: str
    name: str
    exists: bool
    fresh: bool
    trace_count: int
    trace_count_capped: bool
    already_registered: bool
    registered_as: str | None
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ExperimentCreation:
    """The result of creating a fresh experiment (page 1)."""

    experiment_id: str
    name: str


def validate_experiment(
    experiment_id: str,
    *,
    client: ExperimentClient,
    claimed_experiment_ids: dict[str, str] | None = None,
    trace_probe: int = FRESHNESS_TRACE_PROBE,
) -> ExperimentValidation:
    """Validate that ``experiment_id`` is a **fresh** target (fail-closed).

    Fresh ⇔ the experiment exists **and** carries no traces **and** is not already
    claimed by a registered agent (``claimed_experiment_ids`` maps a claimed
    experiment id → the agent that owns it). A read failure inside ``client``
    propagates (the caller turns it into an honest error) — this function never
    invents a "fresh" verdict on missing information.

    Args:
        experiment_id: The MLflow experiment id to validate.
        client: The (injected) experiment surface.
        claimed_experiment_ids: Experiment ids already owned by a registered agent.
        trace_probe: How many traces to probe (freshness only needs "any?").
    """
    claimed = claimed_experiment_ids or {}
    info = client.get_experiment(experiment_id)
    if info is None:
        return ExperimentValidation(
            experiment_id=experiment_id,
            name="",
            exists=False,
            fresh=False,
            trace_count=0,
            trace_count_capped=False,
            already_registered=False,
            registered_as=None,
            reasons=[
                f"no MLflow experiment with id {experiment_id!r} is visible to this "
                "identity — check the id, or create a fresh experiment instead"
            ],
        )

    n = client.count_traces(experiment_id, limit=trace_probe)
    capped = n >= trace_probe
    owner = claimed.get(experiment_id)
    reasons: list[str] = []
    if n > 0:
        shown = f"{trace_probe}+" if capped else str(n)
        reasons.append(
            f"experiment already has {shown} trace(s) — one agent per experiment; "
            "point at a new/empty experiment or create one so prior traces are not mixed in"
        )
    if owner is not None:
        reasons.append(f"experiment is already registered to agent {owner!r}")

    fresh = n == 0 and owner is None
    return ExperimentValidation(
        experiment_id=experiment_id,
        name=info.name,
        exists=True,
        fresh=fresh,
        trace_count=n,
        trace_count_capped=capped,
        already_registered=owner is not None,
        registered_as=owner,
        reasons=reasons,
    )


def create_experiment(name: str, *, client: ExperimentClient) -> ExperimentCreation:
    """Create a fresh experiment named ``name`` (fail-closed, honest on denial).

    Refuses to create when an experiment of that name already exists (it may be
    another agent's — never silently reuse it). A creation denied by the workspace
    is surfaced as :class:`ExperimentPermissionError` with the documented
    prerequisite; only a genuinely-created experiment returns a result.
    """
    clean = name.strip()
    if not clean:
        raise ValueError("an experiment name is required to create one")
    existing = client.get_experiment_by_name(clean)
    if existing is not None:
        raise ValueError(
            f"an experiment named {clean!r} already exists (id {existing.experiment_id}); "
            "choose a different name or validate that experiment instead of creating it"
        )
    experiment_id = client.create_experiment(clean)
    if not experiment_id:
        # Fail-closed: the client returned no id — do NOT report a creation.
        raise ExperimentPermissionError(
            f"MLflow returned no experiment id when creating {clean!r} — refusing to "
            "report a created experiment that did not get created"
        )
    return ExperimentCreation(experiment_id=str(experiment_id), name=clean)


# ---------------------------------------------------------------------------
# Live MLflow implementation (lazy imports; no MLflow touched until used)
# ---------------------------------------------------------------------------

#: MLflow REST error code for a genuinely-absent resource — the one case we map to
#: ``None`` rather than re-raising (everything else, incl. permission, propagates).
_NOT_FOUND_CODES = frozenset({"RESOURCE_DOES_NOT_EXIST", "NOT_FOUND", "ENDPOINT_NOT_FOUND"})
#: Substrings that mark a permission/authorization failure on create — surfaced as
#: the honest, actionable ExperimentPermissionError with the deploy prerequisite.
_PERMISSION_MARKERS = ("PERMISSION_DENIED", "permission", "not authorized", "forbidden")


class MlflowExperimentClient:
    """Live :class:`ExperimentClient` over Databricks-managed MLflow.

    Mirrors the workspace/tracking configuration of
    :func:`ail.ingest.mlflow_source._new_tag_client`: tracking URI ``databricks``,
    registry ``databricks-uc``, the active CLI profile selecting the workspace. All
    MLflow imports are lazy so constructing the wizard never pulls the MLflow
    runtime until a live call is actually made.
    """

    def __init__(
        self,
        *,
        profile: str | None = None,
        tracking_uri: str = "databricks",
        registry_uri: str = "databricks-uc",
    ) -> None:
        self._profile = profile
        self._tracking_uri = tracking_uri
        self._registry_uri = registry_uri
        self._client: Any = None

    def _ensure(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import mlflow
            from mlflow import MlflowClient
        except ImportError as exc:  # pragma: no cover - import guard
            raise ExperimentAccessError(
                "the onboarding write-path requires mlflow (pip install 'mlflow>=3.14,<4')"
            ) from exc
        if self._profile:
            os.environ.setdefault("DATABRICKS_CONFIG_PROFILE", self._profile)
        mlflow.set_tracking_uri(self._tracking_uri)
        mlflow.set_registry_uri(self._registry_uri)
        self._client = MlflowClient()
        return self._client

    def get_experiment(self, experiment_id: str) -> ExperimentInfo | None:
        client = self._ensure()
        try:
            exp = client.get_experiment(experiment_id)
        except Exception as exc:  # noqa: BLE001 - classify not-found vs auth (fail-closed)
            if _is_not_found(exc):
                return None
            raise ExperimentAccessError(_access_hint(experiment_id, self._profile, exc)) from exc
        if exp is None:
            return None
        return ExperimentInfo(experiment_id=str(exp.experiment_id), name=str(exp.name))

    def get_experiment_by_name(self, name: str) -> ExperimentInfo | None:
        client = self._ensure()
        try:
            exp = client.get_experiment_by_name(name)
        except Exception as exc:  # noqa: BLE001 - not-found -> None, else honest error
            if _is_not_found(exc):
                return None
            raise ExperimentAccessError(_access_hint(name, self._profile, exc)) from exc
        if exp is None:
            return None
        return ExperimentInfo(experiment_id=str(exp.experiment_id), name=str(exp.name))

    def create_experiment(self, name: str) -> str:
        client = self._ensure()
        try:
            return str(client.create_experiment(name))
        except Exception as exc:  # noqa: BLE001 - a denied create is an honest permission error
            if _is_permission(exc):
                raise ExperimentPermissionError(_create_hint(name, self._profile, exc)) from exc
            raise ExperimentAccessError(
                f"could not create experiment {name!r}: {type(exc).__name__}: {exc}"
            ) from exc

    def count_traces(self, experiment_id: str, *, limit: int) -> int:
        self._ensure()
        try:
            import mlflow

            traces = mlflow.search_traces(
                locations=[experiment_id], max_results=limit, return_type="list"
            )
        except Exception as exc:  # noqa: BLE001 - a read failure is an honest access error
            raise ExperimentAccessError(_access_hint(experiment_id, self._profile, exc)) from exc
        return len(traces)


def build_experiment_client(profile: str | None = None) -> ExperimentClient:
    """The live experiment seam pointed at the configured Databricks workspace."""
    return MlflowExperimentClient(profile=profile)


def _error_code(exc: Exception) -> str:
    """Best-effort MLflow/Databricks REST error code off ``exc`` (``""`` if none)."""
    code = getattr(exc, "error_code", None)
    return "" if code is None else str(code)


def _is_not_found(exc: Exception) -> bool:
    if _error_code(exc) in _NOT_FOUND_CODES:
        return True
    msg = str(exc).lower()
    return "does not exist" in msg or "not found" in msg


def _is_permission(exc: Exception) -> bool:
    code = _error_code(exc)
    if code and "PERMISSION" in code.upper():
        return True
    msg = str(exc).lower()
    return any(marker.lower() in msg for marker in _PERMISSION_MARKERS)


def _access_hint(target: str, profile: str | None, exc: Exception) -> str:
    prof = profile or os.environ.get("DATABRICKS_CONFIG_PROFILE") or "(default/ambient)"
    return (
        f"could not read experiment {target!r} (profile={prof}): "
        f"{type(exc).__name__}: {exc}. Check the identity has CAN_VIEW on the "
        "experiment and CAN_USE on the SQL warehouse backing the UC trace store. "
        "No freshness verdict was produced."
    )


def _create_hint(name: str, profile: str | None, exc: Exception) -> str:
    prof = profile or os.environ.get("DATABRICKS_CONFIG_PROFILE") or "(default/ambient)"
    return (
        f"the app service principal is not authorized to create MLflow experiment "
        f"{name!r} (profile={prof}): {type(exc).__name__}: {exc}. "
        "PREREQUISITE: grant the app SP experiment-create authority in the workspace "
        "(the same deploy-time grant discipline as the warehouse CAN_USE grant). "
        "No experiment was created."
    )
