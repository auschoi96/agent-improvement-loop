"""Register the L2 scorers as **scheduled scorers** on an MLflow experiment.

The scorers in :mod:`ail.judges.scorers` are just *definitions* until they are
registered against a live experiment. This module operationalizes them: it
registers ``correctness`` / ``modularity`` / ``groundedness`` as **scheduled**
scorers (ongoing production monitoring) on a Databricks-managed MLflow
experiment, so MLflow evaluates incoming traces automatically and attaches each
verdict as an assessment **on the scored trace** (``source_type=LLM_JUDGE``).
See ``docs/ARCHITECTURE.md`` (feedback-attachment architecture) for why a
verdict lives on the subject trace rather than nested inside it.

Backend: a Databricks scheduled scorer is served by ``databricks-agents``. The
registration calls themselves are the public ``mlflow.genai.scorers`` API
(``Scorer.register`` / ``Scorer.start`` / ``list_scorers`` / ``delete_scorer``),
but those delegate to the ``databricks-agents`` backend at runtime, so this
module fails fast with a clear message when that package is not installed.
Install it with the ``agents`` extra: ``pip install 'ail[agents]'``.

Cost lever — ``sampling_rate``: every scored trace runs the judge model over the
trace's inputs/outputs, and this corpus has rare ~900K-token traces, so scoring
100% of traces risks runaway cost. ``sampling_rate`` (0–1) is the knob; it
defaults to a conservative :data:`DEFAULT_SAMPLING_RATE` and is meant to be
raised deliberately, never silently pinned to 1.0.

Operational prerequisite (v4 UC trace store): registration makes the scorers
*visible* (``list_scorers``) and schedules them, but the background monitoring
job can only fetch Unity Catalog traces through a SQL warehouse. Until one is
wired up — ``MLFLOW_TRACING_SQL_WAREHOUSE_ID`` or the experiment tag
``mlflow.monitoring.sqlWarehouseId`` — the scorers are registered but score
nothing. Registering still surfaces them and is the prerequisite for scoring.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from ail.judges.scorers import DEFAULT_SCORERS, ScorerSpec, make_scorer

if TYPE_CHECKING:
    from mlflow.genai.scorers import Scorer

__all__ = [
    "DEFAULT_SAMPLING_RATE",
    "register_scorers",
    "list_registered_scorers",
    "unregister_scorers",
]

#: Conservative default fraction of traces a scheduled scorer evaluates. Kept
#: low on purpose: the reference corpus has rare ~900K-token traces and three
#: judges run per scored trace, so 100% sampling risks runaway cost. Raise it
#: deliberately for a given experiment; this is the documented cost lever.
DEFAULT_SAMPLING_RATE = 0.1


def _require_databricks_agents() -> None:
    """Fail fast with guidance if the scheduled-scorer backend is absent.

    The ``mlflow.genai.scorers`` registration API delegates to
    ``databricks-agents`` for a Databricks backend; without it, registration
    fails deep inside MLflow with an opaque error. Detect it up front (without
    importing the heavy package) and raise an actionable :class:`ImportError`.
    """
    if importlib.util.find_spec("databricks.agents") is None:
        raise ImportError(
            "scheduled-scorer registration requires the 'databricks-agents' package "
            "(it backs the mlflow.genai scheduled-scorer API). Install it with: "
            "pip install 'ail[agents]'  (or pip install databricks-agents)."
        )


def _configure_databricks(
    *,
    profile: str | None,
    tracking_uri: str,
    registry_uri: str,
) -> None:
    """Point MLflow at Databricks-managed MLflow + UC (mirrors the ingest seam).

    Follows the same configuration model as
    :class:`ail.ingest.mlflow_source.MLflowTraceSource`: tracking URI
    ``databricks``, registry URI ``databricks-uc``, and an optional CLI profile
    that selects the workspace. Host resolution is best-effort so ambient auth
    (``DATABRICKS_HOST`` / ``DATABRICKS_TOKEN``) is left untouched when set.
    """
    import mlflow

    if profile:
        os.environ.setdefault("DATABRICKS_CONFIG_PROFILE", profile)
        active = os.environ.get("DATABRICKS_CONFIG_PROFILE")
        if active and not os.environ.get("DATABRICKS_HOST"):
            try:
                from databricks.sdk import WorkspaceClient

                host = WorkspaceClient(profile=active).config.host
            except Exception:  # noqa: BLE001 - unusable profile: defer to ambient auth
                host = None
            if host:
                os.environ["DATABRICKS_HOST"] = host

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)


def register_scorers(
    experiment_id: str,
    *,
    sampling_rate: float = DEFAULT_SAMPLING_RATE,
    scorers: Mapping[str, ScorerSpec] = DEFAULT_SCORERS,
    model: str | None = None,
    filter_string: str | None = None,
    profile: str | None = None,
    tracking_uri: str = "databricks",
    registry_uri: str = "databricks-uc",
) -> list[Scorer]:
    """Register ``scorers`` as scheduled scorers on ``experiment_id`` and start them.

    For each spec, builds the MLflow ``Judge`` (:func:`ail.judges.scorers.make_scorer`),
    registers it (``Judge.register``), then activates scheduled scoring
    (``Judge.start``) at ``sampling_rate``. After this returns, the scorers are
    visible via :func:`list_registered_scorers` and MLflow evaluates incoming
    traces in the background, attaching each verdict as an ``LLM_JUDGE``
    assessment on the scored trace.

    Args:
        experiment_id: Target MLflow experiment id.
        sampling_rate: Fraction of incoming traces to score, in ``(0, 1]``. The
            cost lever; defaults to the conservative :data:`DEFAULT_SAMPLING_RATE`.
        scorers: Specs to register (defaults to the built-in
            ``correctness``/``modularity``/``groundedness`` set).
        model: Judge model URI. ``None`` uses MLflow's default judge model for
            the active (Databricks-managed) backend.
        filter_string: Optional ``search_traces``-compatible filter limiting
            which traces are scored.
        profile: Optional Databricks CLI profile selecting the workspace.
        tracking_uri / registry_uri: MLflow backends (Databricks-managed + UC by
            default).

    Returns:
        The list of active (scheduled) ``Scorer`` objects, one per spec.

    Raises:
        ValueError: If ``sampling_rate`` is not in ``(0, 1]`` or ``scorers`` is empty.
        ImportError: If ``databricks-agents`` is not installed.
    """
    _require_databricks_agents()
    if not 0.0 < sampling_rate <= 1.0:
        raise ValueError(f"sampling_rate must be in (0, 1], got {sampling_rate!r}")
    if not scorers:
        raise ValueError("no scorers to register")

    _configure_databricks(profile=profile, tracking_uri=tracking_uri, registry_uri=registry_uri)
    from mlflow.genai.scorers import ScorerSamplingConfig

    sampling_config = ScorerSamplingConfig(sample_rate=sampling_rate, filter_string=filter_string)
    active: list[Scorer] = []
    for spec in scorers.values():
        judge = make_scorer(spec, model=model)
        registered = judge.register(experiment_id=experiment_id)
        active.append(
            registered.start(experiment_id=experiment_id, sampling_config=sampling_config)
        )
    return active


def list_registered_scorers(
    experiment_id: str,
    *,
    profile: str | None = None,
    tracking_uri: str = "databricks",
    registry_uri: str = "databricks-uc",
) -> list[Scorer]:
    """List the scorers currently registered on ``experiment_id``.

    Thin wrapper over ``mlflow.genai.scorers.list_scorers`` that configures the
    Databricks backend first. Use it to verify a registration (count goes from 0
    to >0) and to read each scorer's active ``sample_rate``.
    """
    _require_databricks_agents()
    _configure_databricks(profile=profile, tracking_uri=tracking_uri, registry_uri=registry_uri)
    from mlflow.genai.scorers import list_scorers

    return list(list_scorers(experiment_id=experiment_id))


def unregister_scorers(
    experiment_id: str,
    *,
    names: Sequence[str] | None = None,
    profile: str | None = None,
    tracking_uri: str = "databricks",
    registry_uri: str = "databricks-uc",
) -> list[str]:
    """Stop and delete scheduled scorers on ``experiment_id``.

    For each target (``names`` if given, else every registered scorer) it stops
    scheduled scoring (best-effort — a scorer that was never started raises,
    which is ignored) and then deletes the registration. Returns the names
    actually removed. This is the inverse of :func:`register_scorers`, useful to
    roll back a registration or to re-register at a different sampling rate.
    """
    _require_databricks_agents()
    _configure_databricks(profile=profile, tracking_uri=tracking_uri, registry_uri=registry_uri)
    from mlflow.genai.scorers import delete_scorer, list_scorers

    existing = {s.name: s for s in list_scorers(experiment_id=experiment_id)}
    targets = list(names) if names is not None else list(existing)
    removed: list[str] = []
    for name in targets:
        scorer = existing.get(name)
        if scorer is None:
            continue
        try:
            scorer.stop(experiment_id=experiment_id)
        except Exception:  # noqa: BLE001 - a not-started scorer can't be stopped; still delete it
            pass
        delete_scorer(name=name, experiment_id=experiment_id)
        removed.append(name)
    return removed
