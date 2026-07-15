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

Coverage contract — ``sampling_rate`` defaults to ``1.0`` so every incoming
subject trace is evaluated. Deployers may lower it explicitly for a constrained
environment, but the autonomous framework never silently samples away evidence.

MemAlign-aware by construction: creating a scorer routes through the
**align-then-register** flow (:func:`create_aligned_scorer`). When a non-empty
labeled Alignment Set is supplied, the judge is aligned with MemAlign
(:func:`ail.judges.alignment.align_judge`) *before* it is registered, so the
registered scorer is the **aligned** judge. When no labels exist yet — the state
of the reference experiment today, which has zero human labels — the base judge
is registered and flagged ``aligned=false`` (an authoritative
:class:`~ail.judges.contract.AlignmentReport` plus a best-effort experiment tag),
so the agreement floor / distrusted machinery treats it as *not yet trusted*
until labels exist and it is aligned and audited against the Human Anchor. This
makes MemAlign the default path whenever labels are present, rather than an
optional afterthought.

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
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ail.judges.alignment import align_judge, build_memalign_optimizer, unaligned_report
from ail.judges.contract import AlignmentReport
from ail.judges.scorers import DEFAULT_SCORERS, ScorerSpec, make_scorer

if TYPE_CHECKING:
    from mlflow.genai.judges.base import AlignmentOptimizer
    from mlflow.genai.scorers import Scorer

    from ail.pools import AlignmentSet

__all__ = [
    "DEFAULT_SAMPLING_RATE",
    "ALIGNED_TAG_PREFIX",
    "ScorerRegistration",
    "create_aligned_scorer",
    "register_prealigned_scorer",
    "register_scorers",
    "list_registered_scorers",
    "unregister_scorers",
]

#: Evaluate every incoming subject trace by default. A separate idempotent backfill
#: job repairs historical/missed coverage, so the intended steady state is 100%.
DEFAULT_SAMPLING_RATE = 1.0

#: Prefix for the best-effort experiment tag recording a judge's alignment state.
#: The scheduled-scorer API exposes no per-scorer metadata field, so the
#: ``aligned`` flag is recorded (authoritatively) on the returned
#: :class:`~ail.judges.contract.AlignmentReport` and (queryably, best-effort) as
#: an experiment tag ``ail.judge.<name>.aligned = "true" | "false"``.
ALIGNED_TAG_PREFIX = "ail.judge."


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


@dataclass(frozen=True, slots=True)
class ScorerRegistration:
    """The outcome of one align-then-register: the active scorer + its provenance.

    ``scorer`` is the live (scheduled) ``Scorer`` doing background scoring;
    ``judge`` is the registered ``Judge`` object (the **aligned** judge when a
    labeled set was supplied, else the base judge) — callable, so the agreement
    cadence can audit *this* judge via :func:`ail.judges.agreement.score_anchor`
    without re-aligning. ``aligned`` mirrors ``report.aligned`` for convenience;
    ``report`` is the serializable :class:`~ail.judges.contract.AlignmentReport`
    recording whether the judge was MemAlign-aligned before registration (and,
    when it was not, why it is flagged not-yet-trusted).
    """

    scorer: Scorer
    judge: Scorer
    aligned: bool
    report: AlignmentReport


def _tag_alignment(experiment_id: str, judge_name: str, aligned: bool) -> bool:
    """Best-effort record of a judge's alignment state as an experiment tag.

    Writes ``ail.judge.<name>.aligned = "true"|"false"``. The scheduled-scorer
    API has no per-scorer metadata slot, so this is the queryable companion to
    the authoritative flag on the returned :class:`AlignmentReport`. Best-effort:
    any failure (offline, no permissions, missing experiment) is swallowed and
    reported as ``False`` — tagging never blocks a registration.
    """
    try:
        from mlflow import MlflowClient

        MlflowClient().set_experiment_tag(
            experiment_id, f"{ALIGNED_TAG_PREFIX}{judge_name}.aligned", str(aligned).lower()
        )
    except Exception:  # noqa: BLE001 - tagging is provenance, never a precondition
        return False
    return True


def create_aligned_scorer(
    spec: ScorerSpec,
    *,
    experiment_id: str,
    alignment_set: AlignmentSet | None = None,
    optimizer: AlignmentOptimizer | None = None,
    model: str | None = None,
    sampling_rate: float = DEFAULT_SAMPLING_RATE,
    filter_string: str | None = None,
    profile: str | None = None,
    tracking_uri: str = "databricks",
    registry_uri: str = "databricks-uc",
) -> ScorerRegistration:
    """Build, **align-when-labels-exist**, then register one scheduled scorer.

    The MemAlign-aware creation path:

    1. Build the judge from ``spec`` (:func:`ail.judges.scorers.make_scorer`).
    2. **If** ``alignment_set`` is non-empty, align the judge with MemAlign
       (:func:`ail.judges.alignment.align_judge`) using ``optimizer`` — or a
       freshly :func:`~ail.judges.alignment.build_memalign_optimizer` one when
       ``optimizer`` is ``None`` — and register the **aligned** judge.
    3. **Else** register the base judge and flag it ``aligned=false`` (the
       experiment has no human labels yet, so MemAlign has nothing to learn
       from). The flag is recorded on the returned report and best-effort as an
       experiment tag; the agreement floor treats an unaligned, unmeasured judge
       as distrusted (fail-closed) until it is aligned and audited.

    Args:
        spec: The scorer definition to build and register.
        experiment_id: Target MLflow experiment id.
        alignment_set: Optional labeled Alignment-Set traces. When non-empty, the
            judge is MemAlign-aligned before registration; when ``None`` or empty,
            the base judge is registered and flagged unaligned.
        optimizer: Optional pre-built alignment optimizer; only consulted when
            aligning. ``None`` builds a default MemAlign optimizer (which requires
            the optional ``dspy`` dependency — see
            :func:`~ail.judges.alignment.build_memalign_optimizer`).
        model: Judge model URI. ``None`` uses MLflow's default judge model.
        sampling_rate: Fraction of traces to score, in ``(0, 1]``.
        filter_string: Optional ``search_traces`` filter limiting scored traces.
        profile / tracking_uri / registry_uri: MLflow/Databricks backend config.

    Returns:
        A :class:`ScorerRegistration` with the active scorer and its alignment
        provenance.

    Raises:
        ValueError: If ``sampling_rate`` is not in ``(0, 1]``.
        ImportError: If ``databricks-agents`` (or, when aligning, ``dspy``) is
            not installed.
    """
    _require_databricks_agents()
    if not 0.0 < sampling_rate <= 1.0:
        raise ValueError(f"sampling_rate must be in (0, 1], got {sampling_rate!r}")
    _configure_databricks(profile=profile, tracking_uri=tracking_uri, registry_uri=registry_uri)

    judge = make_scorer(spec, model=model)
    if alignment_set is not None and len(alignment_set) > 0:
        opt = optimizer if optimizer is not None else build_memalign_optimizer()
        outcome = align_judge(judge, alignment_set, optimizer=opt)
        judge_to_register, report = outcome.judge, outcome.report
    else:
        report = unaligned_report(getattr(judge, "name", spec.name))
        judge_to_register = judge

    scorer = _register_and_start(
        judge_to_register,
        experiment_id=experiment_id,
        sampling_rate=sampling_rate,
        filter_string=filter_string,
    )
    # Tag with the stable spec name (MemAlign preserves the judge name on align),
    # so the provenance tag key does not shift between the base and aligned judge.
    _tag_alignment(experiment_id, spec.name, report.aligned)
    return ScorerRegistration(
        scorer=scorer, judge=judge_to_register, aligned=report.aligned, report=report
    )


def register_prealigned_scorer(
    judge: Scorer,
    report: AlignmentReport,
    *,
    experiment_id: str,
    sampling_rate: float = DEFAULT_SAMPLING_RATE,
    filter_string: str | None = None,
    profile: str | None = None,
    tracking_uri: str = "databricks",
    registry_uri: str = "databricks-uc",
) -> ScorerRegistration:
    """Register an **already-aligned** judge, skipping the align step.

    The measure-before-register companion to :func:`create_aligned_scorer`. That
    function aligns *and* registers in one call, so a caller cannot inspect the
    aligned judge before it goes live. The auto-align trigger
    (:mod:`ail.judges.auto_align`) must: align a judge, measure its held-out
    agreement, and register **only** when it passes the floor and does not
    regress — and it must register *that exact judge*, so the judge whose
    agreement was measured is the judge that goes live (the honest basis for the
    fail-closed rollback). Re-aligning through :func:`create_aligned_scorer` would
    run MemAlign a second time and register a *different* judge than the one
    measured, breaking that guarantee (and paying for a second reflection pass).

    This reuses the same backend guard, Databricks config, register/start, and
    alignment tag as :func:`create_aligned_scorer`; it only omits the alignment.
    ``report`` is the :class:`~ail.judges.contract.AlignmentReport` for the judge
    already produced by :func:`ail.judges.alignment.align_judge`, carried through
    onto the returned registration.

    Raises:
        ValueError: If ``sampling_rate`` is not in ``(0, 1]``.
        ImportError: If ``databricks-agents`` is not installed.
    """
    _require_databricks_agents()
    if not 0.0 < sampling_rate <= 1.0:
        raise ValueError(f"sampling_rate must be in (0, 1], got {sampling_rate!r}")
    _configure_databricks(profile=profile, tracking_uri=tracking_uri, registry_uri=registry_uri)

    scorer = _register_and_start(
        judge,
        experiment_id=experiment_id,
        sampling_rate=sampling_rate,
        filter_string=filter_string,
    )
    _tag_alignment(experiment_id, report.base_judge_name, report.aligned)
    return ScorerRegistration(scorer=scorer, judge=judge, aligned=report.aligned, report=report)


def _register_and_start(
    judge: Scorer,
    *,
    experiment_id: str,
    sampling_rate: float,
    filter_string: str | None,
) -> Scorer:
    """Register a (possibly aligned) judge and start scheduled scoring.

    Assumes the Databricks backend is already configured (callers configure it
    once up front). Registers then starts, returning the active scorer.
    """
    from mlflow.genai.scorers import ScorerSamplingConfig

    sampling_config = ScorerSamplingConfig(sample_rate=sampling_rate, filter_string=filter_string)
    registered = judge.register(experiment_id=experiment_id)
    return registered.start(experiment_id=experiment_id, sampling_config=sampling_config)


def register_scorers(
    experiment_id: str,
    *,
    sampling_rate: float = DEFAULT_SAMPLING_RATE,
    scorers: Mapping[str, ScorerSpec] = DEFAULT_SCORERS,
    alignment_set: AlignmentSet | None = None,
    optimizer: AlignmentOptimizer | None = None,
    model: str | None = None,
    filter_string: str | None = None,
    profile: str | None = None,
    tracking_uri: str = "databricks",
    registry_uri: str = "databricks-uc",
) -> list[ScorerRegistration]:
    """Register ``scorers`` as scheduled scorers on ``experiment_id`` and start them.

    Every scorer routes through the MemAlign-aware :func:`create_aligned_scorer`
    path, so MemAlign is used **by construction whenever labels exist**: pass a
    non-empty ``alignment_set`` and *all* scorers are aligned before
    registration; omit it and they are registered as base judges flagged
    ``aligned=false`` (not yet trusted). After this returns, the scorers are
    visible via :func:`list_registered_scorers` and MLflow evaluates incoming
    traces in the background, attaching each verdict as an ``LLM_JUDGE``
    assessment on the scored trace.

    Args:
        experiment_id: Target MLflow experiment id.
        sampling_rate: Fraction of incoming traces to score, in ``(0, 1]``. The
            cost lever; defaults to the conservative :data:`DEFAULT_SAMPLING_RATE`.
        scorers: Specs to register (defaults to the built-in
            ``correctness``/``modularity``/``groundedness``/``token_efficiency`` set).
        alignment_set: Optional labeled Alignment Set. When non-empty, every
            scorer is MemAlign-aligned before registration; when ``None``/empty,
            base judges are registered and flagged unaligned.
        optimizer: Optional pre-built alignment optimizer (only used when
            aligning). ``None`` builds a default MemAlign optimizer.
        model: Judge model URI. ``None`` uses MLflow's default judge model for
            the active (Databricks-managed) backend.
        filter_string: Optional ``search_traces``-compatible filter limiting
            which traces are scored.
        profile: Optional Databricks CLI profile selecting the workspace.
        tracking_uri / registry_uri: MLflow backends (Databricks-managed + UC by
            default).

    Returns:
        One :class:`ScorerRegistration` per spec (active scorer + alignment
        provenance).

    Raises:
        ValueError: If ``sampling_rate`` is not in ``(0, 1]`` or ``scorers`` is empty.
        ImportError: If ``databricks-agents`` is not installed.
    """
    _require_databricks_agents()
    if not 0.0 < sampling_rate <= 1.0:
        raise ValueError(f"sampling_rate must be in (0, 1], got {sampling_rate!r}")
    if not scorers:
        raise ValueError("no scorers to register")

    return [
        create_aligned_scorer(
            spec,
            experiment_id=experiment_id,
            alignment_set=alignment_set,
            optimizer=optimizer,
            model=model,
            sampling_rate=sampling_rate,
            filter_string=filter_string,
            profile=profile,
            tracking_uri=tracking_uri,
            registry_uri=registry_uri,
        )
        for spec in scorers.values()
    ]


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
