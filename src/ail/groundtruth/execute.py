"""Stage 2 — execute the target agent to capture its own candidate response.

Runs a candidate case's task input through an
:class:`~ail.ingest.base.AgentAdapter` and records what the agent produced as a
:class:`~ail.groundtruth.schema.CandidateResponse`. This is the agent's *own*
output, captured verbatim for a human to judge — it is **not** an expected
output, and this stage never touches :class:`~ail.groundtruth.schema.Expectations`.

MLflow logging is optional and decoupled. :func:`log_candidate_run` records the
capture event (the input, the agent's response, the trace id) to whatever MLflow
tracking backend is configured — Databricks-managed MLflow in production
(``mlflow.set_tracking_uri("databricks")``), a local ``file://`` store in
tests. It logs **no** expectations (there are none), so the auditable record can
never seed co-adaptation. MLflow is imported lazily so the offline pipeline does
not require it at import time.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ail.groundtruth.schema import CandidateResponse, GroundTruthCase
from ail.ingest.base import AgentTask

if TYPE_CHECKING:
    from ail.ingest.base import AgentAdapter

__all__ = ["execute_candidate", "log_candidate_run", "MLFLOW_STAGE_TAG"]

#: Tag value stamped on MLflow runs produced by this stage, so capture runs are
#: trivially distinguishable from optimization/eval runs in the same experiment.
MLFLOW_STAGE_TAG = "ail.groundtruth.execute"

#: How much of the prompt / response we log as an MLflow *param* (params are
#: length-limited). The full response is always logged as a text artifact.
_PARAM_PREVIEW_CHARS = 250


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _task_from_case(case: GroundTruthCase) -> AgentTask:
    ti = case.task_input
    return AgentTask(
        prompt=ti.prompt,
        system_prompt=ti.system_prompt,
        model=ti.model,
        params=dict(ti.params),
    )


def execute_candidate(
    case: GroundTruthCase,
    adapter: AgentAdapter,
    *,
    log_to_mlflow: bool = False,
    experiment: str | None = None,
) -> GroundTruthCase:
    """Run the agent on ``case`` and attach its candidate response.

    Returns a **new** case (frozen) with
    :attr:`~ail.groundtruth.schema.GroundTruthCase.candidate_response` set to
    whatever the agent produced. Review state and expectations are left exactly
    as they were — executing an agent does not approve anything and does not
    invent expectations.

    Args:
        case: a candidate case (typically from :mod:`~ail.groundtruth.capture`).
        adapter: the agent to run. Per the :class:`~ail.ingest.base.AgentAdapter`
            contract, ordinary agent failures are captured on the result rather
            than raised, and surface here as ``candidate_response.success=False``.
        log_to_mlflow: if ``True``, also record the capture via
            :func:`log_candidate_run`.
        experiment: MLflow experiment id/name to log under (only when
            ``log_to_mlflow``).
    """
    result = adapter.run(_task_from_case(case))
    trace = result.trace
    candidate = CandidateResponse(
        output_text=result.output_text,
        producer=adapter.name,
        model=trace.model,
        trace_id=trace.trace_id,
        success=result.success,
        error=result.error,
        duration_ms=result.duration_ms,
        captured_at=_utc_now_iso(),
    )
    executed = case.model_copy(update={"candidate_response": candidate})
    if log_to_mlflow:
        log_candidate_run(executed, experiment=experiment)
    return executed


def log_candidate_run(
    case: GroundTruthCase,
    *,
    experiment: str | None = None,
    run_name: str | None = None,
) -> str | None:
    """Log a candidate execution to MLflow for auditability. Returns the run id.

    Records the task input, the agent's response, and the originating trace id.
    Deliberately logs **no expectations** — those do not exist at execute time
    and must never be model-derived. Returns ``None`` if the case has no
    candidate response to log (nothing was executed).
    """
    candidate = case.candidate_response
    if candidate is None:
        return None

    import mlflow  # lazy: the offline pipeline must not require mlflow at import

    if experiment is not None:
        # A numeric value is an experiment id; anything else is a name/path.
        if experiment.isdigit():
            mlflow.set_experiment(experiment_id=experiment)
        else:
            mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=run_name or f"capture-{case.case_id}") as run:
        mlflow.set_tags(
            {
                "ail.stage": MLFLOW_STAGE_TAG,
                "ail.groundtruth.case_id": case.case_id,
                "ail.groundtruth.schema_version": case.schema_version,
                "ail.groundtruth.review_status": case.review.status.value,
                "ail.groundtruth.trace_id": candidate.trace_id or "",
            }
        )
        params: dict[str, Any] = {
            "case_id": case.case_id,
            "producer": candidate.producer or "",
            "model": candidate.model or "",
            "success": candidate.success,
            "prompt_preview": case.task_input.prompt[:_PARAM_PREVIEW_CHARS],
        }
        mlflow.log_params(params)
        # Full text goes to artifacts (params are length-limited). The agent's
        # OWN response only — never an expectation.
        mlflow.log_text(case.task_input.prompt, "task_prompt.txt")
        mlflow.log_text(candidate.output_text, "candidate_response.txt")
        return run.info.run_id
