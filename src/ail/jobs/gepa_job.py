"""On-demand, UI-dispatched GEPA optimization job.

The Databricks App triggers this wheel entrypoint with bounded job parameters.  The
job then re-resolves the selected agent from the governed UC ``agent_registry`` and
validates the browser-supplied experiment id against that row before spending model
compute.  Today the executable path is intentionally limited to the reference
``claude_code`` agent; every other agent fails closed until its adapter is available
on job compute.

The result is a candidate only.  It is logged as ``gepa/gepa_candidate.json`` in the
agent's separate reviewer experiment and printed as a compact ``AIL_GEPA_RESULT``
marker for the AppKit run-output endpoint.  Nothing in this module registers,
promotes, or applies the evolved body.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from ail.compare import configure_monitoring_warehouse
from ail.ingest.adapters.claude_code import ClaudeCodeAdapter
from ail.jobs.multi_agent import resolve_registered_agent
from ail.jobs.publish_job import resolve_job_auth
from ail.loop.proposals import (
    ActionKind,
    ChangeKind,
    GateStatus,
    LocalApplySpec,
    LocalApplyTargetKind,
    ProofSummary,
    ProposedAction,
    ProposedChange,
    RiskClass,
    TriggerKind,
    TriggerSignal,
    derive_proposal_id,
)
from ail.loop.publish_proposals import insert_proposal_if_absent
from ail.optimize import VerifySpec
from ail.optimize.gepa_runner import (
    DEFAULT_REFLECTION_LM,
    GepaConfig,
    GepaOptimizationResult,
    run_gepa_optimization,
)
from ail.optimize.prompt_registry import candidate_improvement
from ail.publish import _build_workspace_client
from ail.registry import Agent, OptimizationTargetKind
from ail.task_suite.loader import load_task_suite

RESULT_MARKER = "AIL_GEPA_RESULT="
RESULT_ARTIFACT_PATH = "gepa/gepa_candidate.json"
OPTIMIZER_NAME = "gepa.optimize (Optimize Anything)"
SUPPORTED_AGENT_NAME = "claude_code"
DEFAULT_AGENT_MODEL = "databricks-claude-sonnet-4-6"
DEFAULT_SUITE_VERSION = "phase2-mini"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one explicitly confirmed, human-gated GEPA candidate search."
    )
    parser.add_argument("--agent", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--warehouse-id", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--suite-version", default=DEFAULT_SUITE_VERSION)
    parser.add_argument("--max-metric-calls", type=int, default=6)
    parser.add_argument("--holdout-fraction", type=float, default=0.4)
    parser.add_argument("--max-train-tasks", type=int, default=2)
    parser.add_argument("--reflection-lm", default=DEFAULT_REFLECTION_LM)
    parser.add_argument("--agent-model", default=DEFAULT_AGENT_MODEL)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--confirmed-costly-run", default="false")
    parser.add_argument(
        "--token-secret-scope", default=os.environ.get("AIL_TOKEN_SECRET_SCOPE", "")
    )
    parser.add_argument("--token-secret-key", default=os.environ.get("AIL_TOKEN_SECRET_KEY", ""))
    return parser.parse_args(argv)


def _assets_root() -> Path:
    """Return a root containing packaged ``eval/`` plus ``run_plan.yaml``.

    A serverless wheel has the force-included assets under ``ail/_eval_bundle``.
    Editable/local execution falls back to the repository root.  The caller still
    verifies every required path before running, so a partial package fails closed.
    """
    import ail

    packaged = Path(ail.__file__).resolve().parent / "_eval_bundle"
    if packaged.is_dir():
        return packaged
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "eval" / "task_suite").is_dir() and (parent / "run_plan.yaml").is_file():
            return parent
    return packaged


def _load_verify_specs(root: Path) -> dict[str, VerifySpec]:
    path = root / "run_plan.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"packaged GEPA run plan is missing: {path}")
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    specs: dict[str, VerifySpec] = {}
    for task_id, entry in raw.items():
        if not isinstance(entry, dict) or "command" not in entry:
            raise ValueError(f"run-plan entry {task_id!r} must contain a command")
        specs[str(task_id)] = VerifySpec(
            name=str(entry.get("name", f"verify-{task_id}")),
            command=entry["command"],
            cwd=entry.get("cwd"),
            shell=bool(entry.get("shell", False)),
            timeout_seconds=int(entry.get("timeout_seconds", 600)),
        )
    if not specs:
        raise ValueError("packaged GEPA run plan is empty")
    return specs


def _validate_request(args: argparse.Namespace) -> None:
    if args.confirmed_costly_run.lower() != "true":
        raise ValueError(
            "GEPA is live and costly; confirmed_costly_run must be true for an explicit dispatch"
        )
    if not 1 <= args.max_metric_calls <= 500:
        raise ValueError("max_metric_calls must be between 1 and 500")
    if not 0.0 < args.holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1")
    if not 1 <= args.max_train_tasks <= 20:
        raise ValueError("max_train_tasks must be between 1 and 20")
    if not args.reflection_lm.startswith("databricks:/"):
        raise ValueError("the dispatched reflection_lm must be a databricks:/ model URI")


def _resolve_agent(args: argparse.Namespace) -> Agent:
    agent = resolve_registered_agent(
        args.agent,
        warehouse_id=args.warehouse_id,
        catalog=args.catalog,
        schema=args.schema,
    )
    if agent.experiment_id != args.experiment_id:
        raise ValueError(
            f"experiment mismatch for agent {agent.agent_name!r}: the registry owns "
            f"{agent.experiment_id!r}, but the dispatcher supplied {args.experiment_id!r}"
        )
    if agent.agent_name != SUPPORTED_AGENT_NAME:
        raise ValueError(
            f"GEPA job compute currently supports only {SUPPORTED_AGENT_NAME!r}; "
            f"agent {agent.agent_name!r} has no executable adapter here"
        )
    if not agent.reviewer_experiment_id:
        raise ValueError(
            f"agent {agent.agent_name!r} has no reviewer_experiment_id; "
            "a separate reviewer experiment is required for optimizer traces and artifacts"
        )
    if agent.reviewer_experiment_id == agent.experiment_id:
        raise ValueError("reviewer_experiment_id must be separate from the subject experiment")
    if not agent.target_workspace:
        raise ValueError(
            f"agent {agent.agent_name!r} has no target_workspace; the local companion "
            "cannot apply an approved candidate"
        )
    if agent.optimization_target is None:
        raise ValueError(
            f"agent {agent.agent_name!r} has no optimization_target; configure a "
            "project-relative target and validation command before launching costly GEPA"
        )
    return agent


def _experiment_name(experiment_id: str) -> str:
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri("databricks")
    experiment = MlflowClient().get_experiment(experiment_id)
    if experiment is None or not experiment.name:
        raise ValueError(f"reviewer MLflow experiment {experiment_id!r} is not visible")
    return str(experiment.name)


def _configure_claude_fmapi(model: str) -> None:
    """Route Claude Agent SDK traffic through Databricks FMAPI with run-as auth."""
    host = os.environ.get("DATABRICKS_HOST", "")
    token = os.environ.get("DATABRICKS_TOKEN", "")
    if not host or not token:
        raise RuntimeError("resolved job auth did not provide DATABRICKS_HOST/TOKEN")
    clean_host = host.removeprefix("https://").removeprefix("http://").rstrip("/")
    os.environ.update(
        {
            "ANTHROPIC_BASE_URL": f"https://{clean_host}/serving-endpoints/anthropic",
            "ANTHROPIC_API_KEY": token,
            "ANTHROPIC_AUTH_TOKEN": token,
            "ANTHROPIC_MODEL": model,
            "ANTHROPIC_CUSTOM_HEADERS": "x-databricks-use-coding-agent-mode: true",
            "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
            "CLAUDE_CODE_STREAM_CLOSE_TIMEOUT": "3600000",
        }
    )


def _log_candidate(
    *, agent: Agent, result: GepaOptimizationResult, suite_version: str
) -> tuple[str, str]:
    """Log the candidate JSON in the reviewer experiment and return run/artifact URIs."""
    import mlflow

    reviewer_id = agent.reviewer_experiment_id
    if not reviewer_id:  # narrowed earlier; keep this side-effect boundary fail-closed
        raise ValueError("reviewer_experiment_id is required")
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(experiment_id=reviewer_id)
    with mlflow.start_run(run_name=f"gepa-candidate-{agent.agent_name}") as run:
        mlflow.log_params(
            {
                "ail.agent_name": agent.agent_name,
                "ail.subject_experiment_id": agent.experiment_id,
                "ail.reviewer_experiment_id": reviewer_id,
                "ail.gepa.suite_version": suite_version,
                "ail.gepa.max_metric_calls": result.max_metric_calls,
                "ail.gepa.reflection_lm": result.reflection_lm,
            }
        )
        mlflow.set_tags(
            {
                "ail.optimization": "gepa",
                "ail.candidate_only": "true",
                "ail.human_gate_required": "true",
                "ail.candidate_changed": str(result.changed).lower(),
            }
        )
        mlflow.log_dict(result.model_dump(mode="json"), RESULT_ARTIFACT_PATH)
        run_id = str(run.info.run_id)
    return run_id, f"runs:/{run_id}/{RESULT_ARTIFACT_PATH}"


def _result_marker(
    *,
    agent: Agent,
    result: GepaOptimizationResult,
    mlflow_run_id: str,
    artifact_uri: str,
    proposal: ProposedAction | None,
    proposal_reason: str,
) -> dict[str, Any]:
    evolved = result.holdout_evolved
    seed = result.holdout_seed_baseline
    workspace_host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
    mlflow_run_url = (
        f"{workspace_host}/ml/experiments/{agent.reviewer_experiment_id}/runs/{mlflow_run_id}"
        if workspace_host
        else None
    )
    return {
        "schema_version": "ail.jobs.gepa_result/v1",
        "agent_name": agent.agent_name,
        "subject_experiment_id": agent.experiment_id,
        "reviewer_experiment_id": agent.reviewer_experiment_id,
        "mlflow_run_id": mlflow_run_id,
        "mlflow_run_url": mlflow_run_url,
        "artifact_path": RESULT_ARTIFACT_PATH,
        "artifact_uri": artifact_uri,
        "optimizer": OPTIMIZER_NAME,
        "proposal_id": proposal.proposal_id if proposal is not None else None,
        "proposal_status": proposal.status.value if proposal is not None else None,
        "proposal_created": proposal is not None,
        "proposal_reason": proposal_reason,
        "candidate_changed": result.changed,
        "candidate_promoted": False,
        "human_gate_required": True,
        "suite_version": result.suite_version,
        "suite_content_hash": result.suite_content_hash,
        "max_metric_calls": result.max_metric_calls,
        "gepa_total_metric_calls": result.gepa_total_metric_calls,
        "gepa_num_candidates": result.gepa_num_candidates,
        "gepa_best_val_score": result.gepa_best_val_score,
        "holdout_savings_delta_pct": result.holdout_savings_delta_pct,
        "holdout_evolved_savings_pct": (
            evolved.realized_token_savings_pct if evolved is not None else None
        ),
        "holdout_seed_savings_pct": seed.realized_token_savings_pct if seed is not None else None,
    }


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _review_diff(target_path: str, seed: str, candidate: str) -> str:
    lines = difflib.unified_diff(
        seed.splitlines(),
        candidate.splitlines(),
        fromfile=f"a/{target_path}",
        tofile=f"b/{target_path}",
        lineterm="",
    )
    return "\n".join(lines) + "\n"


def _local_target_kind(kind: OptimizationTargetKind) -> LocalApplyTargetKind:
    return LocalApplyTargetKind(kind.value)


def _approval_proposal(
    *,
    agent: Agent,
    result: GepaOptimizationResult,
    mlflow_run_id: str,
    artifact_uri: str,
) -> tuple[ProposedAction | None, str]:
    """Build the exact local-apply approval only for a held-out winner."""
    improving, reason = candidate_improvement(result)
    if not improving:
        return None, reason
    evolved = result.holdout_evolved
    seed = result.holdout_seed_baseline
    if evolved is None or seed is None:  # candidate_improvement already checked
        return None, "held-out artifacts are missing"
    proof = ProofSummary.from_phase2_artifact(evolved)
    if not (proof.proved_improvement and proof.correctness_held):
        return None, "evolved held-out proof did not PROMOTE with correctness held"
    target = agent.optimization_target
    if target is None:  # narrowed before compute; defense in depth at persistence
        return None, "agent has no registered optimization target"
    reviewer_id = agent.reviewer_experiment_id
    if reviewer_id is None:
        return None, "agent has no reviewer experiment"
    diff = _review_diff(target.path, result.seed_skill_body, result.evolved_skill_body)
    local_spec = LocalApplySpec(
        target_kind=_local_target_kind(target.kind),
        target_path=target.path,
        artifact_uri=artifact_uri,
        artifact_path=RESULT_ARTIFACT_PATH,
        artifact_field="evolved_skill_body",
        baseline_sha256=_sha256(result.seed_skill_body),
        candidate_sha256=_sha256(result.evolved_skill_body),
        validation_command=target.validation_command,
        validation_timeout_seconds=target.validation_timeout_seconds,
        mlflow_run_id=mlflow_run_id,
        reviewer_experiment_id=reviewer_id,
        holdout_evolved_savings_pct=evolved.realized_token_savings_pct,
        holdout_seed_savings_pct=seed.realized_token_savings_pct,
        holdout_savings_delta_pct=result.holdout_savings_delta_pct,
        holdout_task_ids=list(result.holdout_task_ids),
    )
    change = ProposedChange(
        kind=ChangeKind.EVOLVED_BODY_REF,
        summary=(
            f"Rewrite {target.path} with the reviewed GEPA Optimize Anything winner; "
            f"held-out savings improved by {result.holdout_savings_delta_pct} pct-pts."
        ),
        diff=diff,
        evolved_body_ref=artifact_uri,
        local_apply_spec=local_spec,
    )
    proposal_id = derive_proposal_id(
        agent_name=agent.agent_name,
        action_kind=ActionKind.GEPA_PROMPT,
        change=change,
    )
    return (
        ProposedAction(
            proposal_id=proposal_id,
            agent_name=agent.agent_name,
            experiment_id=agent.experiment_id,
            action_kind=ActionKind.GEPA_PROMPT,
            risk_class=RiskClass.AGENT_CHANGE,
            objective_metric=proof.objective_metric,
            goal_cohort=agent.agent_name,
            trigger=TriggerSignal(
                kind=TriggerKind.AGENT_PLANNER,
                summary=(
                    "User explicitly dispatched GEPA Optimize Anything and the selected "
                    "candidate beat its seed on the held-out frozen-suite split."
                ),
                metric=proof.objective_metric,
                n_traces=len(result.holdout_task_ids),
            ),
            change=change,
            proof=proof,
            gate_status=GateStatus(
                readiness_tier="held_out_frozen_suite",
                can_prove_improvement=True,
                scored_coverage=1.0,
                gated=True,
                reasons=[reason, "local baseline hash + validation required after approval"],
            ),
            created_at=result.generated_at,
            notes=[
                f"optimizer={OPTIMIZER_NAME}",
                "Hosted compute cannot apply this path; local companion required.",
            ],
        ),
        reason,
    )


def _persist_approval_proposal(
    proposal: ProposedAction,
    *,
    warehouse_id: str,
    catalog: str,
    schema: str,
) -> None:
    insert_proposal_if_absent(
        proposal,
        client=_build_workspace_client(None),
        warehouse_id=warehouse_id,
        catalog=catalog,
        schema=schema,
        generated_at=proposal.created_at,
    )


def run(args: argparse.Namespace) -> int:
    _validate_request(args)
    auth_path = resolve_job_auth(
        token_secret_scope=args.token_secret_scope or None,
        token_secret_key=args.token_secret_key or None,
    )
    print(
        f"[ail.jobs.gepa_job] auth={auth_path} agent={args.agent} experiment={args.experiment_id}"
    )
    agent = _resolve_agent(args)
    reviewer_id = agent.reviewer_experiment_id
    assert reviewer_id is not None

    configure_monitoring_warehouse(reviewer_id, args.warehouse_id)
    _configure_claude_fmapi(args.agent_model)

    root = _assets_root()
    suite_path = root / "eval" / "task_suite" / args.suite_version / "tasks.yaml"
    fixtures_path = root / "eval" / "phase2_fixtures"
    if not suite_path.is_file():
        raise FileNotFoundError(f"packaged GEPA suite is missing: {suite_path}")
    if not fixtures_path.is_dir():
        raise FileNotFoundError(f"packaged GEPA fixtures are missing: {fixtures_path}")
    suite = load_task_suite(args.suite_version, root=root)
    verify_specs = _load_verify_specs(root)

    reviewer_name = _experiment_name(reviewer_id)
    adapter = ClaudeCodeAdapter(mlflow_experiment=reviewer_name)
    result = run_gepa_optimization(
        suite=suite,
        adapter=adapter,
        verify_specs=verify_specs,
        config=GepaConfig(
            reflection_lm=args.reflection_lm,
            max_metric_calls=args.max_metric_calls,
            holdout_fraction=args.holdout_fraction,
            max_train_tasks=args.max_train_tasks,
            seed=args.seed,
        ),
        fixtures_root=str(root),
    )
    mlflow_run_id, artifact_uri = _log_candidate(
        agent=agent, result=result, suite_version=args.suite_version
    )
    proposal, proposal_reason = _approval_proposal(
        agent=agent,
        result=result,
        mlflow_run_id=mlflow_run_id,
        artifact_uri=artifact_uri,
    )
    if proposal is not None:
        _persist_approval_proposal(
            proposal,
            warehouse_id=args.warehouse_id,
            catalog=args.catalog,
            schema=args.schema,
        )
    marker = _result_marker(
        agent=agent,
        result=result,
        mlflow_run_id=mlflow_run_id,
        artifact_uri=artifact_uri,
        proposal=proposal,
        proposal_reason=proposal_reason,
    )
    print(RESULT_MARKER + json.dumps(marker, separators=(",", ":")), flush=True)
    print(
        (
            "[ail.jobs.gepa_job] candidate complete; pending approval created, local companion "
            "required"
            if proposal is not None
            else "[ail.jobs.gepa_job] candidate complete; no improving approval proposal created"
        ),
        file=sys.stderr,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(_parse_args(argv))
    except Exception as exc:  # noqa: BLE001 - wheel task must exit non-zero with an honest reason
        print(f"[ail.jobs.gepa_job] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
