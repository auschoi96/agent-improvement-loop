"""Databricks Job entrypoint for arrival-triggered continuous RLM review."""

from __future__ import annotations

import argparse
import os

from ail.jobs.publish_job import resolve_job_auth
from ail.l3.continuous import run_continuous_rlm


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one bounded continuous RLM/HALO pass over new MLflow traces."
    )
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--warehouse-id", default=os.environ.get("AIL_WAREHOUSE_ID"))
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--max-results", type=int, default=100)
    parser.add_argument("--max-reviews", type=int, default=2)
    parser.add_argument("--sample-rate", type=float, default=0.10)
    parser.add_argument("--min-tokens", type=int, default=50_000)
    parser.add_argument("--reviewer-experiment", default="")
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument(
        "--token-secret-scope",
        default=os.environ.get("AIL_TOKEN_SECRET_SCOPE", ""),
        help="Secret scope holding the run-as bearer token. Empty => mint from run-as identity.",
    )
    parser.add_argument(
        "--token-secret-key",
        default=os.environ.get("AIL_TOKEN_SECRET_KEY", ""),
        help="Secret key within --token-secret-scope.",
    )
    args = parser.parse_args(argv)
    if not args.warehouse_id:
        parser.error("--warehouse-id is required (or set AIL_WAREHOUSE_ID)")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    auth_path = resolve_job_auth(
        token_secret_scope=args.token_secret_scope or None,
        token_secret_key=args.token_secret_key or None,
    )
    print(
        f"[ail.jobs.continuous_rlm] auth={auth_path} host={os.environ.get('DATABRICKS_HOST')} "
        f"experiment={args.experiment} sample_rate={args.sample_rate} "
        f"max_reviews={args.max_reviews}"
    )
    report = run_continuous_rlm(
        args.experiment,
        judge_model=args.judge_model,
        sql_warehouse_id=args.warehouse_id,
        max_results=args.max_results,
        max_reviews=args.max_reviews,
        sample_rate=args.sample_rate,
        min_tokens=args.min_tokens,
        reviewer_experiment_id=args.reviewer_experiment or None,
        max_turns=args.max_turns,
        temperature=args.temperature,
    )
    print(
        "[ail.jobs.continuous_rlm] "
        f"scanned={report.n_scanned} already_reviewed={report.n_already_reviewed} "
        f"reviewer_traces_skipped={report.n_reviewer_traces_skipped} "
        f"sampled_out={report.n_sampled_out} selected={report.n_selected} "
        f"reviewed={report.n_reviewed} failed={report.n_failed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
