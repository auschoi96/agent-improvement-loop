"""Fail-closed live validation that every deployed AIL job runs as the production SP."""

from __future__ import annotations

import argparse
import sys
from typing import Any


def run_as_mismatches(jobs: list[Any], expected_sp: str) -> list[str]:
    mismatches: list[str] = []
    managed = 0
    for job in jobs:
        settings = getattr(job, "settings", None)
        tags = getattr(settings, "tags", None) or {}
        if tags.get("project") != "agent-improvement-loop":
            continue
        managed += 1
        name = str(getattr(settings, "name", None) or getattr(job, "job_id", "unknown"))
        run_as = getattr(settings, "run_as", None)
        actual = str(getattr(run_as, "service_principal_name", None) or "")
        if actual != expected_sp:
            effective_user = getattr(run_as, "user_name", None) or getattr(
                job, "creator_user_name", None
            )
            rendered = actual or f"user:{effective_user or 'unset'}"
            mismatches.append(f"{name}: expected service principal {expected_sp}, found {rendered}")
    if managed == 0:
        mismatches.append("no managed agent-improvement-loop jobs were visible")
    return mismatches


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-sp", required=True)
    parser.add_argument("--profile", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from databricks.sdk import WorkspaceClient

    args = _parse_args(argv)
    client = WorkspaceClient(profile=args.profile) if args.profile else WorkspaceClient()
    mismatches = run_as_mismatches(list(client.jobs.list(expand_tasks=False)), args.expected_sp)
    if mismatches:
        for mismatch in mismatches:
            print(f"[ail.validate_run_as] {mismatch}", file=sys.stderr)
        return 1
    print(f"[ail.validate_run_as] all managed jobs run as {args.expected_sp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
