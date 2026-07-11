"""Minimal CLI for connecting a Python callable to AIL."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from ail.sdk import improve, load_callable


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ail", description="Connect any callable to AIL")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run a module:function and emit JSON")
    run.add_argument("--name", required=True, help="stable agent name")
    run.add_argument("--callable", required=True, help="Python module:function")
    run.add_argument("--prompt", required=True, help="task input")
    run.add_argument("--objective", help="improvement objective")
    run.add_argument("--experiment-id", help="MLflow experiment name or ID")
    run.add_argument("--tracking-uri", help="MLflow tracking URI")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        agent = improve(
            args.name,
            load_callable(args.callable),
            objective=args.objective,
            experiment_id=args.experiment_id,
            tracking_uri=args.tracking_uri,
        )
        result = agent.run(args.prompt)
        payload = _json_result(
            result.output,
            result.trace_id,
            result.duration_seconds,
            result.agent_name,
        )
        print(json.dumps(payload))
        return 0
    return 2


def _json_result(
    output: Any,
    trace_id: str | None,
    duration_seconds: float,
    agent_name: str,
) -> dict[str, Any]:
    try:
        json.dumps(output)
        serialized = output
    except TypeError:
        serialized = str(output)
    return {
        "agent_name": agent_name,
        "output": serialized,
        "trace_id": trace_id,
        "duration_seconds": round(duration_seconds, 6),
    }


if __name__ == "__main__":
    sys.exit(main())
