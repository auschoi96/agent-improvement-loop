"""``ail-register-scorers`` -- register the built-in judges on an experiment.

A thin **driver** over :func:`ail.judges.registration.register_scorers`: it parses
the target experiment and backend knobs, delegates registration to the existing
module, and prints the registered scorer names. The CLI intentionally has no
workspace or experiment default; callers must name the experiment they intend to
mutate.
"""

from __future__ import annotations

import argparse
import sys

from ail.judges import registration
from ail.judges.registration import ScorerRegistration, register_scorers


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ail-register-scorers",
        description=(
            "Register the built-in scheduled scorers on an MLflow experiment. "
            "Reuses ail.judges.registration.register_scorers."
        ),
    )
    parser.add_argument(
        "--experiment-id",
        required=True,
        help="MLflow experiment id to register scorers against.",
    )
    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=registration.DEFAULT_SAMPLING_RATE,
        help=(
            "Fraction of traces the scheduled scorers score "
            f"(default {registration.DEFAULT_SAMPLING_RATE})."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Judge model URI; omit for MLflow's default judge model.",
    )
    parser.add_argument(
        "--filter-string",
        default=None,
        help="Optional search_traces filter limiting scored traces.",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Databricks CLI profile selecting the workspace.",
    )
    parser.add_argument(
        "--tracking-uri",
        default="databricks",
        help="MLflow tracking URI (default: databricks).",
    )
    parser.add_argument(
        "--registry-uri",
        default="databricks-uc",
        help="MLflow registry URI (default: databricks-uc).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        registrations = register_scorers(
            args.experiment_id,
            sampling_rate=args.sampling_rate,
            model=args.model,
            filter_string=args.filter_string,
            profile=args.profile,
            tracking_uri=args.tracking_uri,
            registry_uri=args.registry_uri,
        )
    except ValueError as exc:
        print(f"[ail-register-scorers] invalid request: {exc}", file=sys.stderr)
        return 2
    except ImportError as exc:
        print(f"[ail-register-scorers] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - surface backend/auth failures, non-zero
        print(f"[ail-register-scorers] failed to register scorers: {exc}", file=sys.stderr)
        return 1

    print(_render(registrations))
    return 0


def _render(registrations: list[ScorerRegistration]) -> str:
    names = [_registration_name(registration) for registration in registrations]
    lines = [
        "=== ail-register-scorers ===",
        f"  registered scorer count : {len(names)}",
        f"  registered scorers      : {', '.join(names) if names else '(none)'}",
    ]
    return "\n".join(lines)


def _registration_name(registration: ScorerRegistration) -> str:
    scorer = registration.scorer
    return str(getattr(scorer, "name", scorer))


if __name__ == "__main__":
    raise SystemExit(main())
