"""``ail-author-judge`` — author a judge from a natural-language quality description.

A thin **driver** over :func:`ail.judges.authoring.author_judge`: it takes a
dimension name and a plain-language description on the command line and produces a
registered, MemAlign-alignable ``{{ trace }}`` judge plus the name-matched label
schema. It reuses, and never reimplements, the authoring module — the CLI only
parses flags, calls the one function, and prints what it created.

    ail-author-judge answer_helpfulness \\
        --description "Did the agent answer the user's question, completely and usefully?" \\
        --experiment-id 660599403165942 --profile dais-demo

By default it registers the judge as a scheduled scorer (needs the ``agents``
extra — ``pip install 'ail[agents]'``). Pass ``--no-register`` to only build the
judge and create the label schema (a preview that needs no ``databricks-agents``),
e.g. to review the authored rubric before scheduling scoring.

Fail-closed and honest: a bad name/scale or a missing backend/dependency exits
non-zero with an actionable message; it never prints a fabricated success.
"""

from __future__ import annotations

import argparse
import os
import sys

from ail.judges.authoring import DEFAULT_SCALE, AuthoredJudge, author_judge
from ail.judges.registration import DEFAULT_SAMPLING_RATE
from ail.publish import REFERENCE_EXPERIMENT


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ail-author-judge",
        description=(
            "Author a MemAlign-alignable {{ trace }} judge from a natural-language quality "
            "description, and create the label schema whose name matches the judge (so human "
            "labels align it). Reuses ail.judges.authoring; registers via the existing path."
        ),
    )
    parser.add_argument(
        "name",
        help="Quality dimension name (canonicalized to snake_case, e.g. 'answer_helpfulness').",
    )
    parser.add_argument(
        "--description",
        "-d",
        required=True,
        help="Natural-language description of what to judge (the criteria).",
    )
    parser.add_argument(
        "--experiment-id",
        default=REFERENCE_EXPERIMENT,
        help="MLflow experiment for the label schema + registration (default: reference exp).",
    )
    parser.add_argument(
        "--scale",
        choices=("1-5", "pass_fail"),
        default=DEFAULT_SCALE,
        help="Output shape: '1-5' graded (default) or 'pass_fail' categorical.",
    )
    parser.add_argument(
        "--profile",
        default=os.environ.get("DATABRICKS_CONFIG_PROFILE"),
        help="Databricks CLI profile selecting the workspace.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Judge model URI (e.g. 'databricks:/...'); omit for MLflow's default judge model.",
    )
    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=DEFAULT_SAMPLING_RATE,
        help=f"Fraction of traces the scheduled scorer scores (default {DEFAULT_SAMPLING_RATE}).",
    )
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="Build the judge + create the label schema only; do not register a scheduled scorer.",
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Run one optional LLM pass to sharpen vague criteria before templating.",
    )
    parser.add_argument(
        "--refine-endpoint",
        default=None,
        help="Databricks chat endpoint for --refine (else AIL_JUDGE_AUTHOR_LLM_ENDPOINT).",
    )
    parser.add_argument(
        "--overwrite-label-schema",
        action="store_true",
        help="Replace an existing label schema of this name instead of failing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        authored = author_judge(
            args.name,
            args.description,
            experiment_id=args.experiment_id,
            scale=args.scale,
            register=not args.no_register,
            model=args.model,
            sampling_rate=args.sampling_rate,
            refine=args.refine,
            refine_model=args.refine_endpoint,
            overwrite_label_schema=args.overwrite_label_schema,
            profile=args.profile,
        )
    except ValueError as exc:
        print(f"[ail-author-judge] invalid request: {exc}", file=sys.stderr)
        return 2
    except ImportError as exc:
        print(f"[ail-author-judge] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - surface any backend/auth failure, non-zero
        print(f"[ail-author-judge] failed to author judge: {exc}", file=sys.stderr)
        return 1

    print(_render(authored))
    return 0


def _render(authored: AuthoredJudge) -> str:
    """Human-readable summary of what was authored (judge, label schema, alignment)."""
    spec = authored.spec
    lines = [
        "=== ail-author-judge ===",
        f"  judge name        : {spec.name}",
        f"  label schema name : {authored.label_schema.name}   (matches judge name: "
        f"{authored.label_schema.name == spec.name})",
        f"  output scale      : {spec.feedback_value_type}",
        f"  {{{{ trace }}}} rubric  : {'yes' if '{{ trace }}' in spec.instructions else 'NO'}",
    ]
    if authored.registration is not None:
        reg = authored.registration
        lines.append(
            f"  registered scorer : {reg.scorer.name}  "
            f"(aligned={reg.aligned}; not-yet-trusted until aligned + audited)"
        )
    else:
        lines.append("  registered scorer : (skipped — --no-register)")
    lines.append(
        "  next: label ~30-50 traces under this schema in the MLflow UI, then align + audit "
        "(see docs/JUDGE_AUTHORING.md)."
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
