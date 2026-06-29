"""Single entrypoint: compute the L0 baseline and reproduce Example 1.

Run as ``python -m ail.metrics.report`` (see ``main``). It pulls traces from
the reference MLflow experiment via the Wave 0 :class:`MLflowTraceSource`,
computes the L0 report (:func:`ail.metrics.l0_deterministic.compute_l0`), and
writes three artifacts:

* ``l0_baseline_<exp>.json`` — the full L0 contract (what a UI would read).
* ``example1_diagnosis.json`` — the machine-readable Example 1 diagnosis.
* ``example1_diagnosis.md`` — the human-readable Example 1 diagnosis.

"Example 1" is the token-waste scenario from ``docs/ARCHITECTURE.md`` §8: a
bimodal token distribution with a few very large sessions, repeated re-reads of
the same file, and re-run shell-setup boilerplate. :func:`build_example1_diagnosis`
turns an :class:`~ail.metrics.contract.L0MetricsReport` into that diagnosis and
is a **pure function** (no I/O), so it is unit-tested offline against synthetic
traces; only :func:`main` touches the network.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ail.ingest.mlflow_source import MLflowTraceSource
from ail.metrics.contract import L0MetricsReport, RepeatedCall, TraceMetrics
from ail.metrics.l0_deterministic import compute_l0

REFERENCE_EXPERIMENT = "660599403165942"
HIGH_TOKEN_THRESHOLD = 500_000

# Relative tolerance for calling a live value a "match" of a documented one.
TOKEN_TOLERANCE = 0.15

# The 77-trace snapshot recorded in docs/ARCHITECTURE.md §8. The live corpus is
# explicitly "a live, growing corpus, not a stable historical baseline", so the
# diagnosis reconciles the live numbers against these and flags any drift rather
# than asserting they still hold.
DOC_SNAPSHOT_TRACES = 77
DOC_HIGH_TOKEN_SESSIONS: list[int] = [549_000, 943_000]
DOC_MEDIAN_TOKENS = 18_500
DOC_MAX_READ_SAME_PATH = 34
DOC_SHELL_BOILERPLATE_RANGE: tuple[int, int] = (13, 21)


# ---------------------------------------------------------------------------
# Diagnosis (pure: report -> markdown + machine-readable payload)
# ---------------------------------------------------------------------------


def _value_matches(live_values: list[float], expected: float, tol: float = TOKEN_TOLERANCE) -> bool:
    """Whether any live value is within ``tol`` (relative) of ``expected``."""
    return any(abs(v - expected) <= tol * expected for v in live_values)


def _max_repeat(trace: TraceMetrics, kind: str, tool: str | None = None) -> RepeatedCall | None:
    """The most-repeated identity of a given signature kind (optionally tool)."""
    candidates = [
        r
        for r in trace.redundancy.repeated_calls
        if r.signature_kind == kind and (tool is None or r.tool == tool)
    ]
    return max(candidates, key=lambda r: r.count) if candidates else None


def _rank_by_repeat(
    report: L0MetricsReport, kind: str, tool: str | None = None, limit: int = 8
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for m in report.traces:
        top = _max_repeat(m, kind, tool)
        if top is not None:
            rows.append(
                {
                    "trace_id": m.trace_id,
                    "tool": top.tool,
                    "count": top.count,
                    "identity": top.identity,
                    "total_tool_calls": m.total_tool_calls,
                }
            )
    rows.sort(key=lambda r: int(r["count"]), reverse=True)
    return rows[:limit]


def build_example1_diagnosis(report: L0MetricsReport) -> tuple[str, dict[str, Any]]:
    """Turn an L0 report into the Example 1 diagnosis (markdown, payload).

    Pure function — derives everything from ``report``. Returns the markdown
    document and a machine-readable payload with the same numbers.
    """
    agg = report.aggregate
    high = [
        {
            "trace_id": m.trace_id,
            "session_id": m.session_id,
            "model": m.model,
            "total_tokens": m.tokens.total_tokens,
            "input_tokens": m.tokens.input_tokens,
            "output_tokens": m.tokens.output_tokens,
            "total_tool_calls": m.total_tool_calls,
            "duration_seconds": m.duration_seconds,
            "cost_usd": m.cost.total_usd,
            "cost_priced": m.cost.priced,
        }
        for m in report.traces
        if m.tokens.total_tokens >= HIGH_TOKEN_THRESHOLD
    ]

    shell_rows = _rank_by_repeat(report, "shell")
    path_rows = _rank_by_repeat(report, "path")
    read_rows = _rank_by_repeat(report, "path", tool="Read")

    max_read = read_rows[0]["count"] if read_rows else 0
    max_path = path_rows[0]["count"] if path_rows else 0
    max_shell = shell_rows[0]["count"] if shell_rows else 0

    # Statuses are DERIVED from the live numbers against documented expectations
    # with explicit tolerances, so a rerun on a rotated/changed corpus reports
    # drift instead of overstating reproduction.
    live_high = sorted(
        (
            m.tokens.total_tokens
            for m in report.traces
            if m.tokens.total_tokens >= HIGH_TOKEN_THRESHOLD
        ),
        reverse=True,
    )
    high_matched = len(live_high) >= len(DOC_HIGH_TOKEN_SESSIONS) and all(
        _value_matches([float(v) for v in live_high], float(e)) for e in DOC_HIGH_TOKEN_SESSIONS
    )
    high_status = (
        "match"
        if high_matched
        else f"drift: documented {DOC_HIGH_TOKEN_SESSIONS} not all present in live {live_high[:3]}"
    )

    median = agg.token_stats.median
    median_matched = _value_matches([median], float(DOC_MEDIAN_TOKENS))
    median_status = (
        "match"
        if median_matched
        else f"drift: live median {median} vs documented ~{DOC_MEDIAN_TOKENS}"
    )

    shell_lo, shell_hi = DOC_SHELL_BOILERPLATE_RANGE
    shell_reproduced = max_shell >= shell_lo
    shell_status = (
        f"reproduced (max {max_shell}×/trace ≥ documented floor {shell_lo}×)"
        if shell_reproduced
        else f"not reproduced (max {max_shell}× below documented {shell_lo}–{shell_hi}×)"
    )

    read_floor = DOC_MAX_READ_SAME_PATH * (1 - TOKEN_TOLERANCE)
    read_reproduced = max_read >= read_floor
    if read_reproduced:
        read_status = (
            f"reproduced (strongest live re-read {max_read}× ≥ ~{DOC_MAX_READ_SAME_PATH}×)"
        )
    else:
        read_status = (
            f"NOT reproduced — strongest live re-read of one path is {max_read}× "
            f"(any file tool {max_path}×) vs documented {DOC_MAX_READ_SAME_PATH}×; corpus grew "
            f"{DOC_SNAPSHOT_TRACES}->{report.n_traces} and rotates, so the "
            f"{DOC_MAX_READ_SAME_PATH}× trace is no longer present"
        )

    reconciliation = {
        "high_token_sessions": {
            "live": live_high,
            "doc": DOC_HIGH_TOKEN_SESSIONS,
            "tolerance": TOKEN_TOLERANCE,
            "status": high_status,
        },
        "median_tokens": {
            "live": median,
            "doc": DOC_MEDIAN_TOKENS,
            "tolerance": TOKEN_TOLERANCE,
            "status": median_status,
        },
        "shell_boilerplate": {
            "live_max_per_trace": max_shell,
            "doc_range": list(DOC_SHELL_BOILERPLATE_RANGE),
            "reproduced": shell_reproduced,
            "status": shell_status,
        },
        "read_same_path": {
            "live_max_read_same_path": max_read,
            "live_max_same_path_any_file_tool": max_path,
            "doc": DOC_MAX_READ_SAME_PATH,
            "tolerance": TOKEN_TOLERANCE,
            "reproduced": read_reproduced,
            "status": read_status,
        },
    }

    payload: dict[str, Any] = {
        "schema_version": report.schema_version,
        "experiment_id": report.experiment_id,
        "generated_at": report.generated_at,
        "scenario": "Example 1 — token waste (ARCHITECTURE.md §8)",
        "corpus": {
            "n_traces": report.n_traces,
            "status_counts": agg.status_counts,
            "by_model": [
                {"model": g.key, "n_traces": g.n_traces, "total_tokens": g.tokens.total_tokens}
                for g in report.by_model
            ],
            "by_producer": [
                {"producer": g.key, "n_traces": g.n_traces} for g in report.by_producer
            ],
        },
        "tokens": {
            "total": agg.tokens.total_tokens,
            "median": agg.token_stats.median,
            "mean": agg.token_stats.mean,
            "p90": agg.token_stats.p90,
            "max": agg.token_stats.max,
            "min": agg.token_stats.min,
        },
        "cost": {
            "total_usd": agg.cost.total_usd,
            "priced_traces": agg.cost.priced_traces,
            "unpriced_traces": agg.cost.unpriced_traces,
            "flags": report.pricing_flags,
        },
        "high_token_sessions": high,
        "tool_redundancy": {
            "strict_redundancy_rate": agg.redundancy.redundancy_rate,
            "redundant_tool_calls": agg.redundancy.redundant_tool_calls,
            "total_tool_calls": agg.redundancy.total_tool_calls,
            "shell_boilerplate_top": shell_rows,
            "repeated_file_reads_top": read_rows,
            "repeated_file_access_top": path_rows,
        },
        "reconciliation_with_doc": reconciliation,
    }

    md = _render_markdown(report, payload)
    return md, payload


def _fmt(n: float | int | None) -> str:
    if n is None:
        return "—"
    if isinstance(n, float) and n.is_integer():
        n = int(n)
    return f"{n:,}" if isinstance(n, int) else f"{n:,.1f}"


def _tid(trace_id: str) -> str:
    return trace_id.rsplit("/", 1)[-1]


def _render_markdown(report: L0MetricsReport, p: dict[str, Any]) -> str:
    lines: list[str] = []
    a = lines.append
    a("# Example 1 — Token-Waste Diagnosis (L0 baseline)")
    a("")
    a(
        f"Deterministic L0 metrics over experiment `{report.experiment_id}` "
        f"({report.n_traces} traces), generated `{report.generated_at}`. "
        "Reproduces the token-waste scenario in `docs/ARCHITECTURE.md` §8. Every "
        "number below is mechanical (token counts, timestamps, tool spans) — no model in the loop."
    )
    a("")
    a("## Corpus")
    a("")
    a(f"- **Traces:** {report.n_traces}")
    a(f"- **Status:** {p['corpus']['status_counts']}")
    a(
        "- **Producers:** "
        + ", ".join(f"{g['producer']}={g['n_traces']}" for g in p["corpus"]["by_producer"])
    )
    a(
        "- **Models:** "
        + ", ".join(f"{g['model']}={g['n_traces']}" for g in p["corpus"]["by_model"])
    )
    a("")
    a("## Token distribution (bimodal)")
    a("")
    t = p["tokens"]
    a(f"- **Total tokens:** {_fmt(t['total'])}")
    a(
        f"- **Median:** {_fmt(t['median'])} · **Mean:** {_fmt(t['mean'])} · "
        f"**p90:** {_fmt(t['p90'])} · **Max:** {_fmt(t['max'])}"
    )
    a(
        "- A low median with a heavy tail: most sessions are small, a few are enormous. "
        "That tail is where the token spend lives."
    )
    a("")
    a("## High-token sessions (Example 1)")
    a("")
    a(f"Sessions at or above {_fmt(HIGH_TOKEN_THRESHOLD)} total tokens:")
    a("")
    a("| trace | model | total tokens | input | output | tools | duration (s) | est. cost |")
    a("|---|---|---|---|---|---|---|---|")
    for h in p["high_token_sessions"]:
        cost = f"${h['cost_usd']:,.2f}" if h["cost_priced"] else "unpriced"
        a(
            f"| `{_tid(h['trace_id'])}` | {h['model']} | {_fmt(h['total_tokens'])} | "
            f"{_fmt(h['input_tokens'])} | {_fmt(h['output_tokens'])} | {h['total_tool_calls']} | "
            f"{_fmt(h['duration_seconds'])} | {cost} |"
        )
    a("")
    a("## Tool-call redundancy")
    a("")
    r = p["tool_redundancy"]
    a(
        f"- **Strict redundancy rate** (byte-identical repeated calls): "
        f"**{r['strict_redundancy_rate']:.3f}** "
        f"({r['redundant_tool_calls']} of {r['total_tool_calls']} calls). Low — exact-duplicate "
        "calls are rare; the waste is in repeated *targets*, below."
    )
    a("")
    a("### Re-run shell-setup boilerplate (per trace)")
    a("")
    a(
        "Recurrences of the same normalized shell prologue (`cd <dir>`/env setup, "
        "per-session scratch UUIDs collapsed) — the agent re-establishing the same "
        "working directory on call after call:"
    )
    a("")
    a("| trace | repeats | prologue |")
    a("|---|---|---|")
    for row in r["shell_boilerplate_top"]:
        a(f"| `{_tid(row['trace_id'])}` | {row['count']}× | `{row['identity'][:80]}` |")
    a("")
    a("### Repeated file access (per trace)")
    a("")
    a("Same file path targeted repeatedly by the same tool:")
    a("")
    a("| trace | tool | repeats | path |")
    a("|---|---|---|---|")
    for row in r["repeated_file_access_top"]:
        tid, path = _tid(row["trace_id"]), _tid(row["identity"])
        a(f"| `{tid}` | {row['tool']} | {row['count']}× | `{path}` |")
    a("")
    a("## Estimated cost")
    a("")
    c = p["cost"]
    a(
        f"- **Total (priced traces):** ${c['total_usd']:,.2f} across {c['priced_traces']} priced "
        f"trace(s); {c['unpriced_traces']} unpriced."
    )
    a("- **Pricing caveats:**")
    for flag in c["flags"]:
        a(f"  - {flag}")
    a("")
    a("## Reconciliation with `docs/ARCHITECTURE.md` §8 (77-trace snapshot)")
    a("")
    rec = p["reconciliation_with_doc"]
    a("| signal | live | doc snapshot | verdict |")
    a("|---|---|---|---|")
    hts = rec["high_token_sessions"]
    a(
        f"| high-token sessions | {[_fmt(x) for x in hts['live']]} | "
        f"~{hts['doc']} | {hts['status']} |"
    )
    mt = rec["median_tokens"]
    a(f"| median tokens | {_fmt(mt['live'])} | ~{_fmt(mt['doc'])} | {mt['status']} |")
    sb = rec["shell_boilerplate"]
    a(
        f"| shell boilerplate re-runs | up to {sb['live_max_per_trace']}×/trace | "
        f"{sb['doc_range'][0]}–{sb['doc_range'][1]}× | {sb['status']} |"
    )
    rp = rec["read_same_path"]
    a(
        f"| re-read same path | {rp['live_max_read_same_path']}× (Read), "
        f"{rp['live_max_same_path_any_file_tool']}× (any file tool) | {rp['doc']}× | "
        f"{rp['status']} |"
    )
    a("")
    if not rp["reproduced"]:
        doc_read = rp["doc"]
        by_tool = rp["live_max_read_same_path"]
        any_tool = rp["live_max_same_path_any_file_tool"]
        a(
            f"> **Flagged:** the documented *{doc_read}× re-read of the same path* is **not** "
            f"present in the live {report.n_traces}-trace corpus (strongest current re-read "
            f"{by_tool}× by one tool, {any_tool}× across file tools). The corpus is explicitly "
            f"live and growing (the doc snapshot was {DOC_SNAPSHOT_TRACES} traces); that "
            "high-redundancy trace has rotated out. The token-waste shape (bimodal distribution, "
            f"huge tail sessions, re-run shell boilerplate) reproduces; the specific {doc_read}× "
            "figure does not, and is not asserted."
        )
    else:
        a(
            f"> The documented token-waste signals reproduce on the live {report.n_traces}-trace "
            f"corpus, including a re-read of the same path up to {rp['live_max_read_same_path']}× "
            f"(documented ~{rp['doc']}×)."
        )
    a("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Live pull + write (main)
# ---------------------------------------------------------------------------


def _build_source(profile: str | None) -> MLflowTraceSource:
    """Construct an MLflow trace source, preferring explicit env-token auth.

    If ``DATABRICKS_HOST`` and ``DATABRICKS_TOKEN`` are in the environment, use
    them via a bare ``databricks`` tracking URI: this is the auth path that
    works with the MLflow 3 v4 trace REST store on this workspace, where the
    experiment is backed by a UC table and OAuth-profile credentials are
    rejected by ``batchGet``. Otherwise resolve a CLI profile, carrying it in
    the tracking URI so OAuth credentials are picked up for all endpoints.
    """
    if os.environ.get("DATABRICKS_HOST") and os.environ.get("DATABRICKS_TOKEN"):
        # Explicit creds win: drop a conflicting ambient profile so MLflow's
        # per-request credential resolution can't fall back to OAuth (which the
        # v4 trace store rejects) for some spans and the env token for others.
        os.environ.pop("DATABRICKS_CONFIG_PROFILE", None)
        return MLflowTraceSource(tracking_uri="databricks")
    if profile:
        return MLflowTraceSource(tracking_uri=f"databricks://{profile}", profile=profile)
    return MLflowTraceSource()


def generate(
    *,
    experiment_id: str,
    profile: str | None,
    out_dir: Path,
    max_results: int | None,
    generated_at: str | None = None,
) -> L0MetricsReport:
    """Pull traces, compute L0, and write the three artifacts. Returns the report."""
    source = _build_source(profile)
    traces = source.fetch_traces(experiment_id=experiment_id, max_results=max_results)
    stamp = generated_at or datetime.now(UTC).isoformat()
    report = compute_l0(traces, experiment_id=experiment_id, generated_at=stamp)

    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = out_dir / f"l0_baseline_{experiment_id}.json"
    baseline_path.write_text(report.model_dump_json(indent=2) + "\n")

    md, payload = build_example1_diagnosis(report)
    (out_dir / "example1_diagnosis.json").write_text(json.dumps(payload, indent=2) + "\n")
    (out_dir / "example1_diagnosis.md").write_text(md + "\n")

    print(f"wrote {baseline_path}")
    print(f"wrote {out_dir / 'example1_diagnosis.json'}")
    print(f"wrote {out_dir / 'example1_diagnosis.md'}")
    print(
        f"n_traces={report.n_traces} total_tokens={report.aggregate.tokens.total_tokens:,} "
        f"total_cost=${report.aggregate.cost.total_usd:,.2f}"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute the L0 baseline and reproduce Example 1.")
    parser.add_argument("--experiment", default=REFERENCE_EXPERIMENT)
    parser.add_argument(
        "--profile",
        default=os.environ.get("AIL_DATABRICKS_PROFILE", "dais-demo"),
        help="Databricks CLI profile (ignored if DATABRICKS_HOST/DATABRICKS_TOKEN are set).",
    )
    parser.add_argument("--out-dir", default="artifacts", type=Path)
    parser.add_argument("--max-results", default=None, type=int)
    args = parser.parse_args(argv)

    generate(
        experiment_id=args.experiment,
        profile=args.profile,
        out_dir=args.out_dir,
        max_results=args.max_results,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
