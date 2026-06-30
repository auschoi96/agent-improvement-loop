#!/usr/bin/env python3
"""MemAlign **manipulate + rollback** showcase: align a judge, overfit it, roll it back.

This is the operational counterpart to the unit-tested L2 judge layer
(:mod:`ail.judges`). It exercises the *real* MLflow MemAlign path end to end on a
live Databricks workspace and proves a single mechanic:

    adding human-feedback memory to a judge moves its agreement with held-out
    humans, and **retracting** that memory (``unalign``) moves it back.

Four measurements on one frozen, held-out Human Anchor:

* **BASE** — the unaligned ``{{ trace }}`` token-efficiency judge.
* **ALIGNED** — BASE after ``align`` on a genuine, L0-derived alignment set.
* **OVERFIT** — ALIGNED after a *second* ``align`` on a deliberately **biased**
  (label-inverted) subset, which drags held-out agreement DOWN.
* **ROLLED-BACK** — OVERFIT after ``unalign(traces=...)`` retracts exactly the
  biased traces, so agreement RECOVERS.

The unalign API (discovered in ``mlflow.genai.judges.optimizers.memalign``):
``judge.align(...)`` returns a ``MemoryAugmentedJudge``; that judge exposes
``unalign(traces: list[Trace]) -> MemoryAugmentedJudge``, which drops every
episodic example and every guideline whose source traces are all in the given
set. We call it directly on the OVERFIT judge with the biased traces.

It is **OPERATIONAL**: it makes live model + embedding calls (judge scoring,
MemAlign reflection, embeddings) and live trace reads. It must be run **by hand**,
never in CI, and self-guards: it does nothing unless ``AIL_LIVE_MLFLOW=1`` is set,
a Databricks profile + experiment id are supplied, and the optional ``align``
extra (``dspy``) is installed.

Example
-------
    AIL_LIVE_MLFLOW=1 python scripts/demo_memalign_rollback.py \
        --experiment-id 660599403165942 \
        --profile dais-demo \
        --token-cap 50000 \
        --max-traces 120

Honest caveat (read ``docs/MEMALIGN_ROLLBACK.md`` §Caveat): ``token_efficiency``'s
most informative (wasteful, low-score) examples are *huge* traces that do not fit
a ``{{ trace }}`` judge's context, so capping traces at ``--token-cap`` leaves a
small-trace anchor that skews toward high (efficient) labels. The demo therefore
proves the **mechanics** — memory add/retract moves agreement — not a calibrated
agreement *magnitude*. A dimension whose discriminating examples are small (e.g. a
focused correctness or tool-selection judge) would sharpen the numbers.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

from ail.ingest.mlflow_source import MLflowTraceSource
from ail.judges import (
    AgreementConfig,
    MemAlignConfig,
    ScorerSpec,
    TraceLabel,
    align_judge,
    assert_pools_disjoint,
    build_memalign_optimizer,
    make_scorer,
    score_anchor,
    split_labels,
    to_alignment_set,
    to_human_anchor,
)
from ail.metrics import compute_trace_metrics

if TYPE_CHECKING:
    from ail.ingest.base import NormalizedTrace
    from ail.judges.contract import AgreementReport
    from ail.metrics import TraceMetrics

# Models for the MemAlign optimizer, per the demo brief. The judge model (what
# actually scores a trace) is separate and configurable; it defaults to the same
# Claude Sonnet used for reflection so the showcase reads from one model family.
DEFAULT_REFLECTION_LM = "databricks:/databricks-claude-sonnet-4-6"
DEFAULT_EMBEDDING_MODEL = "databricks:/databricks-gte-large-en"
DEFAULT_EMBEDDING_DIM = 1024
DEFAULT_JUDGE_MODEL = "databricks:/databricks-claude-sonnet-4-6"
DEFAULT_TOKEN_CAP = 50_000


# ---------------------------------------------------------------------------
# The {{ trace }}-based token-efficiency judge (built via make_judge).
# ---------------------------------------------------------------------------
#
# Unlike ``ail.judges.scorers.TOKEN_EFFICIENCY`` (which is deliberately fed an L0
# summary, never the raw trace), this demo judge reads ``{{ trace }}`` directly:
# it is the variant whose alignment + agreement we are showcasing, and the cap
# keeps each trace inside the judge's context window.
TOKEN_EFFICIENCY_TRACE = ScorerSpec(
    name="token_efficiency",
    description="Was the token spend justified for the task (1=wasteful, 5=tightly efficient)?",
    feedback_value_type=Literal[1, 2, 3, 4, 5],
    aggregations=("mean", "median", "p90"),
    instructions=(
        "You are rating the TOKEN EFFICIENCY of an agent run on a 1-to-5 scale.\n\n"
        "Inspect the run using the trace:\n{{ trace }}\n\n"
        "Use the tools to read the trace's spans, tool calls, and token usage. Judge "
        "whether the token spend was JUSTIFIED by what the task required, whether any "
        "redundancy (re-reading the same file, re-running identical setup) was AVOIDABLE, "
        "and whether quality-per-token was good. Efficiency is conditioned on SUCCESS: "
        "spending few tokens by doing less or stopping early is NOT efficient.\n\n"
        "Scoring guide:\n"
        "  1 - large avoidable waste, or tokens burned without accomplishing the task\n"
        "  2 - clear avoidable waste with some useful work\n"
        "  3 - acceptable; spend roughly fits the task\n"
        "  4 - efficient; little avoidable redundancy\n"
        "  5 - tightly efficient; spend well justified with no meaningful waste\n\n"
        "Return the single integer (1-5) that best fits and briefly justify it, naming "
        "the specific waste you saw (or saying there was none)."
    ),
)


@dataclass(frozen=True, slots=True)
class DemoConfig:
    experiment_id: str
    profile: str | None
    token_cap: int
    max_traces: int
    anchor_fraction: float
    bias_fraction: float
    seed: int
    labeler_id: str
    judge_model: str
    memalign: MemAlignConfig


# ---------------------------------------------------------------------------
# A deterministic stand-in for the human grader (documented in the caveat).
# ---------------------------------------------------------------------------


def human_label_for(metrics: TraceMetrics) -> float:
    """A reproducible token-efficiency grade (1-5) derived from L0 signals.

    This stands in for a domain expert: it grades on the *deterministic* waste
    signal (byte-identical repeated tool calls), which is exactly what a human
    reviewer would point at. It is intentionally simple and reproducible so the
    demo is repeatable; it is NOT a calibrated ground truth (see the caveat).

    Returns a **float** grade on purpose: agreement on a graded scale treats a
    judge score within ``numeric_tolerance`` of the human grade as agreement
    (``ail.judges.agreement`` applies the tolerance only to float labels — int
    labels require an exact match), so a float gold with a tolerance of 1 means
    "within one grade counts as agreement".
    """
    rate = metrics.redundancy.redundancy_rate
    if rate >= 0.30:
        return 1.0
    if rate >= 0.15:
        return 2.0
    if rate >= 0.07:
        return 3.0
    if rate > 0.0:
        return 4.0
    return 5.0


def bias_label(value: float) -> float:
    """Invert a genuine grade into a deliberately conflicting one (good<->bad).

    Genuinely efficient runs (>=3) are labeled maximally wasteful (1) and
    genuinely wasteful runs (<3) maximally tight (5). Distilled into guidelines,
    this teaches the judge the *opposite* of the real pattern, so held-out
    agreement drops — the manipulation the rollback then reverses.
    """
    return 1.0 if value >= 3 else 5.0


# ---------------------------------------------------------------------------
# Pipeline steps.
# ---------------------------------------------------------------------------


def collect_labeled_traces(source: MLflowTraceSource, cfg: DemoConfig) -> list[TraceLabel]:
    """Read traces, keep those under the token cap, and L0-grade each one.

    The cap is what makes the corpus fit a ``{{ trace }}`` judge; it is also the
    source of the documented label skew (the wasteful low-score runs are the big
    ones, which the cap drops).
    """
    traces: list[NormalizedTrace] = source.fetch_traces(
        experiment_id=cfg.experiment_id,
        max_results=cfg.max_traces,
        order_by=["timestamp_ms DESC"],
    )
    labels: list[TraceLabel] = []
    skipped_oversize = 0
    for trace in traces:
        metrics = compute_trace_metrics(trace)
        if metrics.tokens.total_tokens > cfg.token_cap:
            skipped_oversize += 1
            continue
        labels.append(
            TraceLabel(
                trace_id=trace.trace_id,
                name="token_efficiency",
                value=human_label_for(metrics),
                rationale=(
                    f"L0 redundancy_rate={metrics.redundancy.redundancy_rate:.2f}, "
                    f"{metrics.redundancy.redundant_tool_calls} redundant of "
                    f"{metrics.total_tool_calls} tool calls"
                ),
            )
        )
    print(
        f"  fetched {len(traces)} traces; kept {len(labels)} under the "
        f"{cfg.token_cap:,}-token cap ({skipped_oversize} dropped as oversize)"
    )
    return labels


def _agreement(judge: Any, anchor: Any) -> AgreementReport:
    """Score a judge on the held-out anchor (tolerance 1 on the 1-5 graded scale)."""
    return score_anchor(judge, anchor, config=AgreementConfig(numeric_tolerance=1.0, floor=0.6))


def _print_row(stage: str, report: AgreementReport) -> None:
    flag = "DISTRUSTED" if report.distrusted else "trusted"
    print(
        f"  {stage:<12} agreement_rate={report.agreement_rate:.3f}  "
        f"(scored {report.n_scored}/{report.n_items}, {flag})"
    )


def run_demo(cfg: DemoConfig) -> int:
    source = MLflowTraceSource(profile=cfg.profile)

    print("[1/6] Collecting and L0-grading traces under the token cap...")
    labels = collect_labeled_traces(source, cfg)
    if len(labels) < 6:
        print(
            f"ERROR: only {len(labels)} labeled traces under the cap; need >=6 to form a "
            "good/biased/anchor split. Raise --token-cap or --max-traces.",
            file=sys.stderr,
        )
        return 1

    print("[2/6] Splitting into disjoint pools (anchor held out; alignment -> good + biased)...")
    align_labels, anchor_labels = split_labels(
        labels, anchor_fraction=cfg.anchor_fraction, seed=cfg.seed
    )
    # Sub-split the alignment pool (trace-level, disjoint) into the genuine subset
    # and the subset we will deliberately bias and then retract.
    good_labels, bias_src_labels = split_labels(
        align_labels, anchor_fraction=cfg.bias_fraction, seed=cfg.seed + 1
    )
    biased_labels = [replace(lab, value=bias_label(float(lab.value))) for lab in bias_src_labels]
    if not good_labels or not biased_labels or not anchor_labels:
        print(
            "ERROR: split produced an empty pool; adjust --anchor-fraction / --bias-fraction "
            "or supply more traces.",
            file=sys.stderr,
        )
        return 1

    good_set = to_alignment_set(source, good_labels, labeler_id=cfg.labeler_id)
    biased_set = to_alignment_set(source, biased_labels, labeler_id=cfg.labeler_id)
    anchor = to_human_anchor(anchor_labels, name="token_efficiency", source=source)
    # Prove the frozen wall across all three pools (anchor vs good vs biased).
    assert_pools_disjoint(
        alignment_set=good_set, human_anchor=anchor, task_suite_ids=biased_set.ids
    )
    print(
        f"  good={len(good_set)} traces, biased={len(biased_set)} traces, "
        f"anchor={len(anchor)} held-out items (disjoint wall proven)"
    )

    print("[3/6] Building the {{ trace }} token-efficiency judge (make_judge)...")
    base_judge = make_scorer(TOKEN_EFFICIENCY_TRACE, model=cfg.judge_model)
    optimizer = build_memalign_optimizer(cfg.memalign)

    print("[4/6] Measuring BASE, then aligning on the genuine set...")
    base_report = _agreement(base_judge, anchor)
    aligned = align_judge(base_judge, good_set, optimizer=optimizer).judge
    aligned_report = _agreement(aligned, anchor)

    print("[5/6] OVERFITTING on the biased (label-inverted) subset...")
    overfit = align_judge(aligned, biased_set, optimizer=optimizer).judge
    overfit_report = _agreement(overfit, anchor)

    print("[6/6] ROLLING BACK: unalign(traces=biased) retracts the biased memory...")
    # The discovered MLflow API: a MemoryAugmentedJudge retracts memory by trace.
    rolled_back = overfit.unalign(traces=list(biased_set.traces))
    rolled_back_report = _agreement(rolled_back, anchor)

    print("\n=== MemAlign manipulate + rollback: held-out agreement ===")
    _print_row("BASE", base_report)
    _print_row("ALIGNED", aligned_report)
    _print_row("OVERFIT", overfit_report)
    _print_row("ROLLED-BACK", rolled_back_report)
    print(
        "\nExpected shape: ALIGNED >= BASE, OVERFIT < ALIGNED (manipulation bites), "
        "ROLLED-BACK > OVERFIT (retraction recovers). Magnitudes are coarse — see the "
        "caveat in docs/MEMALIGN_ROLLBACK.md."
    )
    moved_down = overfit_report.agreement_rate < aligned_report.agreement_rate
    recovered = rolled_back_report.agreement_rate > overfit_report.agreement_rate
    print(
        f"\nMechanics: manipulation moved agreement DOWN = {moved_down}; "
        f"rollback RECOVERED agreement = {recovered}"
    )
    return 0


# ---------------------------------------------------------------------------
# Guard + CLI.
# ---------------------------------------------------------------------------


def _guard() -> str | None:
    """Return a reason to refuse to run, or ``None`` if the live preconditions hold."""
    if os.environ.get("AIL_LIVE_MLFLOW") != "1":
        return (
            "refusing to run: this makes live Databricks model calls. Set AIL_LIVE_MLFLOW=1 "
            "to confirm you want to run it by hand (never in CI)."
        )
    import importlib.util

    if importlib.util.find_spec("dspy") is None:
        return (
            "MemAlign requires the optional 'align' extra (dspy). Install it with: "
            "pip install 'dspy>=3.2.1,<4' (or pip install -e '.[align]')."
        )
    return None


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--experiment-id", required=True, help="MLflow experiment id to read traces from"
    )
    p.add_argument("--profile", default=None, help="Databricks CLI profile (sets the workspace)")
    p.add_argument(
        "--token-cap", type=int, default=DEFAULT_TOKEN_CAP, help="max total_tokens/trace"
    )
    p.add_argument("--max-traces", type=int, default=200, help="trace fetch ceiling")
    p.add_argument("--anchor-fraction", type=float, default=0.3, help="held-out anchor fraction")
    p.add_argument(
        "--bias-fraction",
        type=float,
        default=0.4,
        help="fraction of the alignment pool to bias and then retract",
    )
    p.add_argument("--seed", type=int, default=0, help="split seed (reproducible)")
    p.add_argument("--labeler-id", default="demo-grader", help="source_id for human assessments")
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="model that scores a trace")
    p.add_argument(
        "--reflection-lm", default=DEFAULT_REFLECTION_LM, help="MemAlign reflection model"
    )
    p.add_argument(
        "--embedding-model", default=DEFAULT_EMBEDDING_MODEL, help="MemAlign embedding model"
    )
    p.add_argument(
        "--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM, help="MemAlign embedding dim"
    )
    p.add_argument("--retrieval-k", type=int, default=5, help="MemAlign episodic retrieval k")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    reason = _guard()
    if reason is not None:
        print(reason, file=sys.stderr)
        return 2

    if args.profile:
        os.environ["DATABRICKS_CONFIG_PROFILE"] = args.profile

    cfg = DemoConfig(
        experiment_id=args.experiment_id,
        profile=args.profile,
        token_cap=args.token_cap,
        max_traces=args.max_traces,
        anchor_fraction=args.anchor_fraction,
        bias_fraction=args.bias_fraction,
        seed=args.seed,
        labeler_id=args.labeler_id,
        judge_model=args.judge_model,
        memalign=MemAlignConfig(
            reflection_lm=args.reflection_lm,
            retrieval_k=args.retrieval_k,
            embedding_model=args.embedding_model,
            embedding_dim=args.embedding_dim,
        ),
    )
    return run_demo(cfg)


if __name__ == "__main__":
    sys.exit(main())
