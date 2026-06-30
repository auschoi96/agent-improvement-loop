#!/usr/bin/env python3
"""MemAlign **manipulate + rollback** showcase: align a judge, overfit it, roll it back.

This is the operational counterpart to the unit-tested L2 judge layer
(:mod:`ail.judges`). It exercises the *real* MLflow MemAlign path end to end on a
live Databricks workspace and proves a single mechanic:

    adding human-feedback memory to a judge moves its agreement with held-out
    humans, and **retracting** that memory (``unalign``) moves it back.

Four measurements on one frozen, **stratified** held-out Human Anchor:

* **BASE** — the unaligned ``{{ trace }}`` token-efficiency judge.
* **ALIGNED** — BASE after ``align`` on a genuine, human-labeled set.
* **OVERFIT** — ALIGNED after a *second* ``align`` on a deliberately biased subset
  (relabeled to a constant **high** score), which teaches the judge to call
  everything efficient and so drags held-out agreement DOWN.
* **ROLLED-BACK** — OVERFIT after ``unalign(traces=...)`` retracts exactly the
  biased traces, so agreement RECOVERS to ≈ ALIGNED.

Two design choices make the overfit→rollback dynamic *visible* (earlier it could
not fire — see ``docs/MEMALIGN_ROLLBACK.md``):

1. **A representative, stratified anchor.** Labels come from real human
   ``token_efficiency`` assessments on traces tagged ``tags.labeling_set='v1'``;
   the anchor is selected with :func:`ail.judges.stratified_split_labels` so it
   spans the human grade range and **includes the discriminating low-efficiency
   examples** — not only the high ones a uniform random draw would yield on a
   small-trace corpus that skews efficient. An all-high anchor cannot detect a
   judge pushed toward high scores; a stratified one can.
2. **A known-wrong-direction bias.** The bias subset is relabeled to a constant
   high grade (:data:`BIAS_TARGET_GRADE`), which *disagrees* with the low anchor
   examples. So OVERFIT measurably disagrees with the held-out humans, and the
   rollback recovers.

If the available small (judge-ingestible) traces genuinely lack low-grade
examples, the anchor is **not discriminating** and the demo says so honestly
rather than faking a drop — that is a label-availability limit, not a MemAlign
failure.

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
        --labeling-set v1 \
        --token-cap 50000 \
        --max-traces 200
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
    StratifiedAnchorSplit,
    TraceLabel,
    align_judge,
    assert_pools_disjoint,
    build_memalign_optimizer,
    coerce_score,
    make_scorer,
    score_anchor,
    stratified_split_labels,
    to_alignment_set,
    to_human_anchor,
)
from ail.metrics import compute_trace_metrics

if TYPE_CHECKING:
    from ail.ingest.base import NormalizedTrace
    from ail.judges.contract import AgreementReport

# Models for the MemAlign optimizer, per the demo brief. The judge model (what
# actually scores a trace) is separate and configurable; it defaults to the same
# Claude Sonnet used for reflection so the showcase reads from one model family.
DEFAULT_REFLECTION_LM = "databricks:/databricks-claude-sonnet-4-6"
DEFAULT_EMBEDDING_MODEL = "databricks:/databricks-gte-large-en"
DEFAULT_EMBEDDING_DIM = 1024
DEFAULT_JUDGE_MODEL = "databricks:/databricks-claude-sonnet-4-6"
DEFAULT_TOKEN_CAP = 50_000

#: Traces carrying the human ``token_efficiency`` labels this demo reads. The
#: labels are real human assessments applied in the MLflow UI (or via
#: :func:`ail.judges.record_labels`); the tag scopes the demo to that labeled
#: slice. Never fabricated — see ``docs/MEMALIGN_ROLLBACK.md``.
DEFAULT_LABELING_SET = "v1"

#: The constant high grade the bias subset is relabeled to. Pushing the judge to
#: call every run maximally efficient (5) is a known-wrong direction relative to
#: a representative anchor: it *disagrees* with the held-out low-efficiency
#: examples, which is what drags OVERFIT agreement DOWN. Retracting these traces
#: (``unalign``) is what recovers it.
BIAS_TARGET_GRADE = 5.0

#: How close ROLLED-BACK must return to ALIGNED to count as "recovered". Episodic
#: memory is not guaranteed to reconstruct byte-identically after a retract, so
#: recovery is "back to within this of ALIGNED", not "exactly ALIGNED".
DEFAULT_RECOVERY_TOLERANCE = 0.05


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
    labeling_set: str
    token_cap: int
    max_traces: int
    anchor_fraction: float
    bias_fraction: float
    seed: int
    labeler_id: str
    judge_model: str
    memalign: MemAlignConfig


@dataclass(frozen=True, slots=True)
class RollbackDynamics:
    """Whether the four agreement numbers show the manipulate→rollback shape.

    The honest self-check: ``manipulation_moved_down`` is OVERFIT below ALIGNED
    (the biased memory bit), and ``rollback_recovered`` is ROLLED-BACK back above
    OVERFIT *and* within tolerance of ALIGNED (the retraction restored it). Both
    true means the dynamic fired.
    """

    manipulation_moved_down: bool
    rollback_recovered: bool

    @property
    def fired(self) -> bool:
        return self.manipulation_moved_down and self.rollback_recovered


def classify_rollback_dynamics(
    *,
    aligned: float,
    overfit: float,
    rolled_back: float,
    recovery_tolerance: float = DEFAULT_RECOVERY_TOLERANCE,
) -> RollbackDynamics:
    """Classify the down/recover shape from the held-out agreement rates.

    Pure and offline (no model, no MLflow): given the ALIGNED, OVERFIT and
    ROLLED-BACK agreement rates, decide whether the manipulation moved agreement
    DOWN (``overfit < aligned``) and whether the rollback RECOVERED it
    (``rolled_back > overfit`` *and* ``rolled_back >= aligned - tolerance``, i.e.
    back to ≈ ALIGNED). This is the logic the demo's self-check prints; it is
    unit-tested on synthetic numbers.
    """
    moved_down = overfit < aligned
    recovered = rolled_back > overfit and rolled_back >= aligned - recovery_tolerance
    return RollbackDynamics(manipulation_moved_down=moved_down, rollback_recovered=recovered)


# ---------------------------------------------------------------------------
# Reading real human labels off the labeled traces (never fabricated).
# ---------------------------------------------------------------------------


def human_grade(
    trace: NormalizedTrace, *, name: str, labeler_id: str | None = None
) -> tuple[float, str | None] | None:
    """The human grade + rationale for ``name`` on a trace, or ``None`` if unlabeled.

    Reads ``trace.raw.info.assessments`` — the MLflow assessments the labeler
    attached — keeps the ``HUMAN``-sourced ones whose name matches the judge, and
    returns the first one's numeric value (coerced via the shared
    :func:`ail.judges.coerce_score`) with its rationale. ``labeler_id``, when
    given, *prefers* that labeler's assessment but falls back to any human one.
    Returns ``None`` when the trace carries no numeric human grade for ``name`` —
    the caller skips it rather than inventing a label.
    """
    info = getattr(getattr(trace, "raw", None), "info", None)
    assessments = getattr(info, "assessments", None) if info is not None else None
    if not assessments:
        return None
    human = [
        a
        for a in assessments
        if getattr(a, "name", None) == name
        and str(getattr(getattr(a, "source", None), "source_type", "")) == "HUMAN"
    ]
    if labeler_id is not None:
        preferred = [
            a for a in human if getattr(getattr(a, "source", None), "source_id", None) == labeler_id
        ]
        human = preferred or human
    for assessment in human:
        grade = _to_grade(coerce_score(assessment))
        if grade is not None:
            return grade, getattr(assessment, "rationale", None)
    return None


def _to_grade(value: Any) -> float | None:
    """Float value of a numeric grade; ``None`` for booleans or non-numeric labels."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except (TypeError, ValueError):
            return None
    return None


# ---------------------------------------------------------------------------
# Pipeline steps.
# ---------------------------------------------------------------------------


def collect_labeled_traces(source: MLflowTraceSource, cfg: DemoConfig) -> list[TraceLabel]:
    """Read the labeled slice, keep judge-ingestible traces, and pull real human grades.

    Scopes the fetch to ``tags.labeling_set='<set>'`` (the human-labeled slice),
    keeps only traces under ``--token-cap`` (so each fits a ``{{ trace }}`` judge's
    context), and reads each one's **real** human ``token_efficiency`` grade off
    its assessments. Traces with no human grade are skipped (never fabricated).
    """
    traces: list[NormalizedTrace] = source.fetch_traces(
        experiment_id=cfg.experiment_id,
        filter_string=f"tags.labeling_set = '{cfg.labeling_set}'",
        max_results=cfg.max_traces,
        order_by=["timestamp_ms DESC"],
    )
    labels: list[TraceLabel] = []
    skipped_oversize = 0
    skipped_unlabeled = 0
    for trace in traces:
        metrics = compute_trace_metrics(trace)
        if metrics.tokens.total_tokens > cfg.token_cap:
            skipped_oversize += 1
            continue
        graded = human_grade(trace, name="token_efficiency", labeler_id=None)
        if graded is None:
            skipped_unlabeled += 1
            continue
        value, rationale = graded
        labels.append(
            TraceLabel(
                trace_id=trace.trace_id,
                name="token_efficiency",
                value=value,
                rationale=rationale
                or f"human token_efficiency label (labeling_set={cfg.labeling_set})",
            )
        )
    print(
        f"  fetched {len(traces)} traces tagged labeling_set={cfg.labeling_set}; "
        f"kept {len(labels)} with a human grade under the {cfg.token_cap:,}-token cap "
        f"({skipped_oversize} dropped as oversize, {skipped_unlabeled} as unlabeled)"
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


def _print_anchor_coverage(split: StratifiedAnchorSplit) -> bool:
    """Print the anchor's grade coverage; return whether it can detect the bias.

    A discriminating anchor (spans grades AND includes a low-efficiency example)
    is the precondition for the constant-high bias to register as disagreement.
    When it is not discriminating, this prints an explicit, honest warning that
    the dynamic cannot be shown on the available labels.
    """
    grades = ", ".join(f"{g:g}" for g in split.distinct_anchor_grades) or "(none)"
    print(
        f"  anchor grade coverage: {{{grades}}}, span={split.grade_span:g}, "
        f"includes low-efficiency example={split.includes_low}"
    )
    if split.is_discriminating:
        print(
            "  -> representative anchor: a constant-high bias must DISAGREE with the low "
            "examples, so OVERFIT agreement can drop and the rollback can recover it."
        )
        return True
    print(
        "  -> WARNING: this anchor has no discriminating low-efficiency example, so a "
        "high-score bias cannot be detected as disagreement. The overfit->rollback "
        "dynamic CANNOT be shown on these labels. This is a label-availability limit "
        "(small judge-ingestible traces skew efficient), NOT a MemAlign failure. The "
        "numbers below are still reported honestly."
    )
    return False


def run_demo(cfg: DemoConfig) -> int:
    source = MLflowTraceSource(profile=cfg.profile)

    print("[1/6] Reading real human labels on the judge-ingestible labeled slice...")
    labels = collect_labeled_traces(source, cfg)
    if len(labels) < 6:
        print(
            f"ERROR: only {len(labels)} human-labeled traces under the cap; need >=6 to form a "
            "good/biased/anchor split. Raise --token-cap / --max-traces, or label more traces "
            f"with tags.labeling_set={cfg.labeling_set}.",
            file=sys.stderr,
        )
        return 1

    print("[2/6] Stratified split (anchor spans the grade range; alignment -> good + biased)...")
    anchor_split = stratified_split_labels(
        labels, name="token_efficiency", anchor_fraction=cfg.anchor_fraction, seed=cfg.seed
    )
    # Sub-split the alignment pool (trace-level, disjoint, also stratified) into the
    # genuine subset and the subset we will deliberately bias and then retract. The
    # bias subset is stratified too, so it includes low-efficiency traces — making
    # the constant-high relabel a genuinely wrong signal, not a no-op.
    bias_split = stratified_split_labels(
        anchor_split.alignment_labels,
        name="token_efficiency",
        anchor_fraction=cfg.bias_fraction,
        seed=cfg.seed + 1,
    )
    good_labels = bias_split.alignment_labels
    bias_src_labels = bias_split.anchor_labels
    biased_labels = [replace(lab, value=BIAS_TARGET_GRADE) for lab in bias_src_labels]
    anchor_labels = anchor_split.anchor_labels
    if not good_labels or not biased_labels or not anchor_labels:
        print(
            "ERROR: split produced an empty pool; adjust --anchor-fraction / --bias-fraction "
            "or supply more labeled traces.",
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
        f"  good={len(good_set)} traces, biased={len(biased_set)} traces (relabeled to "
        f"{BIAS_TARGET_GRADE:g}), anchor={len(anchor)} held-out items (disjoint wall proven)"
    )
    discriminating = _print_anchor_coverage(anchor_split)

    print("[3/6] Building the {{ trace }} token-efficiency judge (make_judge)...")
    base_judge = make_scorer(TOKEN_EFFICIENCY_TRACE, model=cfg.judge_model)
    optimizer = build_memalign_optimizer(cfg.memalign)

    print("[4/6] Measuring BASE, then aligning on the genuine set...")
    base_report = _agreement(base_judge, anchor)
    aligned = align_judge(base_judge, good_set, optimizer=optimizer).judge
    aligned_report = _agreement(aligned, anchor)

    print("[5/6] OVERFITTING on the biased (constant-high) subset...")
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
        "ROLLED-BACK ~= ALIGNED (retraction recovers)."
    )

    dynamics = classify_rollback_dynamics(
        aligned=aligned_report.agreement_rate,
        overfit=overfit_report.agreement_rate,
        rolled_back=rolled_back_report.agreement_rate,
    )
    print(
        f"\nSelf-check: manipulation moved agreement DOWN = {dynamics.manipulation_moved_down}; "
        f"rollback RECOVERED to ~= ALIGNED = {dynamics.rollback_recovered}"
    )
    print(f"DYNAMIC FIRED = {dynamics.fired}")
    if not dynamics.fired and not discriminating:
        print(
            "  (Expected: the held-out labels were not discriminating enough to show the "
            "dynamic. Label low-efficiency, judge-ingestible traces to make it fire — see "
            "docs/MEMALIGN_ROLLBACK.md.)"
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
        "--labeling-set",
        default=DEFAULT_LABELING_SET,
        help="tags.labeling_set value scoping the human-labeled slice to read",
    )
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
    p.add_argument(
        "--labeler-id",
        default="demo-grader",
        help="source_id stamped on alignment-set human assessments",
    )
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
        labeling_set=args.labeling_set,
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
