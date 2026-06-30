# MemAlign manipulate + rollback showcase

**Status:** operational demo (run by hand, never in CI) · **Script:**
`scripts/demo_memalign_rollback.py`

This showcase runs the real MLflow MemAlign path end to end on a live Databricks
workspace and proves one mechanic of the L2 judge layer (`src/ail/judges/`):

> Adding human-feedback **memory** to a judge moves its agreement with held-out
> humans, and **retracting** that memory (`unalign`) moves it back.

It aligns a judge on **genuine human feedback**, deliberately **overfits** it on
a biased subset so held-out agreement drops, then **rolls back** exactly that
biased memory and shows agreement recover — four measurements on one frozen,
held-out Human Anchor.

The overfit→rollback dynamic only becomes *visible* with two design choices that
fix the earlier blind spot (see
[Why earlier runs couldn't show the drop](#why-earlier-runs-couldnt-show-the-drop)):

1. **A representative, stratified anchor.** The held-out anchor is drawn with
   `ail.judges.stratified_split_labels`, which samples evenly across the human
   grade range so the anchor **includes the discriminating low-efficiency
   examples** (grade 1–2), not only the high ones a uniform draw yields on a
   small-trace corpus that skews efficient.
2. **A known-wrong-direction bias.** The bias subset is relabeled to a constant
   **high** grade (`BIAS_TARGET_GRADE = 5`), which *disagrees* with those low
   anchor examples — so OVERFIT measurably drops, and the rollback recovers.

It is also the live exercise that surfaced three pipeline bugs the mock tests
missed; those are fixed (with offline tests) in `src/ail/judges/` — see
[The three bugs this demo exposed](#the-three-bugs-this-demo-exposed).

## What it prints

```
=== MemAlign manipulate + rollback: held-out agreement ===
  BASE         agreement_rate=0.XXX  (scored N/N, ...)
  ALIGNED      agreement_rate=0.XXX  (scored N/N, ...)
  OVERFIT      agreement_rate=0.XXX  (scored N/N, ...)
  ROLLED-BACK  agreement_rate=0.XXX  (scored N/N, ...)

Self-check: manipulation moved agreement DOWN = True; rollback RECOVERED to ~= ALIGNED = True
DYNAMIC FIRED = True
```

Expected shape: `ALIGNED ≥ BASE`, `OVERFIT < ALIGNED` (the manipulation bites),
`ROLLED-BACK ~= ALIGNED` (retraction recovers). The self-check is **honest**: if
the available labels can't make the dynamic fire (no discriminating low examples
among judge-ingestible traces), it prints `DYNAMIC FIRED = False` and says why,
rather than faking a drop — see
[Honest limitation](#honest-limitation-label-availability).

## How to run

Requires: a Databricks profile with model-serving access, an experiment holding
agent traces **with human `token_efficiency` labels** (tagged
`tags.labeling_set='v1'`), and the optional `align` extra (`dspy`).

```bash
pip install -e '.[dev,align]'          # dspy is the MemAlign optimizer backend

AIL_LIVE_MLFLOW=1 python scripts/demo_memalign_rollback.py \
    --experiment-id 660599403165942 \
    --profile dais-demo \
    --labeling-set v1 \
    --token-cap 50000 \
    --max-traces 200
```

The script **self-guards**: without `AIL_LIVE_MLFLOW=1` it refuses to run (it
makes live, billable model calls), and without `dspy` it exits with install
guidance. It is never collected by pytest (it lives in `scripts/`, and the live
test markers stay deselected by default).

Key flags (all have defaults):

| Flag | Default | Meaning |
|---|---|---|
| `--experiment-id` | _(required)_ | MLflow experiment to read traces from |
| `--profile` | none | Databricks CLI profile (selects the workspace) |
| `--labeling-set` | `v1` | `tags.labeling_set` value scoping the human-labeled slice to read |
| `--token-cap` | `50000` | drop traces whose `total_tokens` exceed this (a `{{ trace }}` judge must fit the trace in context) |
| `--max-traces` | `200` | trace fetch ceiling |
| `--anchor-fraction` | `0.3` | fraction held out as the (stratified) Human Anchor |
| `--bias-fraction` | `0.4` | fraction of the alignment pool to relabel constant-high then retract |
| `--reflection-lm` | `databricks:/databricks-claude-sonnet-4-6` | MemAlign guideline-distillation model |
| `--embedding-model` | `databricks:/databricks-gte-large-en` | MemAlign episodic-memory embeddings |
| `--embedding-dim` | `1024` | embedding dimension |
| `--judge-model` | `databricks:/databricks-claude-sonnet-4-6` | the model that scores a trace |

`MemAlignConfig(reflection_lm=..., embedding_model=..., embedding_dim=...)` is
built from these and passed to `build_memalign_optimizer`.

## Structure (the four stages)

1. **Read real human labels.** Fetch the traces tagged
   `tags.labeling_set='<set>'`, keep those under `--token-cap` (so each fits a
   `{{ trace }}` judge's context), and read each one's **real** human
   `token_efficiency` grade off `trace.info.assessments` (`human_grade`). Labels
   are never fabricated; a trace with no human grade is skipped and reported.
   Grades are floats so a `±1` tolerance counts a within-one-grade judge score as
   agreement.
2. **Stratified split into disjoint pools.** `stratified_split_labels` holds out
   the **Human Anchor** by sampling evenly across the grade-sorted traces — always
   including the lowest- and highest-graded trace — so the anchor spans the range
   and **includes discriminating low examples** (the fix for the old all-high
   anchor). The rest is the alignment pool, sub-split (trace-level, disjoint, also
   stratified) into a **genuine** subset and a **bias** subset.
   `assert_pools_disjoint` proves the frozen wall across all three before any
   model call. The anchor's traces are **blinded** — `to_human_anchor` strips
   their `HUMAN` assessments — so the `{{ trace }}` judge cannot read the gold it
   is measured against off the trace; the gold lives only on
   `AnchorItem.human_label`. The demo prints the anchor's achieved grade coverage
   and whether it is *discriminating*.
3. **Build the judge + align.**
   - **BASE** — the unaligned `{{ trace }}` token-efficiency judge (`make_judge`).
   - **ALIGNED** — `align_judge(base, genuine_set, optimizer)` → a
     `MemoryAugmentedJudge` carrying distilled guidelines + episodic examples.
4. **Manipulate, then roll back.**
   - **OVERFIT** — `align_judge(aligned, biased_set, optimizer)`. The bias subset
     is relabeled to a constant **high** grade (`BIAS_TARGET_GRADE = 5`), which
     teaches the judge to call every run maximally efficient — a known-wrong
     direction that *disagrees* with the anchor's low examples, dragging held-out
     agreement DOWN.
   - **ROLLED-BACK** — `overfit.unalign(traces=biased_set.traces)` retracts
     exactly those traces, leaving the genuine memory intact, so agreement
     recovers to ≈ ALIGNED.

Each stage is measured with `score_anchor(judge, anchor, ...)` on the **same**
held-out anchor, so the four numbers are comparable. `classify_rollback_dynamics`
turns them into the honest `DOWN` / `RECOVERED` self-check.

## The unalign API

Discovered in `mlflow/genai/judges/optimizers/memalign/optimizer.py`:

- `Judge.align(traces, optimizer=...)` returns a **`MemoryAugmentedJudge`** (not a
  plain judge). Re-aligning a `MemoryAugmentedJudge` *adds* the new traces to its
  memory.
- `MemoryAugmentedJudge.unalign(traces: list[Trace]) -> MemoryAugmentedJudge`
  returns a new judge with those traces removed: every episodic example whose
  `_trace_id` is in `{t.info.trace_id for t in traces}` is dropped, and every
  distilled guideline whose source traces were *all* removed is deleted
  (guidelines with at least one surviving source trace are kept).

The demo calls `unalign` **directly** on the OVERFIT judge — it is the genuine
MLflow retraction path, not an `ail` wrapper. MemAlign deliberately refuses to
treat "re-align with empty assessments" as retraction (it raises); `unalign` is
the supported way to remove a trace's contribution.

## The three bugs this demo exposed

Operating this live failed in three places the mock tests never hit. All are
fixed with offline (fake judge/optimizer) tests:

1. **Alignment set dropped its human feedback.** `to_alignment_set` fetched raw
   traces via `source.get_trace(tid).raw`, but those objects did not carry the
   `HUMAN` assessments MemAlign reads from `trace.info.assessments`, so alignment
   failed with *"No valid feedback records found"*. Fixed by attaching each
   label's value onto the raw trace as a `HUMAN` `Feedback`
   (`ail/judges/labeling.py`).
2. **A `{{ trace }}` judge was never scored.** `score_anchor` always called the
   judge with `inputs/outputs/expectations`, so a judge that requires a `trace`
   input (`get_input_fields() == ['trace']`) raised on every item → 0 scored →
   `distrusted`. Fixed by calling the judge with only the fields it declares,
   passing the anchor item's `trace` when required (`ail/judges/agreement.py`,
   `AnchorItem.trace` in `ail/pools.py`).
3. **Numeric-string scores never matched numeric labels.** A judge returning
   `"3"` (string) compared `!=` a human `3.0` (float) and never agreed, even
   within tolerance. Fixed so numeric-looking strings compare numerically against
   numeric human labels, honouring `numeric_tolerance` (`ail/judges/agreement.py`).

## Why earlier runs couldn't show the drop

A `{{ trace }}` judge has to fit the whole trace in its context, so the demo caps
traces at `--token-cap` (50K). For **token efficiency** the most discriminating
examples — the genuinely *wasteful*, low-scoring runs — are often the **huge**
traces the cap drops, so the small traces that remain skew toward **high
(efficient)** labels. The earlier demo then held the anchor out with a **uniform
random** draw, which on that skew yields an anchor of *only* high grades. An
all-high anchor cannot detect a judge pushed toward high scores — biasing the
judge UP only *increased* agreement with it. So the overfit→rollback dynamic
could not fire (it reported `manipulation moved agreement DOWN = False`), and the
old version fabricated grades from an L0 redundancy heuristic rather than reading
real human labels.

This version fixes both: it reads **real** human labels, and holds the anchor out
with `stratified_split_labels`, which always includes the lowest-graded trace. A
constant-high bias then *disagrees* with those held-out low examples, so OVERFIT
drops and `unalign` recovers it.

## Honest limitation: label availability

The fix depends on there being **discriminating low-efficiency examples among the
judge-ingestible (small) traces**. If the labeled, under-cap slice genuinely has
none — every small trace was graded efficient — then no honest anchor can detect a
high-score bias, and the demo says so:

```
  anchor grade coverage: {4, 5}, span=1, includes low-efficiency example=False
  -> WARNING: this anchor has no discriminating low-efficiency example ... The
     overfit->rollback dynamic CANNOT be shown on these labels. This is a
     label-availability limit ..., NOT a MemAlign failure.
...
DYNAMIC FIRED = False
  (Expected: the held-out labels were not discriminating enough ...)
```

The demo never fakes a drop to make the story land. To make the dynamic fire,
label some low-efficiency, *small* traces with `tags.labeling_set='v1'` (or use a
dimension whose discriminating examples are naturally small — a focused
correctness, groundedness, or tool-selection judge — which keeps the same
machinery while making the numbers sharp). The production `token_efficiency`
scorer (`ail.judges.scorers`) sidesteps the cap entirely by judging an **L0
summary** instead of the raw trace; the `{{ trace }}` variant exists here only to
exercise the trace-judge path.

## Related

- `docs/L2_JUDGES_CONTRACT.md` — the three-pool discipline, `align_judge`,
  `score_anchor`, and the agreement contract.
- `src/ail/judges/labeling.py` — recording labels and assembling the disjoint
  Alignment Set / Human Anchor.
- `PROVENANCE.md` — this is a clean-room implementation against public OSS MLflow
  GenAI APIs.
