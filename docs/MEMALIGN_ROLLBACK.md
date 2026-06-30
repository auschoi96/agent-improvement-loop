# MemAlign manipulate + rollback showcase

**Status:** operational demo (run by hand, never in CI) · **Script:**
`scripts/demo_memalign_rollback.py`

This showcase runs the real MLflow MemAlign path end to end on a live Databricks
workspace and proves one mechanic of the L2 judge layer (`src/ail/judges/`):

> Adding human-feedback **memory** to a judge moves its agreement with held-out
> humans, and **retracting** that memory (`unalign`) moves it back.

It aligns a judge on genuine human feedback, deliberately **overfits** it on a
biased subset so held-out agreement drops, then **rolls back** exactly that
biased memory and shows agreement recover — four measurements on one frozen,
held-out Human Anchor.

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

Mechanics: manipulation moved agreement DOWN = True; rollback RECOVERED agreement = True
```

Expected shape: `ALIGNED ≥ BASE`, `OVERFIT < ALIGNED` (the manipulation bites),
`ROLLED-BACK > OVERFIT` (retraction recovers). The **magnitudes are coarse** — see
the [caveat](#honest-caveat-token_efficiency-is-the-wrong-dimension-for-this).

## How to run

Requires: a Databricks profile with model-serving access, an experiment holding
agent traces, and the optional `align` extra (`dspy`).

```bash
pip install -e '.[dev,align]'          # dspy is the MemAlign optimizer backend

AIL_LIVE_MLFLOW=1 python scripts/demo_memalign_rollback.py \
    --experiment-id 660599403165942 \
    --profile dais-demo \
    --token-cap 50000 \
    --max-traces 120
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
| `--token-cap` | `50000` | drop traces whose `total_tokens` exceed this (a `{{ trace }}` judge must fit the trace in context) |
| `--max-traces` | `200` | trace fetch ceiling |
| `--anchor-fraction` | `0.3` | fraction held out as the Human Anchor |
| `--bias-fraction` | `0.4` | fraction of the alignment pool to bias then retract |
| `--reflection-lm` | `databricks:/databricks-claude-sonnet-4-6` | MemAlign guideline-distillation model |
| `--embedding-model` | `databricks:/databricks-gte-large-en` | MemAlign episodic-memory embeddings |
| `--embedding-dim` | `1024` | embedding dimension |
| `--judge-model` | `databricks:/databricks-claude-sonnet-4-6` | the model that scores a trace |

`MemAlignConfig(reflection_lm=..., embedding_model=..., embedding_dim=...)` is
built from these and passed to `build_memalign_optimizer`.

## Structure (the four stages)

1. **Collect + grade.** Fetch traces, compute L0 metrics, keep those under
   `--token-cap`, and grade each with a deterministic stand-in for a human
   reviewer (`human_label_for`): the strict, byte-identical **redundancy rate**
   maps to a 1–5 token-efficiency grade. The grades are floats so a `±1`
   tolerance counts a within-one-grade judge score as agreement.
2. **Split into disjoint pools.** `split_labels` holds out the **Human Anchor**;
   the rest is the alignment pool, sub-split (trace-level, disjoint) into a
   **genuine** subset and a **bias** subset. `assert_pools_disjoint` proves the
   frozen wall across all three before any model call. The anchor's traces carry
   **no** human assessment — the judge never sees the gold it is measured against.
3. **Build the judge + align.**
   - **BASE** — the unaligned `{{ trace }}` token-efficiency judge (`make_judge`).
   - **ALIGNED** — `align_judge(base, genuine_set, optimizer)` → a
     `MemoryAugmentedJudge` carrying distilled guidelines + episodic examples.
4. **Manipulate, then roll back.**
   - **OVERFIT** — `align_judge(aligned, biased_set, optimizer)`. The bias subset
     is the same traces with **inverted** labels (`bias_label`: genuinely
     efficient → "wasteful", genuinely wasteful → "tight"), which teaches the
     judge the opposite of the real pattern.
   - **ROLLED-BACK** — `overfit.unalign(traces=biased_set.traces)` retracts
     exactly those traces, leaving the genuine memory intact.

Each stage is measured with `score_anchor(judge, anchor, ...)` on the **same**
held-out anchor, so the four numbers are comparable.

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

## Honest caveat: `token_efficiency` is the wrong dimension for this

A `{{ trace }}` judge has to fit the whole trace in its context, so the demo caps
traces at `--token-cap` (50K). But for **token efficiency** the most informative,
discriminating examples — the genuinely *wasteful* runs that should score low —
are exactly the **huge** traces (hundreds of thousands of tokens of re-reads and
re-runs). The cap drops them. What's left is a small-trace anchor that skews
toward **high (efficient)** labels, so:

- the label distribution is narrow (mostly 4–5), which makes the agreement
  **magnitude** coarse and the absolute numbers not very meaningful; and
- the "human" grader here is a deterministic L0 redundancy heuristic standing in
  for a real reviewer — reproducible, but not a calibrated ground truth.

This is why the demo claims only the **mechanics**: aligning on memory moves
agreement, and `unalign` moves it back. It is *not* a calibration of the
token-efficiency judge. A dimension whose discriminating examples are **small**
enough to fit a `{{ trace }}` judge — a focused correctness, groundedness, or
tool-selection judge — would keep the same machinery while making the agreement
numbers sharp. The production `token_efficiency` scorer (`ail.judges.scorers`)
sidesteps the cap entirely by judging an **L0 summary** instead of the raw trace;
the `{{ trace }}` variant exists here only to exercise the trace-judge path.

## Related

- `docs/L2_JUDGES_CONTRACT.md` — the three-pool discipline, `align_judge`,
  `score_anchor`, and the agreement contract.
- `src/ail/judges/labeling.py` — recording labels and assembling the disjoint
  Alignment Set / Human Anchor.
- `PROVENANCE.md` — this is a clean-room implementation against public OSS MLflow
  GenAI APIs.
