# Auto-align trigger

**Status:** stable · **Module:** `src/ail/judges/auto_align.py` · **Job:**
`src/ail/jobs/auto_align_job.py` (`ail-auto-align`) · **Bundle:**
`resources/auto_align.job.yml`

The auto-align trigger closes the L2 loop. All the pieces already exist —
labelling, MemAlign alignment, judge-vs-human agreement, registration (see
`docs/L2_JUDGES_CONTRACT.md`) — but nothing *drives* them. Today a human labels
traces in the MLflow UI and a judge stays `aligned=false` (and therefore
DISTRUSTED) until someone runs the align/audit flow by hand. This trigger makes
that automatic:

> Run on a schedule and, **per judged dimension**: align the judge with MemAlign
> once enough human labels exist, re-align as more accrue, guard trust with the
> agreement floor, and roll back a regression.

It turns *"a human adds labels"* into *"the judge becomes trusted automatically."*

## What it is (and is not)

It is **orchestration only** — it reuses, and never reimplements, the L2 pieces:

| Step | Reuses |
|---|---|
| Which labels to align from | `read_human_labels` → the `HUMAN` assessments **named for the judge** on the experiment's traces (the L1 `label-schema-name == judge-name` convention, `ail.judges.authoring`) |
| Split into disjoint pools + prove the wall | `ail.judges.labeling.assemble_pools` |
| Align | `ail.judges.alignment.align_judge` (MemAlign, on the Alignment Set only) |
| Audit | `ail.judges.agreement.score_anchor` (held-out Human Anchor, fail-closed) |
| Promote | `ail.judges.registration.register_prealigned_scorer` |

The only new logic is the **trust gating** described below. There is no new judge,
scorer, alignment, or agreement behaviour.

## The decision, per judged dimension

`auto_align_judge(spec, *, experiment_id, source, store, config, ...)` runs one
cadence for one dimension:

1. **Read labels.** Count the `HUMAN`-sourced labels whose name matches the judge.
2. **Floor gate.** Skip (`SKIPPED_BELOW_FLOOR`) if fewer than `label_floor`
   (default **20**) — too few to both align and hold out a meaningful anchor.
3. **Watermark gate.** Skip (`SKIPPED_NO_NEW_LABELS`) if the count has not grown
   past the per-judge **watermark** (the label count at the last cadence that ran).
   This makes the trigger **idempotent** — it never re-aligns on the same labels —
   and is exactly what makes it **re-align over time**: once labels accrue past the
   watermark, it proceeds.
4. **Align + audit.** `assemble_pools` → `align_judge` (a fresh judge from the base
   spec, over the full current label set) → `score_anchor` on the held-out anchor.
5. **Agreement-floor guard (fail-closed).** If the aligned judge is `distrusted`
   — **unmeasured** (empty / under-sampled anchor → `insufficient_data`) *or* below
   the agreement floor — it is **held** (`HELD_DISTRUSTED`), not promoted. An
   unmeasured judge never reads as trusted, and the agreement number is
   `score_anchor`'s, never fabricated.
6. **Rollback (fail-closed toward last-known-good).** If its held-out agreement
   **regresses below the previously-promoted version**, the candidate is **not**
   registered (`ROLLED_BACK`); the prior aligned version stays live.
7. **Promote.** Otherwise register **the exact judge whose agreement was measured**
   (`ALIGNED`) and advance the watermark **and** the last-known-good agreement bar.

```
labels < floor ─────────────────────────────► SKIPPED_BELOW_FLOOR
labels <= watermark ────────────────────────► SKIPPED_NO_NEW_LABELS
align + score → distrusted (unmeasured/low) ─► HELD_DISTRUSTED   (keep prior)
align + score → agreement < prior aligned ──► ROLLED_BACK        (keep prior)
align + score → trusted, no regression ─────► ALIGNED            (register)
cadence raised ─────────────────────────────► FAILED             (isolated, reported)
```

### Why "rollback" is "don't promote"

Each cadence re-aligns a **fresh** judge from the base spec over the full current
label set, so a regression is handled by simply **not promoting** the candidate:
the incumbent scheduled scorer is left untouched (registry-versioning rollback).
There is never a window where a regressed judge is live — the trigger
measures-before-registering, which is why it uses
`register_prealigned_scorer` (register the exact measured judge) rather than
`create_aligned_scorer` (which would re-run MemAlign and register a *different*
judge than the one measured). This is the complement to MemAlign's `unalign`,
which the manipulate/rollback showcase (`docs/MEMALIGN_ROLLBACK.md`) uses for
incremental retraction; a from-scratch re-alignment makes not-promoting the
simplest correct fail-closed move.

### The watermark

The watermark advances to the current label count on **every** cadence that ran
alignment — including a held or rolled-back one — so the same labels are never
re-aligned repeatedly; a future cadence retries only once **more** labels accrue.
The last-known-good agreement bar advances **only** on a promotion.

It is persisted as three experiment tags per judge under `ail.autoalign.<name>.`
(`label_count` / `agreement` / `aligned_at`) — mirroring
`ail.judge.<name>.aligned` from registration, because the scheduled-scorer API
exposes no per-scorer metadata slot. The store (`WatermarkStore`) is injectable;
`ExperimentTagWatermarkStore` is the production implementation, and reads fail
closed to a never-aligned state.

## Scheduled, not event-triggered

The v4 trace store tables are **views** (`cc_trace_unified` / `cc_trace_metadata`),
not Delta tables, so a `table_update` trigger is infeasible — the same reason the
optimization cycle is scheduled. `ail-auto-align` runs on a cron
(`auto_align_cron`, default daily 06:00). It is **model-only**: it reads labeled
traces and calls the reflection / embedding / judge models through the gateway; it
has no SQL write path of its own. Like the other jobs it resolves an explicit
bearer for the v4 trace store (`resolve_job_auth`) and exports the SQL warehouse
(`MLFLOW_TRACING_SQL_WAREHOUSE_ID`) the read needs.

The exit code is **non-zero only when a judge's cadence failed** (raised). A
correct hold, rollback, or skip is a *successful* run — the job never prints a
fabricated success.

## Usage

Library:

```python
from ail.judges import auto_align_scorers, AutoAlignConfig
from ail.judges.agreement import AgreementConfig

report = auto_align_scorers(
    "660599403165942",
    config=AutoAlignConfig(
        label_floor=20,
        agreement=AgreementConfig(floor=0.7, numeric_tolerance=1.0),  # tol 1 for 1-5 judges
    ),
)
report.n_aligned, report.n_rolled_back, report.n_held_distrusted, report.n_skipped
```

CLI / Job (`ail-auto-align`), and the scheduled bundle job:

```bash
ail-auto-align \
    --experiment 660599403165942 \
    --warehouse-id <sql-warehouse-id> \
    --label-floor 20 --agreement-floor 0.7 \
    --no-register            # dry run: run the full decision, register nothing

databricks bundle deploy -t dais_demo --profile dais-demo   # deploys resources/auto_align.job.yml
```

A judge with fewer than `--label-floor` labels simply skips, so running over the
full built-in scorer set is harmless: only dimensions humans are labelling get
aligned.

## Bundle knobs (`databricks.yml`)

| Variable | Default | Meaning |
|---|---|---|
| `auto_align_cron` | `0 0 6 * * ?` | schedule (daily 06:00) |
| `auto_align_pause_status` | `UNPAUSED` | `PAUSED` to deploy dormant |
| `auto_align_label_floor` | `20` | min labels before first alignment |
| `auto_align_agreement_floor` | `0.7` | min judge-vs-human agreement to trust |
| `auto_align_min_anchor_samples` | `1` | min scored anchor items (fail-closed floor) |
| `auto_align_numeric_tolerance` | `0.0` | float-label agreement tolerance |
| `auto_align_sampling_rate` | `0.1` | scheduled-scorer sampling on promotion |
| `auto_align_judges` | `''` (all) | comma-separated judge names to align |
| `auto_align_judge_model` / `auto_align_reflection_lm` / `auto_align_embedding_model` | `''` | model URIs; empty → MLflow defaults |

## Which judges fit

The read-back audits a judge on the **trace** the human labelled (each anchor item
carries the blinded raw trace), so it is the natural fit for **`{{ trace }}`
judges** — the authored judges from `ail.judges.authoring`, whose label schema is
named to match. Field-based judges (`correctness` / `groundedness` / `modularity`,
which score `{{ inputs }}`/`{{ outputs }}`) can still be aligned, but scoring them
on the anchor needs the labelling path to carry `inputs`/`outputs`; without them a
field-based judge scores nothing on the anchor and correctly reads as
`HELD_DISTRUSTED` (fail closed) rather than being falsely promoted.

## Tests

`tests/test_judges_auto_align.py` and `tests/test_auto_align_job.py` mock every
MLflow / model / align / register / score call (in-memory watermark store, fake
trace source) and assert: floor + watermark gating, idempotency, distrusted-when-
unmeasured, and rollback-on-regression. A single live end-to-end is gated behind
`@pytest.mark.live` + `AIL_LIVE_MLFLOW=1` and self-skips otherwise.

## Related

- `docs/L2_JUDGES_CONTRACT.md` — the pieces this trigger orchestrates (labelling,
  `align_judge`, `score_anchor`, registration, the agreement contract).
- `docs/MEMALIGN_ROLLBACK.md` — the live manipulate + `unalign` showcase; the
  agreement-drop dynamic this trigger's rollback guard defends against.
- `docs/READINESS_AND_TRUST.md` — the distrusted-by-default rule and the human-
  labels readiness gate this floor aligns with.
