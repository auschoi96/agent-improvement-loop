# L2 Judged-Metrics Layer

**Status:** stable · **Schema version:** `l2.judges/v1`

The L2 layer (`src/ail/judges/`) is the **judged** tier of the layered metrics
design (`docs/ARCHITECTURE.md` §3): LLM-as-judge scorers built on the public
MLflow GenAI API, aligned with MemAlign, and audited against human labels. It
turns subjective quality ("is this answer correct? is the code modular? is the
claim grounded?") into a measured signal *without* letting the judge and the
agent co-adapt — the failure mode §2 calls the spine of this system.

This is a clean-room implementation against **public OSS MLflow GenAI APIs
only** (`mlflow.genai.judges.make_judge`, `Judge.align`, `MemAlignOptimizer`).
See `PROVENANCE.md`.

## The three-pool discipline (why this layer is shaped the way it is)

Per `docs/ARCHITECTURE.md` §2 there are three **disjoint** pools that are never
mixed:

| Pool | Purpose | Touched by this layer? |
|---|---|---|
| **Task Suite** | fixed tasks, re-run to compare agent versions | never |
| **Alignment Set** | labeled traces that *align* the judge (MemAlign) | `align_judge` only |
| **Human Anchor** | held-out human labels that *audit* the judge | `score_anchor` only |

The discipline is enforced **by the types**, not by convention:

- `align_judge(judge, alignment_set: AlignmentSet, ...)` accepts only an
  `AlignmentSet`. You cannot pass it the Human Anchor or the Task Suite.
- `score_anchor(judge, anchor: HumanAnchor, ...)` accepts only a `HumanAnchor`.
- `assert_pools_disjoint(...)` proves no id leaked across pools and raises
  `PoolOverlapError` otherwise. The loop controller calls it before a cadence.

Measuring agreement on the same labels the judge was aligned against would only
report how well alignment *memorized* them; the Human Anchor is held out for
exactly this reason.

> The pool **storage / curation** (loading traces, freezing the Task Suite,
> promoting human labels) is owned by Waves 1a/1b. This layer owns only the
> **consumer-side handles** it needs (`AlignmentSet`, `HumanAnchor`,
> `AnchorItem`) and the disjointness invariant it honours.

## Decoupled cadence — the anti-co-adaptation safeguard

Judge **alignment** runs on its **own cadence**, deliberately decoupled from
**agent optimization** (`docs/ARCHITECTURE.md` §4). This module has no
dependency on, and makes no call into, the optimizer that tunes the agent.
Aligning a judge (`align_judge`, on the Alignment Set) and auditing it
(`score_anchor`, on the Human Anchor) are separate operations on separate pools
from comparing agent candidates (the Task Suite). Keeping them in separate
functions with disjoint, type-checked inputs is the structural half of
"decoupled cadence"; this section is the documented half.

The **judge-vs-human agreement floor** is the trip-wire: when agreement on the
Human Anchor drops below the configured floor, `AgreementReport.distrusted`
fires and the loop must stop trusting that judge's scores until it is re-aligned
and re-measured. A drifting judge is a distrusted judge — and so is an
*unmeasured* one: an empty or under-sampled anchor fails closed to `distrusted`
rather than reading as trusted (see `insufficient_data` below).

## API surface

### Scorers — `ail.judges.scorers`

Thin, configurable wrappers over `mlflow.genai.judges.make_judge`. Building a
scorer calls **no model** (the judge calls its model lazily on `__call__`), so
construction is offline and free.

```python
from ail.judges import (
    make_correctness_judge, make_modularity_judge,
    make_groundedness_judge, make_token_efficiency_judge,
)

correctness = make_correctness_judge(model="databricks:/...")     # Literal["yes","no"] — the Phase-2 guardrail
modularity  = make_modularity_judge(model="databricks:/...")      # Literal[1,2,3,4,5] — bounded graded scale
groundedness = make_groundedness_judge(model="databricks:/...")   # Literal["yes","no"]
token_eff   = make_token_efficiency_judge(model="databricks:/...")  # Literal[1,2,3,4,5] — hybrid, L0-conditioned

feedback = correctness(inputs=task, outputs=response, expectations=expected)
```

All four use **constrained** structured outputs (an MLflow `Literal[...]`), so a
judge can never emit an out-of-domain value: correctness/groundedness are a true
categorical `"yes"`/`"no"`, and modularity/token-efficiency are bounded to the
integers `1..5`. A bounded `Literal` scale loses `make_judge`'s default mean
aggregation, so those specs restore `["mean", "median", "p90"]` to keep the
graded metric rolling up across traces.

- `ScorerSpec(name, instructions, feedback_value_type, description, aggregations=None)`
  — a reusable scorer definition. The built-in set is `DEFAULT_SCORERS`
  (`correctness`/`modularity`/`groundedness`/`token_efficiency`).
- `make_scorer(spec, *, model=None, instructions=None, feedback_value_type=..., name=None, inference_params=None)`
  — build a `Judge`, overriding rubric/type/name/model per call.
- `with_rubric(spec, instructions)` — a `ScorerSpec` copy with a tuned rubric.

`correctness` is the guardrail Phase 2 needs: a token-reduction intervention may
ship only if correctness does not regress, so it is categorical (clean
pass/fail). `modularity` is graded because structure quality has gradations.
`groundedness` is the anti-hallucination check against the provided context.

#### `token_efficiency` — the hybrid, L0-conditioned scorer

`token_efficiency` (graded `1..5`) is the **hybrid** judge and the direct
Phase-2 partner of `correctness`. It does **not** ask the LLM to count tokens or
recompute redundancy — those are L0 deterministic facts (`ail.metrics`,
un-gameable, `docs/ARCHITECTURE.md` §3). Instead it **consumes** the L0 signals
and adds only the judgement layer.

- `build_token_efficiency_inputs(metrics: TraceMetrics, *, task=None) -> dict`
  is the L0→L2 bridge: it copies the already-computed per-trace L0 signals
  (tokens, tool-call count, redundancy rate, the **named** top repeated targets,
  cost, model, duration) into a compact `{"task": ..., "l0_signals": {...}}`
  dict. Nothing is re-derived; the judge reads this summary.
- **Large-trace-safe by design.** It scores off that L0 *summary*, never
  `{{ trace }}`. The corpus reaches 943K-token traces (§8) that exceed a judge's
  context window, so a flat `{{ trace }}` judge is not an option here; the L0
  digest is. (A recursive-review digest, when L3 exists, plugs into the same
  `inputs` slot.)
- **Quality-conditioned / anti-gaming.** Efficiency is scored *conditioned on
  task success*: spending few tokens by doing less or producing a wrong result
  scores **low**, not high. The rubric refuses to reward "fewer tokens, worse
  outcome", which is exactly why it is paired with `correctness` as the Phase-2
  guardrail — tokens may fall only when quality does not. Its rationale names the
  specific waste (which repeated target / which boilerplate) so the verdict is
  actionable.

### Alignment (MemAlign) — `ail.judges.alignment`

```python
from ail.judges import align_judge, build_memalign_optimizer, MemAlignConfig, AlignmentSet

alignment_set = AlignmentSet.of(labeled_traces)        # Alignment Set pool only
outcome = align_judge(correctness, alignment_set)      # optimizer=None → MLflow's default MemAlign
aligned_judge = outcome.judge                          # better-aligned MLflow Judge
record = outcome.report                                # serializable AlignmentReport

# To configure MemAlign (requires the optional `align` extra — see below):
optimizer = build_memalign_optimizer(MemAlignConfig(retrieval_k=3, reflection_lm="databricks:/..."))
outcome = align_judge(correctness, alignment_set, optimizer=optimizer)
```

> **To run MemAlign alignment, install the optional `align` extra:**
> `pip install -e ".[align]"` (or `pip install "ail[align]"`). It pulls in
> `dspy`, the optimizer backend MLflow's `MemAlignOptimizer` requires. The extra
> is **optional and lazy-imported**: the base (unaligned) judges, `import
> ail.judges`, and CI all work without it — only the judge-alignment cadence
> (`build_memalign_optimizer` / `align_judge` with a real optimizer) needs it.

- `align_judge(judge, alignment_set, *, optimizer=None, generated_at=None) -> AlignmentOutcome`
  — wraps `judge.align(traces=..., optimizer=...)`. Raises `ValueError` on an
  empty set.
- `build_memalign_optimizer(config=None)` — constructs a configured
  `MemAlignOptimizer`. **Imported lazily**; raises a clear `ImportError` when the
  optional `align` extra (`dspy`) is absent (so importing this package, and CI,
  never require `dspy`).
- `MemAlignConfig(reflection_lm, retrieval_k, embedding_model, embedding_dim)` —
  mirrors the public optimizer constructor; `None` values defer to MLflow's
  defaults.

### Agreement (judge-vs-human) — `ail.judges.agreement`

```python
from ail.judges import score_anchor, compute_agreement, AgreementConfig, HumanAnchor, AnchorItem, log_agreement

anchor = HumanAnchor.of([
    AnchorItem(item_id="t1", human_label="yes", inputs=task, outputs=response, expectations=expected),
    # ...
])
report = score_anchor(aligned_judge, anchor, config=AgreementConfig(floor=0.8))
if report.distrusted:
    ...  # agreement fell below the floor → stop trusting this judge
log_agreement(report)   # best-effort MLflow logging (metrics + JSON artifact)
```

- `compute_agreement(pairs, *, judge_name, config=None, generated_at=None) -> AgreementReport`
  — **pure** (no model, no MLflow) over `ScorePair(item_id, judge_value,
  human_value, error)` pairs.
- `score_anchor(judge, anchor, *, config=None, generated_at=None) -> AgreementReport`
  — runs the judge over the anchor, coerces each result (`coerce_score`), pairs
  it with the human label, and delegates to `compute_agreement`. It calls the
  judge with **only the input fields it declares** (`Judge.get_input_fields`): a
  field-based judge gets `inputs`/`outputs`/`expectations`, a `{{ trace }}`-based
  judge gets the item's `trace`. A per-item judge exception is captured (recorded
  as an `error`, counted as a non-agreement) so one bad item never aborts the
  measurement.
- `AgreementConfig(floor=0.7, numeric_tolerance=0.0, case_insensitive=True, min_samples=1)`.
  `case_insensitive` is applied **uniformly** — the agreement decision *and* the
  Cohen's-kappa discretization / label space honour it, so kappa never folds
  labels a deployer chose to keep distinct. `min_samples` is the fail-closed
  knob: below that many *scored* items the judge is unmeasured (see below).
- `log_agreement(report, *, run_id=None) -> bool` — logs
  `judge_human_agreement` / `judge_distrusted` (and `judge_human_cohen_kappa`)
  as metrics plus the full report as a JSON artifact; best-effort, returns
  `False` rather than raising when no run/MLflow is available.

### Registration — `ail.judges.registration` (MemAlign-aware by construction)

Creating/registering a scheduled scorer routes through an **align-then-register**
flow, so MemAlign is the default path whenever labels exist rather than an
optional afterthought:

```python
from ail.judges import create_aligned_scorer, register_scorers, TOKEN_EFFICIENCY

# One scorer, MemAlign-aware:
reg = create_aligned_scorer(
    TOKEN_EFFICIENCY,
    experiment_id="660599403165942",
    alignment_set=alignment_set,   # non-empty -> aligns; None/empty -> base judge
)
reg.aligned          # True iff a labeled set aligned it before registration
reg.report           # serializable AlignmentReport (aligned true/false + notes)
reg.judge            # the registered judge (aligned or base) — auditable via score_anchor

# All scorers at once, same path (aligns ALL when a labeled set is supplied):
registrations = register_scorers("660599403165942", alignment_set=alignment_set)
```

- `create_aligned_scorer(spec, *, experiment_id, alignment_set=None, optimizer=None, model=None, sampling_rate=DEFAULT_SAMPLING_RATE, filter_string=None, profile=None, ...) -> ScorerRegistration`
  — builds the judge; **if** `alignment_set` is non-empty, aligns it with
  MemAlign (`align_judge`, using `optimizer` or a default
  `build_memalign_optimizer()`) and registers the **aligned** judge; **else**
  registers the base judge and flags it `aligned=false`.
- `register_scorers(experiment_id, *, alignment_set=None, optimizer=None, ...) -> list[ScorerRegistration]`
  — routes **every** scorer through `create_aligned_scorer`, so a non-empty
  `alignment_set` aligns all of them and an absent one registers base judges.
- `ScorerRegistration(scorer, judge, aligned, report)` — the active scheduled
  `scorer`, the registered `judge` (auditable), the `aligned` flag, and the
  `AlignmentReport` provenance.

**Why a base judge is flagged, not silently trusted.** The reference experiment
has **zero** human labels (`docs/ARCHITECTURE.md` §8), so MemAlign has nothing to
learn from yet. An unaligned judge is registered with `aligned=false` recorded
authoritatively on its `AlignmentReport` and, best-effort, as an experiment tag
`ail.judge.<name>.aligned = "false"` (the scheduled-scorer API exposes no
per-scorer metadata slot). This dovetails with the agreement floor: an unaligned
judge has not been measured against the Human Anchor, so it reads as
`distrusted` (fail-closed, `insufficient_data`) until it is aligned **and**
audited. It is registered and visible — it scores — but the loop does not trust
its numbers until the labeling + alignment + agreement cadence has run.

### Labeling — `ail.judges.labeling` (so MemAlign has input)

MemAlign aligns *against human labels*; this module records them and assembles
the two **disjoint** pools the rest of the layer consumes. It is the input stage
that unblocks alignment on an experiment with no labels.

```python
from ail.judges.labeling import TraceLabel, record_labels, assemble_pools

labels = [
    TraceLabel(trace_id="tr-1", name="token_efficiency", value=2,
               rationale="re-read foo.py 34x for no gain",
               inputs=build_token_efficiency_inputs(metrics_1, task="refactor X"),
               outputs="<agent response>"),
    # ... ~30-50 human-graded traces ...
]
record_labels(labels, labeler_id="austin")                # HUMAN assessments on traces
alignment_set, anchor = assemble_pools(source, labels, judge_name="token_efficiency")
```

- `record_label` / `record_labels` — write each label to its subject trace as
  an MLflow **feedback** assessment (`mlflow.log_feedback`, `source_type=HUMAN`,
  `name=` the judge name MemAlign will align), plus any `expectations` as
  **expectation** assessments (`mlflow.log_expectation`, also `HUMAN`). This is
  the §11 feedback-attachment model: a human `token_efficiency` (`HUMAN`) and a
  judge `token_efficiency` (`LLM_JUDGE`) coexist on one trace, keyed by
  `(name, source_type)`.
- `split_labels(labels, *, anchor_fraction=0.3, seed=0)` — deterministically
  partitions labels into `(alignment_labels, anchor_labels)` **by trace** (every
  label of a trace lands in one pool), so a trace can never be in both the
  Alignment Set and the Human Anchor.
- `to_alignment_set(source, labels, *, labeler_id="expert") -> AlignmentSet` —
  fetches the **raw** MLflow traces via the ingest seam and **attaches each
  label's value** onto the trace as a `HUMAN` `Feedback` (`trace.info.assessments`),
  because MemAlign reads its feedback there and the re-fetched raw trace does not
  reliably carry it (alignment otherwise fails with *"No valid feedback records
  found"*).
- `to_human_anchor(labels, *, name=None, source=None) -> HumanAnchor` — builds
  anchor items (one per trace for a given judge `name`) carrying the human label +
  the `inputs`/`outputs`/`expectations` to re-run the judge. Pass `source` to also
  carry each item's **raw trace** (without the gold label), so a `{{ trace }}`-based
  judge can be scored on the anchor.
- `assemble_pools(source, labels, *, judge_name=None, anchor_fraction=0.3, seed=0) -> (AlignmentSet, HumanAnchor)`
  — does the split, builds both pools, and calls `assert_pools_disjoint(...)` to
  **prove** no trace id leaked across the `Pool`-keyed wall before returning.

Workflow for a human: label ~30–50 traces (`record_labels`), `assemble_pools`,
`create_aligned_scorer(spec, alignment_set=...)` to align-and-register, then
`score_anchor(reg.judge, anchor)` to audit the aligned judge against the held-out
anchor. Mixing the two pools is impossible by construction and re-proven on
assembly — measuring agreement on the labels the judge was aligned against would
only report memorization, the co-adaptation §2 forbids.

## Output contract — `AgreementReport` (`l2.judges/v1`)

Produced by `compute_agreement` / `score_anchor`, serialized verbatim by
pydantic (`model_dump_json()`); models set `extra="forbid"` so drift is loud.
This is what the Phase-4 leaderboard's "judge-human agreement trend + drift
alarm" reads.

```jsonc
{
  "schema_version": "l2.judges/v1",
  "judge_name": "correctness",
  "pool": "human_anchor",            // always the Human Anchor
  "n_items": 20,                     // total anchor items
  "n_scored": 19,                    // items the judge produced a value for
  "n_agreements": 16,
  "agreement_rate": 0.8,             // n_agreements / n_items (floor applies here)
  "floor": 0.7,
  "distrusted": false,               // true when agreement_rate < floor OR insufficient_data
  "insufficient_data": false,        // true when n_scored < min_samples (unmeasured judge)
  "cohen_kappa": 0.61,               // chance-corrected agreement; null when N/A
  "numeric_tolerance": null,         // set only when float labels used a tolerance
  "label_space": ["no", "yes"],
  "items": [
    { "item_id": "t1", "human_value": "yes", "judge_value": "yes", "agree": true,  "error": null },
    { "item_id": "t2", "human_value": "no",  "judge_value": null,  "agree": false, "error": "judge produced no value" }
    // ...
  ],
  "generated_at": "2026-06-29T00:00:00+00:00",
  "notes": [ /* human-readable caveats (errored items, float-tolerance, empty anchor) */ ]
}
```

Notes:

- **The floor is applied to the raw `agreement_rate`**, which is over *all*
  items — an item the judge could not score (`error` set) counts as a
  non-agreement, so a judge that crashes on half the anchor cannot look fully
  trustworthy.
- **Fail-closed on insufficient data.** An empty anchor — or any anchor with
  fewer than `min_samples` *scored* items — leaves the judge *unmeasured*. An
  unmeasured judge must never read as trusted, so `insufficient_data` is set and
  `distrusted` fires regardless of the (vacuous) rate. A consumer that only
  reads `distrusted` therefore stays safe; `insufficient_data` is the extra bit
  that distinguishes "could not measure" from "measured and failed the floor".
  This is the anti-co-adaptation correction to an earlier false-clear where a
  zero-item anchor reported `distrusted: false`.
- **`cohen_kappa`** is the chance-corrected companion (raw agreement is inflated
  by class imbalance, which matters for a guardrail). It is `null` when
  undefined/uninformative (no pairs, a single label space) or when float labels
  were compared with a tolerance.

## `AlignmentReport` (`l2.judges/v1`)

The serializable record of one MemAlign cadence (the aligned `Judge` object is
returned alongside it on `AlignmentOutcome.judge`):

```jsonc
{
  "schema_version": "l2.judges/v1",
  "base_judge_name": "correctness",
  "pool": "alignment_set",
  "optimizer": "MemAlign",
  "n_alignment_traces": 24,
  "aligned": true,
  "generated_at": "2026-06-29T00:00:00+00:00",
  "notes": [ "aligned on the Alignment Set only ...; alignment cadence is decoupled from agent optimization." ]
}
```

The same shape records the **unaligned** case (`unaligned_report`, used by
`create_aligned_scorer` when no labeled set is supplied): `aligned: false`,
`n_alignment_traces: 0`, and a note explaining the judge is flagged
not-yet-trusted until labels exist. This is the authoritative `aligned` flag the
loop reads (the experiment tag `ail.judge.<name>.aligned` is its best-effort,
queryable companion).

## Resolved MLflow version

Built and verified against **MLflow `3.14.0`** (satisfies the `mlflow>=3.14,<4`
pin; no bump required). The GenAI APIs used — `make_judge`, `Judge.align`,
`MemAlignOptimizer`, the `mlflow.genai.scorers` registration surface
(`Scorer.register`/`start`, `ScorerSamplingConfig`, backed by
`databricks-agents` ≥ 1.11), and the assessment-logging API
(`mlflow.log_feedback` / `mlflow.log_expectation` with an
`AssessmentSource(source_type=HUMAN)`) — are all present in 3.14.0.
```
