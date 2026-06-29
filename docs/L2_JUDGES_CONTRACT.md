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
and re-measured. A drifting judge is a distrusted judge.

## API surface

### Scorers — `ail.judges.scorers`

Thin, configurable wrappers over `mlflow.genai.judges.make_judge`. Building a
scorer calls **no model** (the judge calls its model lazily on `__call__`), so
construction is offline and free.

```python
from ail.judges import make_correctness_judge, make_modularity_judge, make_groundedness_judge

correctness = make_correctness_judge(model="databricks:/...")   # categorical "yes"/"no" — the Phase-2 guardrail
modularity  = make_modularity_judge(model="databricks:/...")    # graded int 1..5
groundedness = make_groundedness_judge(model="databricks:/...")  # categorical "yes"/"no"

feedback = correctness(inputs=task, outputs=response, expectations=expected)
```

- `ScorerSpec(name, instructions, feedback_value_type, description)` — a reusable
  scorer definition. The built-in set is `DEFAULT_SCORERS`
  (`correctness`/`modularity`/`groundedness`).
- `make_scorer(spec, *, model=None, instructions=None, feedback_value_type=..., name=None, inference_params=None)`
  — build a `Judge`, overriding rubric/type/name/model per call.
- `with_rubric(spec, instructions)` — a `ScorerSpec` copy with a tuned rubric.

`correctness` is the guardrail Phase 2 needs: a token-reduction intervention may
ship only if correctness does not regress, so it is categorical (clean
pass/fail). `modularity` is graded because structure quality has gradations.
`groundedness` is the anti-hallucination check against the provided context.

### Alignment (MemAlign) — `ail.judges.alignment`

```python
from ail.judges import align_judge, build_memalign_optimizer, MemAlignConfig, AlignmentSet

alignment_set = AlignmentSet.of(labeled_traces)        # Alignment Set pool only
outcome = align_judge(correctness, alignment_set)      # optimizer=None → MLflow's default MemAlign
aligned_judge = outcome.judge                          # better-aligned MLflow Judge
record = outcome.report                                # serializable AlignmentReport

# To configure MemAlign (requires the optional `dspy` dependency):
optimizer = build_memalign_optimizer(MemAlignConfig(retrieval_k=3, reflection_lm="databricks:/..."))
outcome = align_judge(correctness, alignment_set, optimizer=optimizer)
```

- `align_judge(judge, alignment_set, *, optimizer=None, generated_at=None) -> AlignmentOutcome`
  — wraps `judge.align(traces=..., optimizer=...)`. Raises `ValueError` on an
  empty set.
- `build_memalign_optimizer(config=None)` — constructs a configured
  `MemAlignOptimizer`. **Imported lazily**; raises a clear `ImportError` when the
  optional `dspy` dependency is absent (so importing this package, and CI, never
  require `dspy`).
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
  it with the human label, and delegates to `compute_agreement`. A per-item judge
  exception is captured (recorded as an `error`, counted as a non-agreement) so
  one bad item never aborts the measurement.
- `AgreementConfig(floor=0.7, numeric_tolerance=0.0, case_insensitive=True)`.
- `log_agreement(report, *, run_id=None) -> bool` — logs
  `judge_human_agreement` / `judge_distrusted` (and `judge_human_cohen_kappa`)
  as metrics plus the full report as a JSON artifact; best-effort, returns
  `False` rather than raising when no run/MLflow is available.

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
  "distrusted": false,               // true when agreement_rate < floor
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

## Resolved MLflow version

Built and verified against **MLflow `3.14.0`** (satisfies the `mlflow>=3.14,<4`
pin; no bump required). All three GenAI APIs used — `make_judge`, `Judge.align`,
and `MemAlignOptimizer` — are present in 3.14.0.
```
