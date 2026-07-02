# Judge Authoring

**Status:** stable · **Module:** `src/ail/judges/authoring.py` · **CLI:** `ail-author-judge`

The judge-authoring capability is the **"human explains what they're looking for
→ we make a judge for it"** front door to the L2 layer. A user describes, in plain
language, a quality dimension they care about — *"did the agent actually answer
the question?"*, *"did it follow the user's instructions?"* — and gets back a
**registered, MemAlign-alignable LLM judge** plus the **matching label schema** a
human labels against.

It is **additive**. It does not rebuild the judge scaffolding
(`docs/L2_JUDGES_CONTRACT.md`); it **composes** it. An authored judge is an
ordinary [`ScorerSpec`](../src/ail/judges/scorers.py) built with `make_scorer`
and registered through the existing `create_aligned_scorer` path, so everything
the L2 layer already does — MemAlign alignment, agreement auditing against the
Human Anchor, scheduled scoring — works on an authored judge unchanged.

## The two hard conventions (guaranteed for you)

These come from a documented MemAlign review. Getting either wrong **silently
breaks alignment** — the judge learns from nothing and no error is raised. The
authoring path guarantees both by construction, so a user never has to know them.

### 1. The judge is a `{{ trace }}`-template judge

The authored rubric embeds the MLflow `{{ trace }}` template variable. That is
what makes the judge **MemAlign-alignable**: `align_judge` learns from human
feedback attached to the *real traces* the `{{ trace }}` judge reads, so a
`{{ trace }}` judge yields genuine training examples.

Authoring a judge on **app-computed / derived inputs** instead (e.g. feeding it a
pre-summarized dict) would yield **zero** MemAlign training examples — the exact
mistake this capability exists to avoid. The one deliberate exception in this
codebase is `token_efficiency` (see [the exclusion](#tokencost-stays-deterministic-l0)).

Structurally, an authored judge is what `align_judge` / `score_anchor` accept:
`judge.get_input_fields()` includes a `trace` field, and `judge.align(...)` works.

### 2. The label schema's name **exactly** matches the judge name

`align()` pairs a human's feedback to a judge's scores by **matching the
label-schema `name` to the judge `name`**. A mismatch (e.g. judge
`answer_helpfulness` vs. schema `helpfulness`) means MemAlign finds no feedback
for the judge and alignment quietly does nothing.

Authoring removes the footgun: both the judge name and the label-schema name are
derived from a **single** `normalize_judge_name(...)` (canonical lowercase
`snake_case`), and `create_matching_label_schema(...)` calls
`mlflow.genai.label_schemas.create_label_schema(name=<judge_name>,
type="feedback", ...)`. The pairing therefore holds **by construction** — it is
asserted in the tests and printed by the CLI.

## Usage

### Python

```python
from ail.judges import author_judge

authored = author_judge(
    "Answer Helpfulness",                                  # -> judge name 'answer_helpfulness'
    "Did the agent actually answer the user's question, completely and usefully?",
    experiment_id="660599403165942",
    scale="1-5",                                           # or "pass_fail"
    profile="dais-demo",
)

authored.spec           # the reusable ScorerSpec (rubric + constrained output type)
authored.judge          # the registered, {{ trace }}, MemAlign-alignable Judge
authored.label_schema   # the created LabelSchema; .name == authored.spec.name
authored.registration   # ScorerRegistration provenance (aligned flag + report)
```

Preview without registering (no `databricks-agents` needed) — build the judge and
create the label schema, e.g. to review the rubric first:

```python
authored = author_judge(name, description, experiment_id=exp, register=False)
print(authored.spec.instructions)   # inspect the authored rubric
```

### CLI

```bash
ail-author-judge answer_helpfulness \
    --description "Did the agent actually answer the user's question, completely and usefully?" \
    --experiment-id 660599403165942 --profile dais-demo

# Preview only (no scheduled scorer, no agents extra):
ail-author-judge answer_helpfulness -d "..." --no-register
```

Key flags: `--scale {1-5,pass_fail}` · `--model <judge-model-uri>` ·
`--sampling-rate` · `--no-register` · `--refine` (+ `--refine-endpoint`) ·
`--overwrite-label-schema`. Registration needs the `agents` extra
(`pip install 'ail[agents]'`); the CLI fails closed with guidance if it is
missing.

## From description to gradeable rubric

Turning the NL description into a **concrete, gradeable** rubric is a
**deterministic template** (`build_instructions`). It:

- embeds `{{ trace }}` and states the dimension + the user's criteria;
- tells the judge to read the trace and judge **only** this dimension; and
- appends a **bounded** output rubric with a **required one-line rationale** that
  must name the specific trace evidence — so a verdict is actionable, not a bare
  number.

Two output shapes, both **constrained** structured outputs (the judge can never
emit an out-of-domain value):

| `scale` | judge output type | label-schema input | when |
|---|---|---|---|
| `"1-5"` (default) | `Literal[1,2,3,4,5]` (aggregations `mean/median/p90` restored) | `InputNumeric(1, 5)` | graded dimensions where gradations matter |
| `"pass_fail"` | `Literal["pass","fail"]` | `InputPassFail("pass","fail")` | a hard categorical guardrail |

The label schema is created with `enable_comment=True` so the human can record
the same one-line rationale the judge is asked to produce.

### Optional LLM refinement (behind a flag)

For vague criteria, an **optional single LLM pass** (`refine=True`) sharpens the
description before templating. It is deliberately minimal: one call through the
injectable `CriteriaRefiner` seam (default: a Databricks chat endpoint resolved
from `--refine-endpoint` / `AIL_JUDGE_AUTHOR_LLM_ENDPOINT`), and it falls back to
the original text if the model returns nothing. Every test injects a mock refiner;
no model is called unless `refine=True`.

> **Seam — adversarial Designer/Critic refinement.** A future lane will add an
> adversarial loop that iteratively hardens the rubric. It plugs into the **same**
> `CriteriaRefiner` seam (or replaces `refine_criteria`). It is **not** built now —
> the single-pass seam is the clean insertion point.

## Large traces (v1 scope + the digest seam)

A `{{ trace }}` judge is **context-bound**: it must fit the whole trace in the
judge model's context window. v1 deliberately **scopes to judge-ingestible
traces**.

> **Seam — RLM/HALO digest.** For very large traces, a future lane will feed the
> judge a **digest** of the trace instead of the raw trace. That substitution
> happens at the trace-**feeding** boundary (whatever supplies the `{{ trace }}`
> value at score time — `score_anchor` / the scheduled scorer), **not** in the
> authored rubric, so the authored judge does not change. The digest wiring is
> intentionally **not** built here; the `{{ trace }}` template (`TRACE_TEMPLATE_VAR`)
> is the single documented slot it will occupy. Nothing here precludes it.

## token/cost stays deterministic L0

Authoring is the general route for **human-defined QUALITY dimensions**. It does
**not** cover token/cost, which stay **deterministic L0** and are the one
deliberate exception to convention 1:

- The `token_efficiency` scorer (`ail.judges.scorers`, see `scorers.py:341`) is
  intentionally a **computed-inputs** judge: it reads an already-computed L0
  summary (`build_token_efficiency_inputs`) via `{{ inputs }}`, **not**
  `{{ trace }}`. It is therefore **deliberately not MemAlign-aligned** — the
  un-gameable L0 deterministic layer (`ail.metrics`) already covers token/cost, so
  there is nothing subjective for MemAlign to learn.
- This is by design, not an oversight. **Computed-inputs judges are deliberately
  not aligned.** Authoring leaves that scorer and its behaviour untouched and adds
  the general `{{ trace }}` authoring path alongside it.

## What authoring reuses vs. adds

**Reuses** (unchanged): `make_scorer` / `make_judge` (judge construction),
`ScorerSpec` (the spec shape + the aggregation-restore rule), `create_aligned_scorer`
/ the whole registration + align-then-register flow, and the MLflow backend
configuration seam. `align_judge`, `score_anchor`, and the agreement/alignment
contracts are untouched.

**Adds**: `src/ail/judges/authoring.py` (`normalize_judge_name`,
`build_instructions`, `build_judge_spec`, `refine_criteria`,
`create_matching_label_schema`, `author_judge`, `AuthoredJudge`), the
`ail-author-judge` CLI (`src/ail/jobs/author_judge.py`), its tests
(`tests/test_judges_authoring.py`), and this doc.

## Related

- `docs/L2_JUDGES_CONTRACT.md` — the L2 layer: scorers, the three-pool discipline,
  `align_judge`, `score_anchor`, the agreement contract.
- `docs/MEMALIGN_ROLLBACK.md` — the live MemAlign alignment flow and the
  `{{ trace }}`-judge context (token-cap) that motivates the large-trace seam.
- `src/ail/judges/labeling.py` — recording the human labels (under the
  name-matched schema) and assembling the disjoint Alignment Set / Human Anchor
  that align and audit the authored judge.
