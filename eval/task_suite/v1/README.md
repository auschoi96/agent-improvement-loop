# Task Suite — `v1-seed`

The **frozen, held-out benchmark** of the evaluation wall
(`docs/ARCHITECTURE.md` §2). These tasks are re-run to compare agent versions.
The optimizer and the judge-alignment path are **never** allowed to train
against them — that is the load-bearing anti-co-adaptation guarantee. If the
optimizer could see these tasks, "improvement" would be measured against the
very thing being optimized, and the number would lie.

- **Artifact:** [`tasks.yaml`](./tasks.yaml) — load with
  `ail.task_suite.load_task_suite("v1")`.
- **Curated source:** `src/ail/task_suite/seed.py` (`build_seed_suite()`).
  `tasks.yaml` is the *frozen serialization* of that builder; a test pins that
  they agree, so the artifact cannot drift from its reviewed source.
- **Content version:** `v1-seed` · **schema version:** `ail.task_suite/v1`.

## How it was seeded

The 22 tasks are derived **only** from committed artifacts:

- `artifacts/example1_diagnosis.md` / `.json` — the token-waste diagnosis.
- `artifacts/l0_baseline_660599403165942.json` — the deterministic L0 baseline
  over experiment `660599403165942` (91 traces).

No live MLflow trace content was read. The v4 trace store routes reads through a
SQL warehouse this identity is **not** authorized for (reads fail with
`PermissionDenied`), so the prompts could not be taken from the raw session
inputs.

> `artifacts/labeling_set_40.json` is named in the Wave 1b brief but is **not**
> present in this branch's tree or history, so it was not used. The three
> artifacts above were sufficient to curate a representative suite.

### Categories (from the dominant diagnosis patterns)

| category | n | what it captures |
|---|---|---|
| `heavy_tail_high_token` | 8 | the bimodal tail — sessions of 250K–943K tokens, where the spend lives |
| `high_tool_call_volume` | 5 | outsized action counts (up to 156 tool calls), incl. one non-wasteful read-heavy contrast case |
| `repeated_target_boilerplate` | 5 | re-run shell prologue (`cd` up to 27×) and the same file edited up to 8× |
| `typical_short_session` | 4 | the low-median bulk (incl. a tool-free Q&A) — coverage so the optimizer is held to "do not regress these" |

Difficulty (`easy`/`medium`/`hard`) is set from each source session's magnitude
(tokens and tool-call count).

## ⚠️ Prompts are `v1-seed` reconstructions

Because raw trace content is not yet readable, each task's `prompt` is a
reconstruction of the work the session evidently did, derived from its
observable L0 profile (project path, tool mix, repeated targets). It is **not**
the verbatim session input. Each task's `notes` field records the hard metrics
(tokens, tool calls, cost, repeated targets) and what is derived vs. unknown, and
every task carries the real `source_trace_id` it was abstracted from.

**Enrichment path:** when warehouse access lands, the prompts can be replaced
with real trace content **without a schema change** — curate a new artifact
version (e.g. `v2`), re-freeze, and leave this frozen `v1-seed` in place for
provenance. A frozen artifact is never edited in place (see below).

## Freeze / immutability

`tasks.yaml` is **frozen** (`frozen: true`) and carries a `content_hash` over
its tasks. The contract enforces immutability two ways:

1. **In memory** — `TaskSuite` is a frozen Pydantic model; the mutation helpers
   (`with_task` / `with_tasks`) raise `TaskSuiteFrozenError` once frozen.
2. **On disk** — `load_task_suite` recomputes the hash and raises
   `TaskSuiteIntegrityError` if it disagrees with the stored one, so an edited
   frozen artifact fails closed. It **also** rejects any artifact that is not
   frozen: a persisted suite must be sealed, so flipping `frozen: true -> false`
   to dodge the hash check is itself treated as tampering. There is no load path
   that returns a suite without a verified hash.

The hash is tamper *detection* for accidental drift and an integrity seal, not a
cryptographic guarantee against a determined committer. The structural guarantee
that the optimizer / ground-truth path cannot reach this pool lives in
`ail.groundtruth` (`promote.py`'s `TaskSuiteProtectedError` and
`schema.validate_pool_membership`, which keeps the Task-Suite ground-truth set
empty) and is asserted by `tests/test_task_suite.py`.
