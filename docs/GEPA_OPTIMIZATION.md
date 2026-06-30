# GEPA agent-optimization loop (Stage 5)

This is the engine that turns **evaluation feedback into an automatically-evolved
agent artifact** — the `optimize/` step of the loop (`docs/ARCHITECTURE.md` §1, the
`gepa_runner` box in §4). It wraps [GEPA](https://github.com/gepa-ai/gepa) (a
reflective prompt-evolution optimizer, installed via `dspy`) around the **existing**
frozen-suite comparison machinery so that the objective GEPA climbs is *our*
objective — a deterministic, fail-closed token-reduction-with-correctness-held
score — not a vibe.

- **Module:** `src/ail/optimize/gepa_runner.py` (offline-tested seam)
- **CLI:** `scripts/run_gepa_optimization.py` (live, guarded by `AIL_LIVE_GEPA=1`)
- **Artifact under evolution:** the token-efficiency **skill body** — the same
  markdown the Phase-2 lever (`src/ail/optimize/lever.py`) injects into a candidate's
  system prompt via `SkillInjectionIntervention`.

```
frozen Task Suite
      │  split_suite()  (deterministic, seeded)
      ├── TRAIN split ──────────────► GEPA (gepa.optimize)
      │                                  │  proposes a new skill body
      │                                  ▼
      │             FrozenSuiteGepaAdapter.evaluate()    ← fitness = our harness
      │                run_phase2_comparison(task)       (L0 token + L1 correctness)
      │                fitness_from_outcome() → score    fail-closed
      │                                  │
      │             make_reflective_dataset()            ← L0/L1 feedback → reflection LM
      │                                  ▼
      │                       best evolved skill body
      │
      └── HELD-OUT split ──► live harness only (final validation) ──► report vs seed
                                                                          │
                                                              GepaOptimizationResult
                                                              (CANDIDATE — human gate)
```

## Why `gepa.optimize`, not `mlflow.genai.optimize_prompts`

Both wrap GEPA (MLflow's `optimize_prompts` delegates to a `GepaPromptOptimizer`
internally). We call the general-purpose `gepa.optimize` ("optimize anything") with a
custom `GEPAAdapter`, for two reasons specific to this loop:

1. **Our fitness is a two-arm, fail-closed comparison — not a scorer over one
   output.** `mlflow.genai.optimize_prompts` scores `predict_fn(inputs)` against a
   reference `outputs` column using `Scorer` objects (`Correctness`, `Equivalence`,
   …). Our objective runs **both** a baseline arm and a candidate arm, gates on
   execution success **and** L1 programmatic correctness *non-regression*, and only
   then rewards a strict token reduction. And the frozen suite carries **no
   human-authored expectations** to populate an `outputs` column — that is exactly
   why the harness runs under `NO_LLM_JUDGE` (see `src/ail/compare/harness.py`). The
   `GEPAAdapter.evaluate(batch, candidate)` seam lets us call
   `run_phase2_comparison` directly and return one fail-closed fitness float per
   task. There is no natural way to express "also run a baseline arm and fail
   closed" through a scorer/`expectations` shape.

2. **The artifact is an injected skill body, not a registry prompt template.**
   `optimize_prompts` requires the prompts to live in the MLflow Prompt Registry and
   a `predict_fn` that calls `PromptVersion.format` at inference. Our artifact is a
   free-text blob wired through `SkillInjectionIntervention`; `gepa.optimize`'s
   `seed_candidate: dict[str, str]` maps onto "evolve this text" one-to-one, with no
   registry round-trip. The natural second component is an agent
   system-prompt/instructions string — also just a named text entry in the same
   dict.

No capability is lost by going direct: we still pass `reflection_lm` and we control
the reflective-dataset shape ourselves.

## The fitness: our frozen-suite objective, fail-closed

GEPA requires a **per-example, higher-is-better** score. We compute it from the
harness's own decision — no new scoring logic:

`FrozenSuiteGepaAdapter.evaluate(batch, candidate)` → for each train task:

1. Build a candidate `LeverConfig` that injects the candidate skill body
   (`candidate_lever_config`).
2. Run `run_phase2_comparison(suite, adapter, candidate=…, task_ids={task})` — the
   **unchanged** Phase-2 harness: baseline arm + candidate arm, L0 deltas, execution
   guardrail, and the deterministic **L1 programmatic** correctness guardrail
   (`NO_LLM_JUDGE` — no uncalibrated judge in the decision path).
3. Map the resulting `TaskOutcome` → fitness with `fitness_from_outcome`:

   | Harness outcome | Fitness |
   | --- | --- |
   | `PROMOTE` (objective met **and** all guardrails passed) | realized token-reduction fraction in `(0, 1]` |
   | `BLOCK` — execution failed | `0.0` |
   | `BLOCK` — L1 correctness regressed / no verdict / not configured | `0.0` |
   | `BLOCK` — no token reduction | `0.0` |
   | no outcome produced | `0.0` |

This is **fail-closed**: a candidate that breaks L1 correctness, crashes, or fails to
reduce tokens scores `0.0`. The only way to score above zero is the harness's own
`PROMOTE` — and the more tokens it cuts (with correctness held), the higher the
score. GEPA sums minibatch scores for acceptance and averages over the (train)
valset for tracking; both are honest because the score *is* the realized,
deterministic reduction.

**Feedback → reflection.** `make_reflective_dataset` turns each task's L0 token delta
+ L1 correctness outcome + harness decision (with blocking reasons) into the feedback
string the reflection LM reads, so the next proposed body is shaped by *why* a
candidate did or did not win — not by a bare number. Reflective mutation uses
`reflection_lm` (default `databricks:/databricks-claude-sonnet-4-6`, normalized to
litellm's `databricks/<model>` form at the GEPA boundary).

## The anti-overfit wall (load-bearing)

`docs/ARCHITECTURE.md` §2: the Task Suite is *never fed to the optimizer*, because an
agent that trains against its own benchmark co-adapts and the reported gain lies. We
preserve that principle by **splitting** the frozen suite (`split_suite`):

- **TRAIN split** — a working set deliberately carved out for GEPA to optimize
  against. This is *not* the held-out wall.
- **HELD-OUT split** — the frozen wall. GEPA never touches it; it is scored
  **only** by the live harness *after* optimization, and the reported headline is the
  evolved artifact's held-out result vs the seed artifact's held-out result
  (`GepaOptimizationResult.holdout_savings_delta_pct`).

Two guarantees keep GEPA off the held-out tasks:

1. **At the call boundary** — only `split.train_tasks` is handed to `gepa.optimize`,
   as *both* `trainset` and `valset`. The held-out tasks are never passed in, so the
   optimizer has nothing held-out to evaluate.
2. **Structurally, inside the fitness function** — `FrozenSuiteGepaAdapter.evaluate`
   raises `HeldOutLeakError` if it is *ever* asked to score a held-out (or otherwise
   non-train) task id, and records every id it does score in
   `evaluated_task_ids`. The wall is a hard failure, not a convention.

The split is deterministic (seeded shuffle of sorted task ids, or an explicit
`holdout_task_ids` list) and disjoint by construction (`SuiteSplit.assert_disjoint`).

### The test that proves it

`tests/test_gepa_runner.py` proves the wall is only ever called with train tasks:

- `TestSplit` — train ∩ held-out = ∅ and train ∪ held-out = the suite (no cap);
  capping train leaves the dropped tasks *unused*, never moved to held-out.
- `TestFitnessWall::test_evaluate_raises_on_held_out_task` /
  `…_on_non_suite_task` — the fitness function raises `HeldOutLeakError` on a
  held-out or stranger id; `…_records_only_train_ids` — a normal call records only
  train ids.
- `TestLoopOnlyEvaluatesTrain::test_gepa_is_handed_only_train_tasks` — end-to-end
  (with a fake `gepa.optimize` that exercises the real fitness path): GEPA received
  only train tasks as `trainset`/`valset`, and the adapter's `evaluated_task_ids` is
  a subset of train and **disjoint** from held-out.

## The human gate

`run_gepa_optimization` returns a `GepaOptimizationResult` carrying the **candidate**
evolved body + its live held-out result vs the seed's. It does **not** write the
skill to disk, register it, or promote it. `human_gate_required` is always `True`;
the object has no side effect that applies the artifact. Promotion is a separate
human step — review the held-out delta, then apply the evolved body to the asset
(`src/ail/optimize/assets/skills/token-efficient-execution/SKILL.md`) deliberately.
`tests/test_gepa_runner.py::TestReturnsCandidateNotPromotion` asserts the loop never
mutates the on-disk skill asset.

## Cost / fidelity

Every fitness evaluation runs the agent **live** — a baseline arm **and** a candidate
arm per train task — and the reflection LM is called for every proposed mutation. So
cost is bounded and configurable through `GepaConfig`:

| Dial | Effect |
| --- | --- |
| `max_metric_calls` | GEPA's total evaluation budget — the dominant cost dial. |
| `holdout_fraction` / `holdout_task_ids` | sets the train size (the per-iteration cost). |
| `max_train_tasks` | caps how many train tasks GEPA actually optimizes against. |

**Known cost — baseline re-run.** Because fitness reuses the two-arm
`run_phase2_comparison`, the baseline arm is re-run for *every* candidate evaluation
even though it does not change. We accept this rather than restructure the harness
(which would break the per-arm isolation contract the comparison depends on); the
budget dials above bound the total. A future optimization could cache the baseline
per task.

**Cheaper proxy fitness (optional).** A cheaper proxy agent may drive the GEPA inner
loop via `run_gepa_optimization(train_adapter=…)` while the **final** selected
candidate is always validated on the live `adapter` over the held-out split — so the
reported number keeps full fidelity even if the inner search used a proxy.
`tests/test_gepa_runner.py::…test_proxy_train_adapter_drives_inner_loop_live_adapter_validates`
pins this: the proxy runs the train tasks, the live adapter runs only the held-out
tasks.

## Running it (live)

```bash
AIL_LIVE_GEPA=1 python scripts/run_gepa_optimization.py \
    --suite-version phase2-mini \
    --experiment /Shared/dais-demo-agent-improvement \
    --profile dais-demo \
    --run-plan run_plan.yaml \
    --holdout-id ts-route-05 --holdout-id ts-config-04 \
    --max-metric-calls 60 \
    --output artifacts/gepa_candidate.json
```

The run plan maps task ids → L1 verification commands (same format as
`scripts/run_phase2_comparison.py`); a task with no entry has no correctness signal
and scores zero (fail-closed). The reflection LM uses litellm's Databricks provider,
so the environment needs Databricks credentials (`DATABRICKS_HOST` +
`DATABRICKS_TOKEN`, or a configured `--profile`).

The output `GepaOptimizationResult` JSON carries: `evolved_skill_body` (the
candidate), `seed_skill_body`, `changed`, the `train_task_ids` / `holdout_task_ids`
split, GEPA metadata (`gepa_total_metric_calls`, `gepa_num_candidates`,
`gepa_best_val_score`), and the two live held-out artifacts
(`holdout_evolved`, `holdout_seed_baseline`) whose realized savings the human gate
compares.
