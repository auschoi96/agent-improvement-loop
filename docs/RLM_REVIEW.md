# RLM (HALO) deep review — model, goal-steering, and the scheduled job

The **RLM reviewer** (L3, `docs/ARCHITECTURE.md` §3, §11) runs the adopted **HALO**
recursive-LM engine over a *whole* long trace to discover token waste and failure
modes a fixed single-call judge would miss, and attaches a structured verdict to the
subject trace as `rlm_*` `LLM_JUDGE` assessments. It runs under its **own** MLflow
trace so the reviewer's tokens are never summed into the subject trace's L0 cost.

This doc covers three Phase-1 decisions: **which model**, **how the review is
steered**, and **how it is scheduled**.

## 1. Model = `databricks-gpt-5-5-pro` (most powerful *viable*)

The RLM job's judge model defaults to `databricks-gpt-5-5-pro`.

- **Claude/Opus is blocked.** HALO always sends `parallel_tool_calls` on its model
  calls (`engine.model_config.ModelConfig.parallel_tool_calls` defaults to `True` and
  is projected onto every SDK call). Databricks Claude serving endpoints reject that
  parameter, so any `databricks-claude-*` judge fails. HALO's `ModelConfig` is an
  **external** library we do not edit, so we cannot suppress `parallel_tool_calls`.
- `gpt-5-6` **does not exist** on the gateway.
- `databricks-gpt-5-5-pro` is the top OpenAI-compatible (chat-completions) model the
  gateway offers, and OpenAI-compatible endpoints accept `parallel_tool_calls` — so it
  is the most powerful **viable** judge.

### The reasoning-effort prefix mismatch (and how we handled it)

HALO auto-maxes reasoning effort by **model-name prefix**
(`engine.model_config.max_reasoning_effort_for_model`): a **dotted** `gpt-5.5` (or
`gpt-5.4` / `gpt-5.1-codex-max`) prefix → `xhigh`; other `gpt-5`/o-series → `high`;
everything else → `None`. The Databricks alias is **provider-prefixed and hyphenated**:
`databricks-gpt-5-5-pro`. That string does not start with `gpt-5` **at all**, so HALO's
check returns `None` — the strongest effort would silently never apply.

We cannot edit HALO's `ModelConfig`, so we fix it in **our** config wrapper
(`ail.l3.reviewer`):

1. `resolve_reasoning_effort(model)` **normalizes** the alias to the form HALO's table
   recognizes — strip a leading `databricks-` segment and restore the dotted minor
   version (`gpt-5-5` → `gpt-5.5`) — then **delegates to HALO's own**
   `max_reasoning_effort_for_model`. So `databricks-gpt-5-5-pro` → `gpt-5.5-pro` →
   `xhigh`. We reuse HALO's family→effort table rather than hardcoding an effort, so we
   stay in lockstep with it; non-reasoning families (e.g. Claude) still resolve to
   `None` and get no effort parameter.
2. `build_engine_config` sets that resolved effort as an **explicit**
   `ModelConfig.reasoning_effort` override. HALO honors an explicit override ahead of
   its own auto-detection (`ModelConfig.effective_reasoning_effort`), so
   `databricks-gpt-5-5-pro` now actually runs at `xhigh`.

`reasoning_effort` is threaded (like `temperature`) through
`build_engine_config → run_halo_review → review_trace → run_continuous_rlm`, and the
job exposes `--reasoning-effort` for an explicit override. The input is normalized by
`normalize_reasoning_effort` at the job boundary **and** defensively in
`build_engine_config`: empty, `none`, and `auto` (any case) all mean "no override,
auto-resolve" and become `None` — so `--reasoning-effort none` reads as *auto* and still
yields `xhigh`, rather than injecting HALO's literal `effort=none` (which would disable
reasoning). A genuine unrecognized value (e.g. `banana`) still fails loud at HALO's
`ReasoningEffort` validation. Verified:
`max_reasoning_effort_for_model("databricks-gpt-5-5-pro")` is `None`, but
`resolve_reasoning_effort("databricks-gpt-5-5-pro") == "xhigh"` and the built
`ModelConfig.effective_reasoning_effort() == "xhigh"` for `None` / empty / `none` /
`auto` inputs alike.

## 2. Goal-steering (not a fixed rubric)

The reviewer scores against a `ReviewRubric`, whose `objective` is baked into HALO's
prompt (*"judge each guideline, and make every recommendation, in service of that
objective"*). Instead of only the fixed `DEFAULT_RUBRIC` (whose objective is "same task
quality, fewer tokens, lower latency"), the review can be **steered by the user's
compiled goal** (`ail.goals`, `docs/ARCHITECTURE.md` §4):

- `ail.l3.goal_rubric.rubric_from_goal(compiled_goal)` derives a `ReviewRubric` from a
  validated `CompiledGoal` — it **re-points** the objective at the user's goal
  (e.g. `reduce the agent's total tokens by 30% while not regressing correctness`) and
  stamps a goal-derived `rubric_id` (`ail.l3.goal/<metric>-<direction>/v1`).
- It **reuses** the base rubric's guidelines, score scale, and asset directive
  unchanged — only the steering objective changes — so verdicts stay comparable.
- `DEFAULT_RUBRIC` remains the fallback when **no** goal is configured.

The rubric is threaded through the existing seam
(`run_continuous_rlm(rubric=…) → review_trace → build_review_prompt`); the job builds
it from the same goal knobs the local companion planner uses (`--objective-metric`,
`--goal-direction`, `--goal-target`, `--goal-target-kind`, `--guardrail-judge`). An
empty `--objective-metric` keeps `DEFAULT_RUBRIC`. The `CompiledGoal` is fully
validated against the allowlist + readiness contract, so a misconfigured goal fails
loud rather than silently reviewing against a fabricated objective. This is a read-only
review (it only attaches assessments), so there is **no** confirmation gate.

The existing cost/safety guards in `ail.l3.continuous` are untouched: sampling,
idempotent skip of already-reviewed traces, the reviewer-trace skip, the per-run review
cap, and the fail-closed `rlm_review_failed` marker (a degenerate review is never
recorded as a fake-good pass).

## 3. Scheduled Databricks job

`resources/continuous_rlm.job.yml` re-establishes the standalone RLM reviewer as a
**scheduled** job (`ail-continuous-rlm` → `ail.jobs.continuous_rlm:main`). It is
deliberately **scheduled, not trace-arrival-triggered**: the UC-backed trace store is a
**VIEW** (`cc_trace_unified` / `cc_trace_metadata`), so a `table_update` trigger is
infeasible — the reason the original trigger job was retired. It mirrors the sibling
job resources (`l0_publish` / `auto_align`): serverless compute, the locally
built `ail` wheel + `halo-engine`, bundle-level `run_as`, `max_concurrent_runs: 1` with
queueing, and idempotency so it never re-reviews handled traces.

`databricks.yml` variables (defaults):

| var | default | purpose |
|---|---|---|
| `continuous_rlm_judge_model` | `databricks-gpt-5-5-pro` | HALO judge for the standalone job (most powerful viable) |
| `continuous_rlm_reasoning_effort` | `''` (auto) | explicit effort override; empty / `none` / `auto` ⇒ auto-resolve (→ `xhigh`) |
| `continuous_rlm_cron` | `0 */5 * * * ?` | schedule (every 5m) |
| `continuous_rlm_pause_status` | `UNPAUSED` | `PAUSED` ⇒ deployed-but-dormant |
| `continuous_rlm_timeout_seconds` | `7200` | hard bound per batch; queued firings continue unassessed traces |

It reuses the shared `rlm_*` sampling knobs. Registry mode reads each agent's own goal
from `agent_registry`, so reviews target the same goal as the companion without leaking
one agent's objective into another. Idempotency (`has_rlm_assessment`) prevents duplicate
successful reviews; never-attempted traces are selected before failed retries. Scheduled
serverless runs set `--code-sandbox=off`: HALO retains its trace-navigation and subagent
tools but skips the optional Pyodide probe that is unavailable on this runtime. Manual
runs can pass `--code-sandbox=auto` to restore HALO's normal sandbox discovery.

## Live smoke test

`tests/test_l3_reviewer.py::test_live_review_trace` runs an end-to-end review against a
real workspace, gated by `AIL_LIVE_HALO=1` plus `AIL_LIVE_EXPERIMENT_ID` /
`AIL_LIVE_TRACE_ID` / `AIL_LIVE_MODEL` (set `AIL_LIVE_MODEL=databricks-gpt-5-5-pro`);
`attach=False` so it never mutates the experiment. All other L3 tests mock HALO, the
model, and the trace/MLflow calls, so the suite runs fully offline.
