# Getting started — a first-time user guide

This guide walks you from zero to a running self-improvement loop on **your own**
agent. Read the [README](../README.md) for the one-paragraph pitch and
[`ARCHITECTURE.md`](ARCHITECTURE.md) for the full design; this doc is the
hands-on path.

---

## 1. The one mental-model shift: you *connect*, you don't *upload*

You cannot upload an agent and get optimization. The system optimizes **what it
can measure**, and it can only measure **what has been traced**. So the real
first step is *connect your trace stream* — point the loop at an MLflow
experiment your agent already logs to (or install the adapter so it does).

> **0 traces → 0 trustworthy optimization.** This is not a limitation to work
> around; it is the foundation. The loop will tell you it is "collecting — not
> ready yet" rather than show you a green dashboard that lies.

### What "ready" means (the readiness gates)

Different goals need different amounts of data. There is no single magic number;
the [readiness module](READINESS_AND_TRUST.md) computes this per goal and
**refuses to claim improvement until the gate is met**:

| Goal | What it needs before a claim is trustworthy |
|---|---|
| **Token / cost reduction** | **≥50** diverse traces to *prove* it (token distributions are heavy-tailed; a "50% cut" on 5 traces is noise); **≥10** is enough for a baseline + diagnosis + a frozen task suite to compare on |
| **Coding quality / accuracy** | a frozen task suite **+ ≥20 human labels** + a judge whose agreement with you is measured and above the floor |
| **Deep failure-mode discovery** | a few large traces (RLM/HALO review is diagnostic, never a leaderboard score) |

> These are the **code-enforced defaults** (`ReadinessThresholds` in
> `src/ail/readiness/compute.py`): `baseline_min_traces=10`,
> `quality_min_labels=20`, `prove_min_traces=50`, `scored_coverage_floor=0.5`.
> New here? Start with **[CONNECT_YOUR_AGENT.md](CONNECT_YOUR_AGENT.md)** — it's
> the Stage 0 "how do I get my agent's traces in (autolog or OTEL) and how many
> do I need" guide.

If you have zero traces, **expect no improvement claims until you have collected
enough** — and the app's readiness panel will tell you exactly how many more
traces / labels / a frozen baseline you still need.

---

## 2. Prerequisites

- **Python 3.11–3.13** and this repo installed (see step 3).
- A **Databricks workspace** with:
  - an **MLflow experiment** holding your agent's traces (this repo's reference
    experiment is `660599403165942`),
  - access to **model-serving endpoints** for the judge / reflection / embedding
    models (e.g. `databricks-claude-*`, `databricks-gte-large-en`),
  - a **SQL warehouse** you have `CAN_USE` on (the UC-backed trace store reads
    traces through a warehouse — see [§8 Troubleshooting](#8-troubleshooting--operational-notes)).
- For the GenAI/optimization features: the optional `align` extra (installs
  `dspy`, the MemAlign optimizer backend) and `databricks-agents`.

You do **not** need all of this on day one. L0 token/cost metrics work with just
trace access; the quality/optimization stages layer on as you add labels and a
warehouse grant.

---

## 3. Install

```bash
git clone https://github.com/auschoi96/agent-improvement-loop.git
cd agent-improvement-loop
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,align]'      # 'align' pulls dspy for MemAlign; omit if you only want L0
```

Authenticate to your workspace. For **interactive** use a CLI profile is fine:

```bash
databricks auth login --profile my-workspace
```

For **long or repeated runs** (RLM batches, GEPA, alignment), use a **static
bearer matched to the workspace** instead of a profile — the OAuth refresh is
flaky on long jobs and the endpoint host must match the token's workspace
(see [§8](#8-troubleshooting--operational-notes)):

```bash
export DATABRICKS_HOST="https://<your-workspace-host>"
export DATABRICKS_TOKEN="$(databricks auth token --profile my-workspace | jq -r .access_token)"
```

---

## 4. The first-time flow

The loop is six stages. You can stop after any stage — each delivers value. The
honest sequencing is: **prove the cheap/deterministic wins first (L0), and gate
every quality claim behind ground truth.**

### Stage 1 — Connect traces & see the L0 baseline *(no labels needed)*

L0 metrics are deterministic and un-gameable (tokens, cost, latency, tool-call
count, redundancy). This is your irrefutable baseline and it works immediately.

```bash
# Compute + publish L0 metrics from your experiment into UC Delta tables
python -m ail.publish --experiment-id <YOUR_EXPERIMENT_ID> \
    --catalog <catalog> --schema <schema>
```

- Code: [`src/ail/metrics/`](../src/ail/metrics) (the L0 contract,
  [`L0_METRICS_CONTRACT.md`](L0_METRICS_CONTRACT.md)) + `src/ail/publish.py`.
- This already reproduces things like the heavy-token tail and repeated-target
  tool calls — *where the waste is*, which drives Stage 5/6.
- Not sure you have enough data yet? Run `ail-readiness <YOUR_EXPERIMENT_ID>` for
  a one-command preflight of every readiness gate (see
  [CONNECT_YOUR_AGENT § Preflight](CONNECT_YOUR_AGENT.md#preflight-am-i-ready)).

### Stage 1b — Separate agents/cohorts with tags *(optional but recommended)*

If one experiment holds traces from several agents (e.g. Claude Code **and**
Codex), tag them so the loop treats them as distinct cohorts with their own
baselines and goals.

- Tag in the MLflow UI or via [`src/ail/cohorts.py`](../src/ail/cohorts.py);
  search with `tags.<your_key> = '<value>'`. See [`COHORTS.md`](COHORTS.md).
- Goals, readiness, metrics, and scoring all segment by cohort.

### Stage 2 — Register LLM-judge scorers

Register the L2 judges (correctness, modularity, groundedness, token_efficiency)
as scheduled scorers on your experiment.

- Code: [`src/ail/judges/`](../src/ail/judges) ([`L2_JUDGES_CONTRACT.md`](L2_JUDGES_CONTRACT.md)).
- After registration, `list_scorers(experiment_id)` shows them. **Note:** for
  them to actually *run* on new traces, the experiment needs a monitoring SQL
  warehouse wired (the [deploy](DEPLOY.md) flow automates this).

### Stage 3 — Create ground truth & align judges with MemAlign *(the quality unlock)*

A judge you have not validated against humans is **distrusted by default**. To
make a quality claim trustworthy you must:

1. **Label a slice of traces.** In the MLflow UI, open traces (filter by your
   cohort tag) and add human assessments named `correctness` / `modularity` /
   `groundedness` / `token_efficiency` (a 1–5 grade or pass/fail) **with a
   one-line rationale** — the rationale is what MemAlign learns from. ~15–30
   labeled traces is enough for a first alignment. You can also use the
   human-in-the-loop GRP pipeline in [`src/ail/groundtruth/`](../src/ail/groundtruth).
2. **Align.** Assemble disjoint pools, then align a judge against *your* labels:
   ```bash
   # illustrative; see src/ail/judges/ for the exact API
   #   assemble_pools(...)  -> align_judge(base, alignment_set, MemAlignOptimizer) -> register
   ```
3. **Watch judge-vs-human agreement move.** On the reference corpus this took a
   judge from **0.40 (distrusted) → 0.80 (trusted)**. Agreement is tracked as a
   first-class metric with a floor; a drifting judge is re-distrusted. See
   [`MEMALIGN_ROLLBACK.md`](MEMALIGN_ROLLBACK.md) (it also documents the
   `unalign` rollback and which metrics suit a `{{ trace }}` judge vs. a
   deterministic L0 rule).

> **Why this matters:** MemAlign aligns to *human* feedback, never to model
> output. Aligning a judge to another model's labels (e.g. the RLM's) and then
> optimizing the agent against that judge is the **co-adaptation trap** — scores
> climb while real quality stalls. The whole design exists to prevent this.

### Stage 4 — Recursive review of huge traces (RLM / HALO)

Coding-agent traces can be 500K–900K tokens — too long for a single judge call
or a human. The L3 reviewer uses HALO (a trace-specialized recursive LM) to
review them and emit a structured verdict + ranked **asset recommendations**.

```bash
AIL_LIVE_MLFLOW=1 python scripts/run_rlm_batch.py \
    --experiment-id <YOUR_EXPERIMENT_ID> --profile my-workspace
```

- Code: [`src/ail/l3/`](../src/ail/l3). The reviewer runs in its **own** trace so
  its tokens never pollute the agent's L0 numbers; the verdict attaches to the
  subject trace as an assessment.
- Output: per-trace `rlm_*` scores + an aggregate ranked list of recommended
  assets (the input to Stage 6).

### Stage 5 — Optimize the agent prompt/skill with GEPA

GEPA evolves the agent's prompt/skill against the **train split** of the frozen
task suite; fitness is the harness's own PROMOTE decision + realized token
reduction. It **never trains on the held-out split** and **never auto-promotes**
— it produces a human-gated candidate.

```bash
AIL_LIVE_GEPA=1 python scripts/run_gepa_optimization.py \
    --suite-version phase2-mini --run-plan run_plan.yaml \
    --holdout-id <task-a> --holdout-id <task-b> \
    --reflection-lm "databricks:/databricks-claude-opus-4-8" \
    --output artifacts/gepa_candidate.json
```

- Code: [`src/ail/optimize/gepa_runner.py`](../src/ail/optimize) ([`GEPA_OPTIMIZATION.md`](GEPA_OPTIMIZATION.md)).
- Review `artifacts/gepa_candidate.json` (`changed`, held-out vs. seed) and
  promote separately if it holds up. Live GEPA runs real agent sessions per
  fitness eval — keep the budget small; it is slow and billable.

### Stage 5b — Prove an improvement (WITH vs WITHOUT, fail-closed)

Before believing any lever, run the controlled comparison on the **frozen** task
suite: baseline (no intervention) vs candidate, in isolated per-arm workspaces,
correctness gated by deterministic L1 checks.

```bash
python scripts/run_phase2_comparison.py --suite-version phase2-mini --run-plan run_plan.yaml
```

- Code: [`src/ail/compare/`](../src/ail/compare) + [`src/ail/task_suite/`](../src/ail/task_suite)
  ([`PHASE2_LIVE_HARNESS.md`](PHASE2_LIVE_HARNESS.md)).
- It **fails closed**: a crashed candidate, a failed baseline, a missing
  verifier, or a sub-threshold reduction all **BLOCK** — a token "win" off a
  broken run is never counted. On the reference suite this proved a **35.4%
  token reduction with correctness held** (2 PROMOTE / 3 honest BLOCK).

### Stage 6 — Generate helper assets

Turn the RLM's ranked recommendations into a real, deployable Databricks asset —
e.g. a UC **metric view** for tool-call redundancy / token efficiency built from
the *real* L0 columns (with a fabrication guard: a measure with no backing column
is dropped-with-reason, never invented).

- Code: [`src/ail/optimize/assets/`](../src/ail/optimize/assets) ([`ASSET_GENERATOR.md`](ASSET_GENERATOR.md)).
- Generation + validation are static/offline; **deploying** the generated spec is
  a separate operational step you run.

---

## 5. State a goal in natural language

Rather than wiring metrics by hand, declare intent and let the goal compiler turn
it into a concrete objective + metrics + guardrails:

- Code: [`src/ail/goals/`](../src/ail/goals). "Reduce my token cost 30% without
  hurting correctness" → `{objective: total_tokens ↓30%, guardrail: correctness
  must not regress}`. A quality goal automatically lists its judge as a
  guardrail, so the readiness wall requires that judge to be measured.

---

## 6. See it in the app

The loop ships a Databricks App (AppKit) that reads the published L0 tables and
renders the leaderboard, the diagnosis, and (as you progress) the
WITH/WITHOUT comparison and judge-vs-human agreement trend.

- App: [`ail-self-optimizer/`](../ail-self-optimizer). Deploy with the bundle
  (see below). The reference deployment:
  `https://ail-self-optimizer-<id>.aws.databricksapps.com`.

---

## 7. Deploy the whole thing (turnkey, for reuse)

A privileged `databricks bundle deploy` provisions/uses a warehouse, runs the
app + jobs + scorers as a **single service principal**, grants it `CAN_USE`, and
sets the experiment's monitoring tag — so live scoring just works without manual
clicking.

```bash
databricks bundle deploy --target dev      # see databricks.yml + resources/
```

- See [`DEPLOY.md`](DEPLOY.md). **Caveat:** granting `CAN_USE` requires the
  deploying identity to have that authority (workspace admin or `MANAGE` on the
  warehouse). If you deploy as a non-privileged user, an admin runs the grant
  once. There is no bypass — this is the Databricks permission model.

---

## 8. Troubleshooting & operational notes

- **`401 / Credential ... unsupported` reading traces:** the UC-backed trace
  store serves reads through a SQL warehouse. Grant your identity `CAN_USE` on a
  warehouse, and pass `--warehouse`/`DATABRICKS_*` accordingly.
- **`403 Invalid Token` on a model call:** the endpoint host must match the
  token's workspace. A token minted for workspace A will 403 against workspace B
  even for the same model name. Mint the token from the **same** profile whose
  host you set in `DATABRICKS_HOST`.
- **`exit status 45` / OAuth refresh failures on long runs:** profile OAuth
  refreshes ~hourly and the in-process refresh is flaky. For long jobs use a
  **static bearer** (`DATABRICKS_HOST` + `DATABRICKS_TOKEN`) and leave
  `--profile` unset (clear `DATABRICKS_CONFIG_PROFILE`) so no refresh is
  attempted. The durable fix for deployments is the single-SP credential from
  [`DEPLOY.md`](DEPLOY.md).
- **Rate limits / region-unavailable models:** if one workspace throttles or
  lacks a model in-region, point the model flags at a workspace that serves it
  cleanly (the model name resolves per-workspace).
- **Live GEPA is slow:** each fitness eval is a real agent coding session that
  can hit a per-arm timeout; if both seed and candidate score 0 (e.g. the arm
  times out), GEPA correctly declines to promote. Keep budgets small and tasks
  fast.

---

## 9. Where each stage lives (map)

| Stage | Module | Script | Contract doc |
|---|---|---|---|
| 1. Ingest + L0 | `src/ail/ingest`, `src/ail/metrics`, `src/ail/publish.py` | `ail.publish` | [L0_METRICS_CONTRACT](L0_METRICS_CONTRACT.md) |
| 1b. Cohorts | `src/ail/cohorts.py` | — | [COHORTS](COHORTS.md) |
| 2. L2 judges | `src/ail/judges` | — | [L2_JUDGES_CONTRACT](L2_JUDGES_CONTRACT.md) |
| 3. Ground truth + MemAlign | `src/ail/groundtruth`, `src/ail/judges` | `scripts/demo_memalign_rollback.py` | [MEMALIGN_ROLLBACK](MEMALIGN_ROLLBACK.md), [READINESS_AND_TRUST](READINESS_AND_TRUST.md) |
| 4. RLM / HALO | `src/ail/l3` | `scripts/run_rlm_batch.py` | — |
| 5. GEPA | `src/ail/optimize/gepa_runner.py` | `scripts/run_gepa_optimization.py` | [GEPA_OPTIMIZATION](GEPA_OPTIMIZATION.md) |
| 5b. Comparison | `src/ail/compare`, `src/ail/task_suite` | `scripts/run_phase2_comparison.py` | [PHASE2_LIVE_HARNESS](PHASE2_LIVE_HARNESS.md) |
| 6. Assets | `src/ail/optimize/assets` | — | [ASSET_GENERATOR](ASSET_GENERATOR.md) |
| Goals | `src/ail/goals` | — | — |
| Readiness | `src/ail/readiness` | — | [READINESS_AND_TRUST](READINESS_AND_TRUST.md) |
| Deploy | `databricks.yml`, `resources/`, `ail-self-optimizer/` | `databricks bundle deploy` | [DEPLOY](DEPLOY.md) |

---

## 10. The trust guarantees (why it refuses to lie)

This system is built to distinguish *real* improvement from a dashboard that says
"improved." The non-negotiables, all enforced in code:

- **Frozen-eval wall** — the optimizer never trains on the held-out task suite.
- **Fail-closed everywhere** — an un-run, errored, or unverifiable evaluation is
  **never** a pass; a token win off a broken/failed run is never counted.
- **Distrusted-by-default judges** — a judge is untrusted until its agreement
  with human labels is measured above a floor.
- **Judge–agent decoupling** — judges are aligned on a separate cadence from
  agent optimization, against fresh human labels, to break co-adaptation.
- **Honest readiness** — per goal, the loop states what data is still missing and
  declines to claim improvement until the gate is met.

See [`READINESS_AND_TRUST.md`](READINESS_AND_TRUST.md) for the full risk register.
