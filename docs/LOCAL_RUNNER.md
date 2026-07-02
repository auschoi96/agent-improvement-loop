# Local Runner — drive the full optimization cycle on your machine

`ail-local-cycle` runs the **exact same** self-improvement optimization cycle as the
scheduled serverless job (`ail-optimization-cycle`), but on a laptop — because that is
the only place the cycle's **prover** can actually run.

The proof step (`ail.optimize.phase2.run_phase2_comparison`) drives the **Claude Agent
SDK** through `ClaudeCodeAdapter`, which needs a local Claude auth and a local
filesystem for its per-arm git worktrees. Serverless compute cannot run it; your
machine can. So the local runner is where the opt-in **Tier-2 verification**
(`docs/PRODUCT_ARCHITECTURE.md` §3) actually executes end-to-end: it **proposes with a
real proof**, and the app **approves**.

It reuses the whole spine unchanged — in-cycle RLM/HALO review → Lane A deterministic
rules + Lane B LLM planner → candidate builder → **real** frozen-suite prover →
readiness + judge gate → PENDING-proposal publish to the same
`agent_proposed_actions` Unity Catalog table the approval-queue app reads. The runner
adds only three things: env-based static-token auth, a reporting layer that prints
every step, and an explicit LLM-gateway thread. It **never** weakens the gate or
fabricates a proof.

---

## Prerequisites

1. **Install the package with the runner's extras** (from a checkout):

   ```bash
   pip install -e ".[claude,align,l3,agents]"
   ```

   - `claude` → the Claude Agent SDK for the prover (`claude-agent-sdk` is
     self-contained: it bundles its own `claude` binary; the deps are pure-Python
     `anyio`/`mcp`/`sniffio`, so there is **no Node/CLI to install**).
   - `l3` → HALO, for the in-cycle RLM review.
   - `align` / `agents` → judge alignment + scorer registration used elsewhere in the
     cycle.

2. **Authenticate Claude locally** (the prover arms run as *you*). Use your normal
   local Claude login; the SDK picks it up. If `claude-agent-sdk` is not importable the
   runner prints a loud warning at startup and every candidate fails closed (no
   proposals) — it never fabricates a proof.

3. **A static Databricks token + host matched to the experiment's workspace.** A full
   local prover run proves the whole frozen suite and routinely takes longer than an
   OAuth token's ~1h life, so **use a static PAT — not a `--profile` OAuth login**:

   ```bash
   export DATABRICKS_HOST=https://<workspace-host>
   export DATABRICKS_TOKEN=dapi...            # static PAT for that workspace
   ```

   The runner drops any ambient `DATABRICKS_CONFIG_PROFILE` so no span can fall back to
   OAuth mid-run, and **fails loud** if either variable is missing.

4. **A SQL warehouse you have `CAN_USE` on**, in that workspace — used for the readiness
   facts, the pending-proposal cost guard, and the UC publish:

   ```bash
   export AIL_WAREHOUSE_ID=<sql-warehouse-id>
   ```

5. **(Optional) An explicit LLM gateway** for the RLM/HALO review. By default the runner
   uses the workspace's Foundation Model serving endpoints
   (`$DATABRICKS_HOST/serving-endpoints`) authenticated by the same static
   `DATABRICKS_TOKEN`. Override with your own OpenAI-compatible gateway:

   ```bash
   export AIL_LLM_BASE_URL=https://<gateway>/...      # default: <host>/serving-endpoints
   export AIL_LLM_API_KEY=...                         # default: $DATABRICKS_TOKEN
   ```

   The Lane B planner uses MLflow's Databricks deploy client, which reads the same
   `DATABRICKS_HOST` / `DATABRICKS_TOKEN` — so both model callers ride the one static
   token.

---

## One-command invocation

```bash
ail-local-cycle \
  --experiment 660599403165942 \
  --warehouse-id "$AIL_WAREHOUSE_ID" \
  --judge-model databricks-claude-sonnet-4-6 \
  --confirm-goal
```

Or, from a checkout without installing the console script:

```bash
python scripts/run_local_cycle.py --experiment 660599403165942 \
  --warehouse-id "$AIL_WAREHOUSE_ID" --judge-model databricks-claude-sonnet-4-6 --confirm-goal
```

`--confirm-goal` (or `AIL_CONFIRM_GOAL=1`) is **required**: the controller refuses to
optimize an unconfirmed goal, so you assert you have reviewed it. The default goal is
`minimize total_tokens` by 30% relative; override with `--objective-metric`,
`--goal-direction`, `--goal-target`, `--goal-target-kind`, and repeatable
`--guardrail-judge name:threshold`. RLM sampling reuses the existing knobs
(`--max-results`, `--max-reviews`, `--sample-rate`, `--min-tokens`). Run
`ail-local-cycle --help` for the full list.

---

## What it prints

The runner surfaces every stage as it runs, in order:

1. **STEP 1/5 · IN-CYCLE RLM REVIEW** — per reviewed trace: token efficiency, waste
   score, and how many assets were recommended; failed reviews are shown honestly (a
   total RLM failure is non-blocking — the cycle continues over already-attached
   feedback and never fabricates a verdict).
2. **STEP 2/5 · FEEDBACK SIGNALS** — the objective value + baseline, the
   recurrence-ranked RLM-recommended assets, and the L0 redundant-read patterns the
   planners act on.
3. **STEP 3/5 · READINESS GATE** — the readiness tier, `can_prove_improvement`, scored
   coverage, and each judge's trust (agreement rate, distrusted or not).
4. **STEP 4/5 · PLAN → PROVE** — for each decision: which **lane** proposed it (Lane A
   deterministic rule, named; or Lane B LLM planner) and **why** (the trigger summary +
   trace refs); the **candidate** being proved; then the **proof** — baseline vs
   candidate per task, with the **token delta**, the **tool-call delta**, the L1
   **correctness** outcome, and PROMOTE/BLOCK — plus whether the improvement was proved
   and correctness held.
5. **STEP 5/5 · PROPOSALS WRITTEN** — each PENDING proposal written to
   `agent_proposed_actions` (id, action, proof headline, gate), **or** the explicit
   fail-closed statement that nothing cleared proof + gate (in which case the agent's
   slice is replaced with the empty set, clearing any superseded pending proposal).
6. **SUMMARY** — Lane A/B counts, any planner/RLM error, and every fail-closed **skip**
   with its reason (candidate not buildable, not proven on the frozen suite, or gate
   unmet) — so a run with no proposals still tells you exactly why.

---

## Fail-closed guarantees

A proposal is written **only** with a real passing proof and a cleared gate. The runner
reuses the controller's fail-closed logic verbatim and adds no bypass:

- A **crashed / errored** prover run raises → the runner prints the honest error and
  re-raises so the controller records a fail-closed skip; **no proposal**.
- A **timed-out or non-improving** run BLOCKs on the frozen suite (never PROMOTEs) → the
  proof does not clear → **no proposal**.
- An **unmet readiness wall or a distrusted certifying judge** → the gate blocks → **no
  proposal**.

The controller never applies a change; it only proposes.

---

## How proposals flow into the app's approval queue

The runner writes PENDING proposals via the **same** publish path the serverless job
uses (`ail.loop.publish_proposals.publish_agent_proposals`) to the **same** unified
table — `agent_proposed_actions` in
`austin_choi_omni_agent_catalog.agent_improvement_loop` (the `DEFAULT_CATALOG` /
`DEFAULT_SCHEMA`; override with `--catalog` / `--schema`). The write is an
agent-scoped atomic `REPLACE WHERE agent_name = '…'`, so it is idempotent and clears
any superseded pending proposal for that agent, and never touches another agent's rows.

The deployed approval-queue app reads that table **SELECT-only** to populate its
Proposals view. A human reviews the proposal — its **why** (trigger), **what** (change
diff/SQL), **proof** (token savings with correctness held), and **gate status** — and
approves or rejects it. The gated **apply** happens server-side on approval (lane 3),
not here: **local proposes with proof; the app approves.**
