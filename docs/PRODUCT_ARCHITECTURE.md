# Self-Optimizing Agents — Product Architecture

> **Status: source of truth (current planning round).** Where this conflicts with earlier
> design notes — `LOOP_CONTROLLER.md` (proving as a mandatory gate), the "unified optimization
> cycle" — **this document wins.** Those describe a pre-decision design that has been simplified.

## 1. What this is

A **reusable, self-deployable framework**. Anyone clones this repo and deploys their own
instance into their own Databricks workspace; their team then uses the **app UI** to set up and
monitor self-optimizing agents. It is **not** a hosted service someone runs for others — each
deployer runs their own instance.

> The `dais-demo` experiment (`660599403165942`), the `labeling_set=v1` traces, and the 5 sample
> fixtures in this repo are a **reference test instance only** — not part of the product.

## 2. The loop, in one breath

An agent's traces flow into Databricks. An **evaluation layer (LLM judges + RLM), shaped by the
user's stated goals,** scores every trace and writes the feedback onto the trace. A **local agent
reads that feedback and proposes concrete improvements — each with evidence — into the app.** The
user **approves** what they want; a **local agent executes it** (versioned, revertible). The user
**watches real before/after impact** and **reverts** anything that didn't help.

No mandatory proving, no cycle orchestrator, no automated promotion gates.
**The human is the gate; evidence informs the call.**

## 3. Core principle — the human decides on tiered evidence

| Tier | When | What it gives |
|---|---|---|
| **1 — Predict** | Always, cheap | Judge + RLM evidence: *why* a change is recommended |
| **2 — Verify** | Opt-in, on demand | Run the candidate vs. baseline on the **user's frozen suite** → a measured delta |
| **3 — Confirm** | After ship | Organic before/after by agent version + one-click revert |

Proving moved from a *mandatory gate* to *Tier-2 verification the human can choose to run* when
undecided. Everything ships behind human approval; real impact is measured after, and reverted if
it didn't help.

## 4. Compute planes

- **Databricks (model-only, hosted/scheduled):** judges, RLM, judge-authoring, MemAlign
  alignment, L0 metrics + publish, and **the app**.
- **Local companion (deployer-run, Claude Agent SDK):** the **planner, executor, and prover**.
  Runs on any normal machine the deployer controls (**not** Databricks serverless — the Agent SDK
  bundles its own runtime and needs to execute locally). Polls UC for work, writes results to UC.
- **UC Delta tables = the shared surface.** The app coordinates the companion via a request table:
  app writes a request → companion executes locally → writes the result to UC → app displays it.

## 5. Deployment — three components, stood up once by the deployer

1. **Databricks jobs** — `databricks bundle deploy` (judges / RLM / align / L0 publish).
2. **The app** — Databricks Apps (the shared control + monitoring UI).
3. **The companion** — started on deployer-controlled compute; should have a **one-command /
   one-container bootstrap** (Claude auth + Databricks auth, pointed at the workspace).

Per-deployment configuration is via **bundle variables**; the **agent registry +
experiment-per-agent** model gives multi-agent isolation within a deployment.

**Tenancy:** self-hosted, **per team / org** — deployed into their own workspace, serving their
own agents. Not a multi-org SaaS.

## 6. Evaluation layer (Databricks)

### Judges — a human-defined, growing set
- The user describes a quality dimension in natural language → a judge is **authored** for it
  (the judge-authoring capability).
- To be **MemAlign-alignable**, a judge must be a **`{{trace}}` template** (or a clean
  `{{inputs}}`/`{{outputs}}` mapping to the trace's request/response) — **not** app-computed
  inputs. MemAlign sources the judge's declared fields *from the trace*, so a judge scoring on a
  computed summary can't be aligned on the inputs it actually uses.
- The human **label's name must exactly match the judge name**, or `align()` cannot pair the
  feedback and alignment silently fails.
- Judges score on **judge-ingestible (smaller) traces**. For very large traces (100s of K
  tokens), feed the judge the **RLM digest** instead of the raw trace.
- **Token / cost is deterministic L0** — no LLM judge needed. Judges are for **quality**
  dimensions the human defines.

### Auto-align (MemAlign)
- A **scheduled check** counts human labels; when they cross the floor (~20 per dimension) with
  new labels since the last run, it **aligns** the judge and registers the aligned version.
- **Re-aligns over time** as more labels accrue; the **judge-vs-human agreement floor** guards
  trust (an unmeasured judge is distrusted); **rollback** if a re-alignment regresses.
- Scheduled, **not** event-triggered — the trace store tables are views (no table-update trigger).

### RLM (HALO)
- Reviews traces on a **schedule**, **steered by the user's stated goals** (parameterized, not a
  fixed rubric).
- Runs on **`databricks-gpt-5-5-pro` (GPT-5.5 Pro)** — the most powerful viable model. Opus/Claude
  is blocked (HALO always sends `parallel_tool_calls`, which Databricks Claude endpoints reject);
  GPT-5.6 does not exist on the gateway.

### Feedback
All judge scores and RLM findings are written **onto the traces** as assessments — accessible to
the app and to the labeling flow. (Reviewer runs stay in their own traces so the subject trace's
token metric is never polluted.)

## 7. Agent layer (local companion, Claude Agent SDK)

- **Planner** — reads the judge + RLM evidence → produces **evidence-backed proposals** into UC.
  Does **not** prove.
- **Executor** — an **open-ended, smart agent**. It reads the feedback, decides the best course of
  action, and does **whatever is needed**: create / adjust / delete tables, metric views, skills,
  tools, examples, caches, and anything else. **Not** a fixed action list. It produces a
  **versioned candidate change** (see §8). Safety is in the wrapper, not in limiting the agent:
  versioned candidate → optional prove → human approval → post-hoc measurement → revert.
- **Prover (opt-in, Tier 2)** — runs the candidate vs. baseline on the **user's frozen suite** for
  a measured delta when the human wants verification before deciding.

## 8. Versioning & revert — Databricks-native, no git

| Change | Mechanism | Revert |
|---|---|---|
| Prompt / skill / instruction **version** | **MLflow Prompt Registry** (new version + champion alias) | re-point the alias |
| **UC asset** (metric view, table, function) | Created directly in UC | drop / recreate |
| Arbitrary **file / code** change-set | Snapshot before/after to a **UC Volume** | restore the snapshot |

Everything stays in Databricks (visible to the app), nothing to install. **Tradeoff:** for a
complex multi-file *code* edit, a Volume snapshot is coarser than git (whole-set restore, not
line-level diff/revert). Git can be added later as an *option* for heavy code edits; the default is
Databricks-native. Proving needs no git — the prover pulls the candidate from the Registry/Volume
and runs it vs. the champion.

## 9. The user's frozen suite

- **The user builds it themselves**, per their agent, via a **guided app builder**: pick
  representative traces / author tasks, set a success-check per task, then **freeze** it
  (versioned, tamper-guarded).
- Proving is **only as meaningful as the suite is representative.** This is the highest-effort step
  for the user; the app scaffolds it hard (suggest tasks from real traces, propose success-checks)
  but cannot fully automate the human judgment of "representative + correct."

## 10. The app — the single control + monitoring surface

| Stage | What the user does *in the app* | Runs on | Status |
|---|---|---|---|
| 1. Add your agent | Create/point to an experiment, verify tracing, name it | app + DB | exists (wizard) |
| 2. State your goal(s) (NL) | What to optimize + what quality matters — drives judges *and* RLM | app | exists; extend to steer RLM |
| 3. Define your judges (NL) | Describe a quality dimension → a `{{trace}}` judge + name-matched label schema | app → DB job | net-new |
| 4. Build & freeze your suite | Pick traces / author tasks, set checks, freeze | app + DB | net-new |
| 5. Watch it collect | Readiness panel: traces in, judges + RLM scoring, "N traces / M labels — X to go" | DB jobs; app shows | mostly exists |
| 6. Label when prompted | Label flagged traces (names match judges) → at the floor, auto-align → judge trusted; agreement trend | app + DB | net-new (labeling UI + trigger) |
| 7. Review proposals | Queue of evidence-backed recommendations (judge + RLM say why) | companion (planner) → app | queue exists; planner net-new |
| 8. Verify (optional) | "Prove on my suite" → prover runs candidate vs. baseline → delta as added evidence | companion (prover) | prover exists; button net-new |
| 9. Approve → execute | Approve → executor agent makes the change (versioned, revertible) | companion (executor) | apply exists; open-ended agent net-new |
| 10. See impact & revert | Before/after by agent version; revert if it didn't help | app + DB | exists |

## 11. Deliberately dropped (the simplifications)

- The **mandatory prove-gate**, the **unified serverless optimization cycle**, and **automated
  gates on every change** are removed.
- The **frozen suite + prover** are kept **only** as the opt-in Tier-2 verification.

## 12. Build map (current → net-new)

- **Exists / reuse:** trace logging; base judges + MLflow monitoring; RLM/HALO logic; L0 metrics +
  publish; the app (leaderboard, comparison, approval queue, lineage, onboarding wizard, activity);
  MLflow Prompt Registry versioning; the apply engine; the frozen-suite + prover machinery.
- **Extend:** goal input steers RLM; RLM wired to GPT-5.5 Pro; readiness + labeling surfaced in the
  app.
- **Net-new:** judge-authoring (NL → `{{trace}}` judge + matched label schema); auto-align trigger;
  labeling UI; the local companion (planner); the open-ended executor agent; the "verify on my
  suite" button + companion wiring; the user-facing suite builder; UC-Volume snapshot versioning
  for arbitrary file change-sets; the companion one-command bootstrap.

## 13. Open items

- **Companion bootstrap** — a one-command / one-container start, so self-deploy is turnkey.
- **Tenancy** — assumed self-hosted per team/org; confirm if broader scope is intended.
- **Suite representativeness** — the quality of Tier-2 proving depends entirely on the user's
  suite; the app should scaffold it as much as possible.
