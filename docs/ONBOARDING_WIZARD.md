# In-app onboarding wizard + tutorial — QUEUED design of record

> **Status: QUEUED — not built.** This is the spec for a future lane, pinned so it
> isn't lost. It builds on the app's authenticated write-path (introduced by the
> Phase C approval control plane, see `LOOP_CONTROLLER.md`) — so it should be
> sequenced **after** Phase C lane 3b lands (the first in-app write-path).
>
> **Guiding principle: simplicity.** If onboarding a new agent can be reduced to a
> few button clicks and a short guided form, that is the bar. Split across multiple
> focused pages — do not cram everything onto one screen. Follow the
> **AppKit best-practice patterns** (the `databricks-apps-python` / AppKit design
> guidance) for wizard/stepper layout, validation, and state.

## Two deliverables

### A. A guided **"Add an agent" wizard** (multi-page workflow)
When a user wants to onboard a new agent — or a full multi-agent/supervisor system
— the app walks them through a **stepper** that collects everything the framework
needs, one page at a time, validating as it goes. One agent = one MLflow experiment
(see `OBSERVABILITY_APP.md`), so the wizard produces one registered agent.

### B. An **in-app tutorial** (guided walkthrough)
An in-product version of the [`GETTING_STARTED.md`](GETTING_STARTED.md) /
[`CONNECT_YOUR_AGENT.md`](CONNECT_YOUR_AGENT.md) docs: it explains, in the UI,
**exactly what has to happen to set this up per agent** and the requirements/gates,
so a first-time user is never guessing. It should be very clear about the data
prerequisites (below) — the honest "you won't see improvements until N traces +
labels exist" message, per agent.

## The wizard pages (maps the requirements)

1. **Connect tracing → a fresh MLflow experiment.**
   - Validate the agent is tracing to a **new/fresh** experiment (warn if the
     chosen experiment already has unrelated traces — one agent per experiment).
   - Offer a **"Create experiment" button** so the user can generate a fresh
     MLflow experiment from the app (a write action; reuse the Phase-C write-path
     pattern + framework SP).
   - Show the connect instructions (autolog or OTEL) inline.

2. **(Optional) Import OTEL traces.**
   - If the user has traces in **OTEL format**, let them import into the fresh
     experiment. Make clear this is a *backlog import* — the user must **still
     trace going forward** to that same experiment (import alone is a one-time
     seed, not a live feed).

3. **Choose the goal(s) to improve — a fixed dropdown.**
   - Offer ONLY these four options: **Token efficiency · Latency · Accuracy ·
     Cost.** No free-text.
   - Each goal maps to how progress is measured. **Honest design note to resolve
     at build time:** *Latency* and *Cost* are **deterministic L0 metrics** — an
     LLM judge cannot measure them better than the exact number, so a "MemAlign
     judge" for them would be redundant/theater; they should map to the
     deterministic scorer. *Accuracy* genuinely needs a **MemAlign-aligned judge**
     (human-calibrated). *Token efficiency* is hybrid (deterministic L0 signal +
     an optional quality-per-token judge). So the mapping is "goal → its scorer,"
     which is a MemAlign judge for Accuracy (and optionally token-efficiency), and
     a deterministic scorer for Latency/Cost. Capture the user's four-option intent
     but do NOT stand up a fake latency/cost "judge."

4. **Accept the data prerequisites (explicit confirmation).**
   - The user must acknowledge the framework **will not start optimizing until the
     data gates are met.** State the *real, code-enforced* floors
     (`ReadinessThresholds`): **~50 traces** to prove an improvement, and — for a
     **judged** goal (Accuracy; optionally token-efficiency) — **~20 human labels**
     to create/align the MemAlign judge. (A purely deterministic goal like Cost or
     Latency needs the traces but not labels.) The confirmation text should reflect
     exactly which gates apply to the goal(s) they picked.

5. **Later: the app prompts for labeling when it's time.**
   - Once enough traces exist, the app **flags** the agent and asks the user to
     **label traces**, scoped to the theme(s) they selected in step 3 (only ask for
     labels the chosen goal actually needs — e.g. Accuracy labels for an accuracy
     goal). This is the trigger that unblocks the MemAlign judge. Reuse the
     readiness panel's "you need N more labels" signal.

6. **Judge revert (over time).**
   - Make it **simple to revert the MemAlign judge back to a prior point** when the
     user specifies (e.g. an alignment made it worse). This mirrors the prompt
     lineage/revert already built for agent prompts (`ail-revert` /
     `set_prompt_alias`) and the MemAlign `unalign` capability — it needs **judge
     versioning** (track each alignment as a version with its human-agreement
     score) so the user can pick a version to revert to. One-click revert, with the
     before/after agreement shown.

## Jobs / optimization progress

A view (its own page — not crammed into the wizard) showing **running and recent
optimization jobs** (GEPA runs, RLM/HALO batches, MemAlign alignments, asset
generation) with status/progress, so the user can *see* the framework working.
Ties into the loop-controller's proposal/job records.

## Honest technical notes for the future builder

- **Write-path:** experiment creation, OTEL import, label prompts, and judge-revert
  are all **writes** — build them on the authenticated app write-path established
  by Phase C lane 3b (authenticated action, recorded actor, apply under the
  framework SP behind the gates). Do not fabricate an experiment/write without
  auth.
- **Real floors, not invented:** 50 traces to prove, 20 labels for a judged goal,
  0.5 scored-coverage — from `src/ail/readiness/compute.py`. Keep the wizard's
  copy in sync with the code (there was already one docs-vs-code drift fixed).
- **Latency/Cost are deterministic** — resolve the "each goal → a MemAlign judge"
  wording so we don't ship a judge that just reparrots a deterministic number.
- **Reuse, don't duplicate:** experiment/agent registration → `ail.registry`;
  readiness/label prompts → `ail.readiness`; judge alignment/revert →
  `ail.judges` + MemAlign; the tutorial content → the existing onboarding docs.
