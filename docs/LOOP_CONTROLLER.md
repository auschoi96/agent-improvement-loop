# The loop controller — autonomous detect → decide → prove → propose, human approves in the app

> **What this answers:** "Will the framework itself *trigger* actions (update a skill,
> change instructions, create a metric view) when it detects from feedback that it
> needs to — or does a human have to drive each step?"
>
> **The design:** the framework autonomously **detects, decides, proves, and
> proposes** a change; a **human approves the apply in the app**, reviewing *why*
> the change is necessary and the evidence that it works. This is **Option A**
> (autonomous up to the apply; the live change is human-gated) — chosen to match
> the "a human merges; the framework never silently changes production" discipline
> the rest of this project runs on.

## Status (honest)

The **action capabilities already exist and are tested**: asset generation
(`ail.optimize.assets` → UC metric-view specs), GEPA prompt/skill evolution
(`ail.optimize.gepa_runner`), prompt versioning + champion alias
(`ail.optimize.prompt_registry`), RLM/HALO diagnosis (`ail.l3`), the frozen-suite
WITH/WITHOUT proof (`ail.compare`), and the readiness/eval-health gates
(`ail.readiness`).

**Lane 2 — the controller that sequences them autonomously + the proposed-action
model — is now built and tested** (`ail.loop`): the goal-parameterized decision
rules (`ail.loop.decision_rules`), the typed, inert `ProposedAction` record
(`ail.loop.proposals`), the fail-closed `run_cycle` orchestrator over injectable
seams (`ail.loop.controller`), and the agent-scoped publish to the unified
`agent_proposed_actions` UC table lane 3 reads (`ail.loop.publish_proposals`). The
controller **detects, decides, proves, gates, and proposes only** — it applies
nothing and sets no champion alias.

**Lane 3a — the apply-on-approval engine — is built and tested**
(`ail.loop.apply.apply_approved_proposal`): the single, fail-closed place an
approved proposal becomes a live change, driven through injectable seams (registry
/ warehouse / lineage / gate-recheck / body-resolver).

**Lane 3b — the in-app approval queue + the authenticated approve/reject
write-path — is now built and tested**, closing the loop end-to-end. The
observability app gains its first write-path: a **custom, authenticated AppKit
server route** (`ail-self-optimizer/server/plugins/approvals`) that records the
**authenticated** app user as the approver and calls the lane-3a engine
**server-side** with real seams wired to the MLflow prompt registry, the framework
SQL warehouse, `ail.publish_lineage`, and `ail.readiness` (`ail.loop.apply_service`).
The read side is a two-tier SELECT-only `proposed_actions` query rendered by
`ApprovalQueue.tsx`. The loop is now self-triggering up to the human gate: the
framework detects → proves → gates → proposes; the human reviews the why + proof in
the app and clicks Approve; the engine re-checks the proof + gate and applies; the
change is recorded.

## The autonomy boundary (Option A)

| Stage | Who | Autonomous? |
|---|---|---|
| 1. Detect a problem from feedback | controller | ✅ yes |
| 2. Decide which action addresses it | controller (goal-driven rules) | ✅ yes |
| 3. Generate the candidate change | existing capability (asset gen / GEPA) | ✅ yes |
| 4. **Prove** it on the frozen suite (WITH/WITHOUT, fail-closed) | `ail.compare` | ✅ yes |
| 5. Gate on readiness + judge-vs-human agreement | `ail.readiness` | ✅ yes |
| 6. **Propose** it (with the "why" + the proof) into the app queue | controller | ✅ yes |
| 7. **Approve / reject the live apply** | **a human, in the app** | ❌ human-gated |
| 8. Apply the approved change + record to lineage | framework (on approval) | ✅ (after approval) |
| 9. Auto-revert if real post-apply impact regresses | controller | ✅ yes |

Everything up to the apply is autonomous. **A change reaches production only when a
human approves it in the app** — and a change is only ever *surfaced* for approval
if it already passed the proof and the gates (a crashed or non-improving candidate
is never proposed; fail-closed).

## Decision rules: feedback signal → proposed action

The controller compiles the user's natural-language goal (`ail.goals`) into the
objective + guardrails, then on each cycle maps detected feedback to a candidate
action. Illustrative rules (goal-parameterized, not hardcoded magic numbers):

- **RLM/HALO recommends an asset** of type X recurring across ≥ N traces **and** the
  goal metric (e.g. token-efficiency) is below target → generate the **metric view
  / tool** via `ail.optimize.assets`.
- **Redundant-read / boilerplate pattern** dominates the L0 waste diagnosis →
  propose the **read-cache / context-compaction skill** update.
- **A judge flags a quality dimension** (e.g. modularity) below threshold *and* that
  judge's human-agreement is above the trust floor → trigger **GEPA** to evolve the
  skill/instruction toward that dimension.
- **A registered version's real post-apply impact regresses** vs its predecessor →
  propose (and, for additive assets, auto-execute) a **revert**.

Each rule names the *evidence* it fired on, so the proposal carries a defensible
"why".

## The proposed-action record (the "why" payload)

Every proposal the human reviews carries:

- **What** — the concrete change: the skill/prompt diff, the metric-view SQL, or the
  GEPA-evolved body. For an **open-ended** `AGENT_TASK` (the L7b-2 executor's change),
  the "what" is a `plan` (the NL intended change + why) **plus** a `preview_diff` — the
  concrete produced change the human reviews *before* approving — and a
  `produced_change_ref` (an L6 snapshot / UC Volume ref) to commit on approval. The
  executor fills the preview + ref in a sandbox pre-approval; apply is fail-closed until
  that lane exists, and an `AGENT_TASK` never applies via the deterministic evidence-only
  path. See `docs/PRODUCT_ARCHITECTURE.md` §7.
- **Why** — the triggering feedback: the RLM recommendation / judge score / L0 waste
  signal, with trace references.
- **Proof** — the frozen-suite WITH/WITHOUT result (delta on the goal metric **with
  correctness held**), fail-closed (no proof → not proposable).
- **Gate status** — readiness tier + judge-vs-human agreement + scored coverage; an
  uncalibrated/distrusted judge cannot certify a proposal.
- **Risk class** — additive asset (low blast radius, trivially reversible) vs.
  agent-prompt/skill/instruction change (higher blast radius).

## The app as the approval control plane

This adds a **human-in-the-loop approval queue** to the observability app:

- A **Proposals** view, per agent, lists pending proposed actions with the full
  *why + proof + gate status* above — so the human reviews **evidence**, not a bare
  request.
- **Approve** / **Reject** (with a required reason). Both are recorded with the
  **approver identity + timestamp** — rejections are signal too (they tell the
  controller a rule mis-fired) and are auditable.
- On **approve**, the framework applies the change through the existing capabilities
  — register the prompt version + set the `champion` alias (skill/prompt/instruction
  changes), or `CREATE` the metric view (assets) — and records it to the **lineage
  timeline** (`agent_prompt_lineage`), so "what changed, why, with what proven
  delta, approved by whom" is one auditable trail. Revert remains available
  (`ail-revert` / re-point the champion alias).

### Architectural note: the app gains a write-path (deliberate, scoped)

Until now the app has been strictly **read-only** (SELECT against precomputed UC
tables — the two-tier rule). The approval queue is the **first write-path**: an
**authenticated** approve/reject action that (a) records the human decision and
(b) on approval triggers the gated apply. This is a real, intentional departure and
will be built with care:

- The approve/reject endpoint is an **authenticated server action** carrying the
  app user's identity (recorded as the approver) — not an anonymous mutation.
- The **apply runs server-side under the framework service principal** (the
  single-SP + grants from `docs/DEPLOY.md`), behind the same fail-closed gates — the
  button does not bypass the proof/readiness wall.
- Reads stay two-tier (SELECT-only); only the explicit approval/apply actions write.

#### AppKit write-path — validated against the installed SDK (`@databricks/appkit` 0.38.1)

The write mechanism was **confirmed against the installed SDK** (not assumed).
AppKit's `server()` plugin **does** support a custom, authenticated server-side
route:

- A **custom plugin** subclasses `Plugin` and implements `injectRoutes(router)`
  (routes mount under `/api/<pluginName>/…`) — the app registers it in
  `createApp({ plugins: [analytics(), server(), approvals()] })`. This is the
  documented extension point for "custom API routes / background logic".
- The **calling user's identity** is available server-side from the platform's
  forwarded headers (`x-forwarded-email` / `x-forwarded-user`; the SDK's
  `getExecutionContext()` / `asUser(req)` read the same). Lane 3b records that
  authenticated identity as the approver and **ignores** any approver in the request
  body (never client-trusted). An unauthenticated request is refused (401).

So the full authenticated write-path is buildable in-app and was built.

**Transport, not capability — and now BOTH transports are built.** The lane-3a
engine is **Python**; the app server is **Node**. Lane 3b bridges them behind a
**single seam** — the `ApplyBridge` type in `server/plugins/approvals/bridge.ts` —
and there are now **two implementations of that seam**, selected by environment:

- `spawnPythonApplyBridge` (default) invokes `python -m ail.loop.apply_service` as a
  subprocess (synchronous, so `ApplyRefused` / `ApplyRecordError` / success surface
  cleanly). This is the **local-dev / self-hosted** transport, where the `ail`
  package is importable.
- `jobTriggerApplyBridge` is the **deployed (Node-only) image** transport. The
  deployed Databricks App image is Node-only — the framework's Python ships as a
  wheel installed into **serverless Jobs** (`docs/DEPLOY.md`) — so this bridge
  **triggers a pre-deployed Databricks Job** (`ail-apply-job`,
  `resources/apply_service.job.yml`) that runs the *same*
  `ail.loop.apply_service.run_decision` engine under the framework service
  principal, polls the run to a terminal state, and returns the engine's **real**
  result. Because a serverless `python_wheel_task` does not stream stdout back to
  the trigger, the job writes its `ApplyServiceResult` (full JSON) to a small UC
  Delta result table (`agent_apply_results`, keyed by `(proposal_id, decided_at)`)
  and the bridge reads that row back **after** a terminal SUCCESS. Fail-closed: a
  failed run, a run still non-terminal at the timeout, a non-SUCCESS terminal state,
  or a missing/unparseable result row all surface an honest `outcome:"error"` —
  never a fabricated apply.

`ApprovalsPlugin` picks the transport by env: `AIL_APPLY_TRANSPORT=job` (or
`AIL_APPLY_JOB_ID` being set) selects the Job trigger; otherwise the subprocess is
used (`docs/DEPLOY.md` documents the deployed-app env contract). It was built with
`@databricks/sdk-experimental` (already a dep). The engine, the seam wiring, the
authenticated route, and the queue are **unchanged** across both transports — only
the transport differs.

## Build sequence

1. **Lineage / audit timeline + revert CLI** (in progress) — the record the
   controller writes into and the human audits.
2. ✅ **Loop controller + proposed-action model — built** (`ail.loop`): autonomous
   detect→decide→prove→gate→propose; writes pending proposals (with the
   why+proof+gate payload) to the unified `agent_proposed_actions` table; **never
   applies on its own** (there is no apply seam to call). Fail-closed: only proven,
   gated candidates become proposals; a crashed/non-improving candidate or an
   ungated state yields no proposal. Unit-tested end-to-end with injected fakes
   (no live MLflow/agent runs).
3. ✅ **App approval queue + authenticated approve/reject + apply-on-approval —
   built** (lane 3). Lane 3a (`ail.loop.apply`) is the fail-closed engine; lane 3b
   is the human control plane: a two-tier SELECT-only `proposed_actions` query +
   `ApprovalQueue.tsx` for the review surface (why + proof + gate), and an
   authenticated custom AppKit route (`server/plugins/approvals`) →
   `ail.loop.apply_service` that records the authenticated approver and triggers the
   gated apply server-side. On approve the engine re-checks proof + gate and applies
   (registering a version + champion alias, `CREATE`-ing a view, or reverting) and
   records to the lineage timeline; the decision (approve *and* reject, with reason)
   is recorded to the append-only `agent_action_decisions` audit. Reject applies
   nothing. Fail-closed: unauthenticated → refused; non-pending → refused (the engine
   enforces this). Unit-tested with fakes on both sides (no live write).

After these, the loop is genuinely self-triggering up to the human approval gate:
the framework notices a problem, builds and proves the fix, and presents it with
its evidence; the human clicks approve; the change ships and is recorded; if real
impact later regresses, the controller flags/reverts it.
