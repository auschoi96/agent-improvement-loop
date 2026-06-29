# Architecture

## 1. What this is

A reusable self-improvement loop for LLM agents. A deployer states a goal in
natural language (e.g. "cut token usage 30% without hurting correctness",
"make the cheaper model trustworthy for coding"); the system grounds that goal
in measured baselines, diagnoses the dominant waste/failure mode from real
traces, proposes and applies an intervention, and **proves** whether the new
version beats the old one on a held-out benchmark. It works for any agent that
can be (a) traced into MLflow and (b) run against a task input — not just
Claude Code.

## 2. The non-negotiable principle: a frozen evaluation wall

Every self-improvement loop we surveyed (the MLflow self-improving loop, the
OpenAI agent-improvement cookbook, the SICA repo) omits held-out test sets and
judge calibration. That omission is the core failure mode of this class of
system:

> If you optimize the agent against a judge **and** align the judge against
> feedback drawn from the same loop, the agent and its judge co-adapt. Scores
> climb while real quality stalls or regresses. You get a dashboard that says
> "improved" and an agent that isn't.

So the spine of this system is **not** the optimizer — it is a frozen,
human-anchored evaluation harness that the optimization loop is never allowed
to train against. Concretely, three **disjoint** data pools that are never
mixed:

- **Task Suite** — a fixed, versioned set of representative tasks re-run to
  compare agent versions. *Never* fed to the optimizer. Frozen and rotated on a
  deliberate cadence with overfit alarms.
- **Alignment Set** — labeled traces used to align judges (MemAlign). Disjoint
  from the Task Suite.
- **Human Anchor** — a small human-labeled slice used to measure
  judge-vs-human agreement as a first-class metric with a floor. A drifting
  judge is a distrusted judge.

## 3. Layered metrics: cheap/deterministic → expensive/judged

This is the answer to "easy to prove for cost, hard to prove for quality."
Lead with what is irrefutable; gate the subjective behind calibration.

- **L0 — Deterministic** (free, un-gameable): tokens, $, latency, tool-call
  count, redundancy rate, model used. Computed directly from trace metadata.
  *Token/cost proof lives here and is available today.*
- **L1 — Programmatic**: tests / lint / typecheck / build pass. Objective, for
  verifiable coding tasks.
- **L2 — Judged**: LLM-as-judge scorers (correctness, modularity,
  groundedness) built with MLflow `make_judge`, aligned via MemAlign, and
  validated against the Human Anchor.
- **L3 — RLM deep review** (DEFERRED / research): recursive-LM reads full long
  traces to *discover* failure modes a fixed scorer misses — used to decide
  *what to fix / what scorer to add*, never to score the leaderboard. The chosen
  in-platform path is an MLflow `{{ trace }}` judge (`make_judge`), not a Deno
  RLM runtime — see §11.

## 4. The loop

```
NL goal
  │  goals/compiler.py  →  objective + target metric(s) + guardrails
  ▼
diagnose  ── reads traces, L0 metrics, L2/L3 signals → dominant waste/failure mode
  ▼
propose intervention
  ├─ prompt / skill optimization  →  GEPA (optimize/runner.py harvest)
  └─ build a helper asset         →  metric view + tool + skill update,
                                      pipeline, semantic layer  (optimize/assets/)
  ▼
evaluate candidate vs baseline  ── on the FROZEN Task Suite only
  ▼
human gate  ── ship only if it beats the goal metric AND clears guardrails
  ▼
promote + re-baseline
```

Judge alignment (MemAlign) runs on its **own cadence** against fresh Human
Anchor labels — deliberately decoupled from agent optimization to break
co-adaptation.

## 5. Reusability seam: the Agent Adapter

Both harvest sources are Claude-Code-coupled at the trace layer
(`ai-dev-kit/.test/src/skill_test/trace/source.py` → `mlflow autolog claude` +
`~/.claude/projects/*.jsonl`; SkillForge only drives the local `claude`
binary). The portable spine (GRP + judges + MemAlign + GEPA + L0 metrics)
harvests cleanly, but the trace-ingestion + agent-execution interface is the
central **new** build and is what makes "works for any agent" real instead of
aspirational.

Two interfaces (`src/ail/ingest/base.py`):

- **`TraceSource`** — pull traces from MLflow regardless of producer, normalize
  to a common trace record (spans, tool calls, token usage, model, status).
- **`AgentAdapter`** — run the agent on a Task-Suite input and capture a trace.

Shipped adapters: Claude Code (clean-room original; see `PROVENANCE.md`) and
Codex (**new** — Codex does not
autolog to MLflow the way Claude Code does, so a Codex→MLflow trace-capture
path is required and is in scope). Users implement these two interfaces for
their own deployment.

## 6. Repository layout

```
agent-improvement-loop/
├── docs/                       # prose specs (authored directly, not generated)
├── pyproject.toml              # added in Wave 0
├── src/ail/
│   ├── ingest/
│   │   ├── base.py             # TraceSource + AgentAdapter interfaces  (NEW seam)
│   │   ├── mlflow_source.py    # CLEAN-ROOM original (public mlflow Traces API)
│   │   └── adapters/
│   │       ├── claude_code.py  # CLEAN-ROOM original (public claude-agent-sdk + mlflow)
│   │       └── codex.py        # NEW Codex→MLflow capture
│   ├── pools/                  # frozen wall: task_suite / alignment_set / human_anchor
│   ├── groundtruth/            # CLEAN-ROOM GRP (capture→execute→approve→promote) + own schema
│   ├── metrics/                # l0_deterministic (NEW) · l1_programmatic · l2_judged (HARVEST) · l3_rlm (DEFER)
│   ├── judges/                 # make_judge factory + memalign  (HARVEST)
│   ├── optimize/               # gepa_runner (HARVEST) + assets/ (metric-view/tool/skill/pipeline generators)
│   ├── goals/compiler.py       # NL goal → objective + metrics + guardrails  (NEW)
│   └── loop/controller.py      # diagnose → intervene → eval-on-frozen-suite → human-gate → promote
├── app/                        # Databricks App (UI) — built fresh; ai-dev-kit builder app is REFERENCE only
└── eval-criteria/              # rubric folders (REFERENCE skillforge + ai-dev-kit)
```

## 7. Harvest map (cross-vendor verified from source)

Verified independently by Claude Code and Codex/GPT-5 against
`databricks-solutions/ai-dev-kit` `.test/src/skill_test/`.

| Capability | Source | Verdict | Anchor |
|---|---|---|---|
| GEPA loop (`optimize_anything`) | ai-dev-kit | HARVEST | `optimize/runner.py:19,871` |
| `make_judge` scorers | public mlflow.genai | CLEAN-ROOM (original; see `PROVENANCE.md`) | `src/ail/judges/scorers.py` · public `mlflow.genai.judges.make_judge` |
| MemAlign `judge.align()` | public mlflow.genai | CLEAN-ROOM (original; see `PROVENANCE.md`) | `src/ail/judges/alignment.py` · public `Judge.align` / `MemAlignOptimizer` |
| Scheduled scorers (`register`/`start`) | public mlflow.genai + databricks-agents | CLEAN-ROOM (original) | `src/ail/judges/registration.py` · public `mlflow.genai.scorers` |
| MLflow `search_traces` ingestion | public mlflow | CLEAN-ROOM (original) | `src/ail/ingest/mlflow_source.py` · public `mlflow.search_traces` |
| GRP capture→approve→**promote** | ai-dev-kit | CLEAN-ROOM (original; see note) | `src/ail/groundtruth/` · separate `promote_approved()` |
| Claude Code adapter (`ClaudeSDKClient`) | public claude-agent-sdk | CLEAN-ROOM (original) | `src/ail/ingest/adapters/claude_code.py` · public `claude-agent-sdk` |
| e2e "quality up + tokens down" test | ai-dev-kit | REFERENCE (template for Phase 2) | `tests/test_optimize_e2e.py::test_optimize_improves_quality_and_reduces_tokens` |
| Ground-truth schema (`sources`+`regression_intent` required) | SkillForge `GroundTruthV5` | CLEAN-ROOM (own schema; concept only, not read) | `src/ail/groundtruth/schema.py` |
| `/forge` Designer⇄Critic case design | SkillForge | REFERENCE | `/forge`, `/forge-author` skills |
| Claude-Code trace parser, autolog | ai-dev-kit | SKIP/REPLACE (not agent-agnostic) | `trace/source.py`, `trace/parser.py` |
| RLM deep review | DSPy / HALO | DEFER (experimental, needs Deno runtime) | — |

**GRP note (Wave 1a — built clean-room):** the upstream GRP pattern requires
human review (`expectations: {}` filled by the reviewer) and keeps promotion as
a separate step, with no LLM synthesis of expected outputs anywhere — that
property is exactly why the pattern is worth reusing. We **reimplemented it from
scratch** in `src/ail/groundtruth/` (capture → execute → human-approve →
`promote_approved`) rather than harvesting code, because the SkillForge schema's
license is undeclared and the ai-dev-kit `grp/` is under a non-OSI license (see
`PROVENANCE.md`). The no-synthesis invariant is enforced structurally
(expectations are written only by the human-gate `approve` stage) and asserted
by tests.

## 8. Grounded baseline (reference deployment, experiment 660599403165942)

What the real data says — this drives sequencing:

- 77 traces, 19 days, **100% `OK` status**, all Claude Code, **zero Codex
  traces**. A live, growing corpus, not a stable historical baseline.
- Token usage is bimodal: median ~18.5K, but two traces near 549K and a max of
  943K (a 9.2-hour session). The 550K scenario (Example 1) is real and
  reproducible.
- Tool redundancy is real but not pathological: one trace Read the same path
  34×; shell-setup boilerplate re-runs 13–21× per trace.
- **Zero quality labels** anywhere — no assessments, no human ratings, no judge
  scores. All annotations are auto-generated system metadata.
- No session grouping — each trace is independent.

Two consequences:

1. **Token/cost improvement is provable today** (deterministic L0).
2. **Quality improvement is blocked on ground truth that does not exist** —
   MemAlign aligns a judge *against labels*, and there are none. A small
   human-anchored gold set must be created first (GRP + `/forge` methodology,
   no one-shot LLM synthesis).

## 9. Phased plan

- **Phase 0 — Foundations + irrefutable baseline.** Agent-agnostic ingestion +
  L0 deterministic metrics. Reproduce Example 1 on the real traces.
- **Phase 1 — Eval harness + ground-truth bootstrap.** Frozen Task Suite;
  human-labeled gold slice; `make_judge` scorers; MemAlign alignment;
  judge-vs-human agreement tracking. Generalize SkillForge's WITH/WITHOUT
  pattern beyond local skills.
- **Phase 2 — One lever end-to-end on the most provable goal (token
  reduction).** Build metric view + tool + skill update; candidate vs baseline
  on the frozen suite; token drop with correctness guardrail; human gate.
- **Phase 3 — Multi-lever + GEPA + asset generation generalized.** Codex /
  Example 2 lane, gated on getting Codex traces into MLflow first.
- **Phase 4 — Observability product (Databricks App).** Live leaderboard, NL
  goal compiler, judge-human agreement trend + drift alarm, per-intervention
  attribution, per-step human feedback into MemAlign + GEPA.
- **Phase 5 — RLM deep review (research, optional).** Gated behind
  demonstrated value, not built early because HALO/RLM were mentioned. Take the
  recursive-review *idea*, not a Deno dependency.

Milestone 1 = Phase 0 + Phase 1 + the Phase 2 token slice. See
[`MILESTONE-1.md`](MILESTONE-1.md).

## 10. Open risks / decisions

- **Codex trace capture is net-new** and a hard dependency for the Example 2
  lane. Accepted as core scope.
- **RLM** is experimental and needs a Deno runtime; deferred.
- **Cross-vendor review** depends on ≥2 vendors being available (Claude +
  Codex confirmed working; `pi` needs a provider login to serve as a third
  reviewer).
- **License reconciliation** for harvested code is a Wave 0 deliverable (see
  `PROVENANCE.md`); the repo carries no top-level LICENSE until that resolves.

## 11. Feedback-attachment architecture (where a verdict lives)

Once the L2 scorers are registered (`src/ail/judges/registration.py`), MLflow
evaluates traces and writes verdicts back. **Where** a verdict is written is a
deliberate design choice, not an MLflow default to accept blindly, because it
directly governs whether the L0 cost metric stays un-gameable (§3).

**1. A verdict is an assessment *on the subject trace*.** A scheduled scorer
attaches its `Feedback` to the trace it scored, with `source_type=LLM_JUDGE`.
That trace is the one record a reviewer, the leaderboard, and a future human all
look at. Multiple verdicts coexist on the same trace without collision because
an assessment is keyed by **`(name, source_type)`**: a judge's `correctness`
(`LLM_JUDGE`) and a human's `correctness` (`HUMAN`) live side by side on one
trace, and the agreement layer (`ail.judges.agreement`) compares them. This is
why the Human Anchor pairs a judge value with a human label per item — the two
are different sources of the *same* assessment name on the *same* subject.

**2. The reviewer's *computation* is its own trace, linked by metadata —
never nested.** Producing a verdict (a judge model call today, a recursive
deep-review tomorrow) is itself an LLM workload that emits spans and tokens. It
runs as its **own** trace and is linked back to the subject by recording the
subject's id and the reviewer's own trace id in the assessment / trace metadata
(convention: `reviewer_trace_id`). It is **never** added as child spans of the
agent's trace.

**3. Why nesting would corrupt L0.** Trace-level token usage is read from
`mlflow.trace.tokenUsage` (see `ail.ingest.mlflow_source` →
`_MD_TOKEN_USAGE`), which **sums the token usage of the trace's child LLM
spans**. If a reviewer's LLM spans were nested inside the agent's trace, the
agent's trace-level `tokenUsage` would silently include the *judge's* tokens —
inflating the agent's measured cost and breaking the one metric §2/§3 promise to
keep deterministic and un-gameable. Keeping the reviewer's spans in a separate
trace keeps the agent's L0 cost exactly the agent's, and makes the cost of
*judging* separately measurable.

**4. Trace-based judges are the chosen RLM path (no Deno).** `make_judge`
accepts a `{{ trace }}` template, so a judge can read an entire long trace and
discover failure modes a fixed input/output rubric misses. That is the
**in-platform equivalent of the deferred RLM / HALO recursive review** (§3 L3,
§9 Phase 5) — the recursive-review *idea* realized through MLflow trace judges
rather than a separate Deno runtime. A `{{ trace }}` deep-review judge still
obeys rules 1–3: it attaches its verdict to the subject trace and runs its own
review as a separate, linked trace. This is the recorded RLM direction; the
Deno-based RLM dependency stays deferred.
