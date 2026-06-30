# Provenance & License Reconciliation

This file records where each non-trivial module in the repository came from and
flags any license obligation that must be resolved before the project takes a
top-level LICENSE. Until that reconciliation is complete, the repository carries
**no top-level LICENSE**.

A license-provenance check is an explicit **Wave 0** deliverable.

## Clean-room reimplemented modules (original work)

The three ingestion modules below were **reimplemented clean-room** as original
work. They preserve only this repository's own public interface (the
`ail.ingest.base` contracts, which are this project's design) and were written
**without reading, cloning, fetching, browsing, or grepping
`databricks-solutions/ai-dev-kit` in any way**. Each was re-derived solely from
(a) this repository's own interfaces and tests and (b) the **public**
documentation/source of the packages listed under "Public sources" below.

| Module | Status | Re-derived from (public sources only) |
|---|---|---|
| `src/ail/ingest/base.py` — `TokenUsage` (fields + `total_tokens`/`cache_tokens`) and `ToolCall` `mcp__` parsing | **Original (clean-room)** | Public Anthropic Messages API `usage` object schema; public MCP/Claude Code tool-naming convention `mcp__<server>__<tool>`. The rest of `base.py` (`TraceStatus`, `SpanKind`, `NormalizedSpan`, `NormalizedTrace`, `TraceSource`, `AgentTask`, `AgentRunResult`, `AgentAdapter`) is this repo's own design and was left as-is. |
| `src/ail/ingest/mlflow_source.py` | **Original (clean-room)** | Public `mlflow` Traces API (`mlflow.search_traces`, `mlflow.get_trace`), the public `mlflow.entities.Trace` shape, public MLflow trace-metadata/span-attribute key conventions, and public `databricks-sdk` (`WorkspaceClient`) + MLflow configuration docs for Databricks-managed MLflow (`tracking_uri="databricks"`, `registry_uri="databricks-uc"`). |
| `src/ail/ingest/adapters/claude_code.py` | **Original (clean-room)** | Public `claude-agent-sdk` (`ClaudeSDKClient`, `ClaudeAgentOptions`, `HookMatcher`, and the `AssistantMessage`/`ResultMessage`/`SystemMessage`/`UserMessage`/`TextBlock`/`ToolUseBlock`/`ToolResultBlock` types); public `mlflow.claude_code.tracing` (`setup_mlflow`, `process_transcript`) and its documented env-var contract; public Claude Code `.mcp.json` / `mcpServers` configuration conventions. |

### Public sources used

- **`mlflow`** (OSS, Apache-2.0), pinned `mlflow>=3.14,<4`, resolved/verified against **3.14.0**:
  `mlflow.search_traces` (scoping via `locations=[experiment_id]`, `return_type="list"`),
  `mlflow.get_trace`, `mlflow.set_tracking_uri` / `mlflow.set_registry_uri`,
  `mlflow.entities.Trace` (`.info`, `.data.spans`), and `mlflow.claude_code.tracing`.
- **`databricks-sdk`** (OSS, Apache-2.0): `databricks.sdk.WorkspaceClient` and its
  `.config.host`, with the active workspace selected by a Databricks CLI profile.
- **`claude-agent-sdk`** (public on PyPI, MIT): the client, options, hook, and
  message/content-block types, all imported lazily as an optional dependency.
- **Public Anthropic Messages API** `usage` schema: `input_tokens`, `output_tokens`,
  `cache_creation_input_tokens`, `cache_read_input_tokens`.
- **Public MCP / Claude Code** tool-naming convention: `mcp__<server>__<tool>`.

### base.py provenance gap — disclosed and closed

An independent audit flagged that, in earlier revisions, `base.py`'s `TokenUsage`
properties and `ToolCall` `mcp__` parsing appeared derived from an uncited
upstream file. That gap is **closed**: those two pieces were reimplemented
clean-room from the public Anthropic `usage` schema and the public MCP naming
convention (above). No upstream code was consulted. The surrounding `base.py`
interfaces were already this repository's own design and were not changed.

## Phase-2 optimization lever (original work)

The Phase-2 token-efficiency lever is **original work for this repository**, built
only on this project's own contracts (`ail.compare`, `ail.ingest.base`,
`ail.task_suite`, `ail.groundtruth`) plus `pydantic` / `pyyaml` / the stdlib. It
introduces **no new dependency** and harvests **no upstream code** — in
particular it does **not** harvest the GEPA loop noted as a future `optimize/`
source in `docs/ARCHITECTURE.md` §7; that remains unincorporated.

| Module | Status | Notes |
|---|---|---|
| `src/ail/optimize/assets/skills/token-efficient-execution/SKILL.md` | **Original** | Behavioural skill authored for this repo (avoid redundant re-reads, batch shell commands, drop `cd`/setup boilerplate). Not copied from any skills library; never written to a user/`polly`/`~/.claude` skills dir. |
| `src/ail/optimize/assets/__init__.py`, `src/ail/optimize/lever.py` | **Original** | `SKILL.md` loader + the `Intervention` that injects the skill into a candidate task's system prompt; the BASELINE/CANDIDATE configs. |
| `src/ail/optimize/phase2.py`, `scripts/run_phase2_comparison.py` | **Original** | The frozen-suite comparison runner + artifact contract and its thin CLI; reuses `ail.compare.compare_candidate` unchanged for the actual comparison. |

## Outstanding source (not yet incorporated)

| Upstream repo | Module | Upstream license | Status |
|---|---|---|---|
| `databricks-field-eng/skillforge` @ `3569232e` | `python/skillforge/eval/schema.py` (`GroundTruthV5`) | **Undeclared** — no `LICENSE` file in repo; `package.json` says `"MIT"` (unverified, no license text) | **NOT incorporated and NOT read.** Wave 1a (`src/ail/groundtruth/`) was resolved by writing our **own** clean-room schema instead of copying `GroundTruthV5`; only the conceptual contract was reimplemented. See the Wave 1a clean-room note above. |
| `databricks-solutions/ai-dev-kit` | `grp/` (capture → approve → promote pipeline) | Databricks "DB license" (proprietary, non-OSI) | **NOT incorporated and NOT read.** The GRP capture/execute/approve/promote stages in `src/ail/groundtruth/` are original work; only the *pattern* (human-gated, separate promote, no expected-output synthesis) was reimplemented. |

## History (why this remediation happened)

Earlier revisions of `mlflow_source.py` and `claude_code.py` were *harvested* from
`databricks-solutions/ai-dev-kit` (`@ c4947868`, under the Databricks "DB
license", a proprietary non-OSI license), and `base.py` carried the uncited
derivation noted above. Because the DB license is **not** a permissive
open-source grant, that code blocked any Apache-2.0 release. The three modules
were therefore reimplemented clean-room (this PR) so the repository contains no
DB-licensed code in these paths. The `HARVEST` / `CHANGES FROM UPSTREAM` headers
that previously sat atop the two harvested files were removed, since the files
are now original work and there is no longer an upstream change-notice
obligation to satisfy for them.

## Original work (NOT harvested — clean-room, Apache-2.0-ready)

These modules are 100% original work authored for this repository. They contain
**no** code copied from `ai-dev-kit`, `skillforge`, or any other source — they
only consume this repo's own ingestion types (`ail.ingest.base`) and public OSS
APIs (pydantic; and, for the live pull, the public `mlflow.search_traces` /
`get_trace`). They carry no upstream license obligation and are compatible with
the intended Apache-2.0 release.

| Module | Description |
|---|---|
| `src/ail/metrics/contract.py` | L0 metrics output contract (pydantic v2 schema). **Original.** |
| `src/ail/metrics/l0_deterministic.py` | L0 deterministic metric computation (tokens, cost, latency, tool-call redundancy). **Original.** |
| `src/ail/metrics/report.py` | Single-entrypoint baseline report + Example 1 reproduction. **Original.** |
| `docs/L0_METRICS_CONTRACT.md` | Prose spec of the L0 contract. **Original.** |
| `tests/test_l0_metrics.py`, `tests/test_report.py` | Tests for the above. **Original.** |
| `src/ail/pools.py` | Canonical `Pool` vocabulary (the three disjoint pools), shared by `groundtruth` and `judges` so the identity is defined once. **Original.** |
| `src/ail/groundtruth/schema.py` | Our own ground-truth contract (pydantic v2): `Source`/`SourceKind`, `ReviewStatus`, `TaskInput`, `Expectations`, `CandidateResponse`, `ReviewRecord`, `GroundTruthCase`, `GroundTruthSet` (the shared `Pool` is imported from `ail.pools`). **Clean-room original** — see the Wave 1a note below. |
| `src/ail/groundtruth/capture.py` | Stage 1: build candidate cases from normalized traces. **Original.** |
| `src/ail/groundtruth/execute.py` | Stage 2: run the agent to capture its own candidate response; optional MLflow audit logging. **Original.** |
| `src/ail/groundtruth/approve.py` | Stage 3: the human gate — a reviewer fills expectations and approves/rejects. **Original.** |
| `src/ail/groundtruth/promote.py` | Stage 4: separate explicit `promote_approved()` into a frozen, disjoint pool. **Original.** |
| `src/ail/groundtruth/store.py` | Per-pool persistence + review-queue round-trip. **Original.** |
| `tests/test_groundtruth_schema.py`, `tests/test_groundtruth_pipeline.py`, `tests/test_groundtruth_mlflow.py` | Tests for the ground-truth package (incl. the no-synthesis assertions). **Original.** |
| `src/ail/judges/scorers.py` | L2 scorer factory: `ScorerSpec` + the `correctness`/`modularity`/`groundedness`/`token_efficiency` rubrics, thin wrappers over public `mlflow.genai.judges.make_judge`. The hybrid `token_efficiency` judge + its `build_token_efficiency_inputs` L0→L2 bridge consume the project's own `ail.metrics` L0 contract. Rubrics written from scratch. **Original (clean-room).** |
| `src/ail/judges/alignment.py` | MemAlign wrapper over public `Judge.align` / `MemAlignOptimizer`, plus the `unaligned_report` provenance helper. **Original (clean-room).** |
| `src/ail/judges/agreement.py` | Pure judge-vs-human agreement metric (rate, floor, Cohen's kappa, fail-closed insufficient-data) + best-effort MLflow logging. **Original.** |
| `src/ail/judges/pools.py` | Consumer-side pool handles (`AlignmentSet`/`HumanAnchor`/`AnchorItem`) + `assert_pools_disjoint` (re-exports the shared `ail.pools.Pool`). **Original.** |
| `src/ail/judges/contract.py` | L2 output contract (pydantic v2): `AgreementReport`, `AlignmentReport`. **Original.** |
| `src/ail/judges/registration.py` | Scheduled-scorer registration over public `mlflow.genai.scorers` (`Scorer.register`/`start`/`list_scorers`/`delete_scorer`), backed by `databricks-agents`, with the MemAlign-aware **align-then-register** pipeline (`create_aligned_scorer`, `register_scorers(alignment_set=...)`). **Original.** |
| `src/ail/judges/labeling.py` | Human-label recording over the public assessment API (`mlflow.log_feedback` / `mlflow.log_expectation` with `AssessmentSource(source_type=HUMAN)`) + assembly of disjoint `AlignmentSet`/`HumanAnchor` pools (reuses the `ail.ingest` seam for raw-trace fetch; proves disjointness via `assert_pools_disjoint`). **Original (clean-room).** |
| `tests/test_judges.py`, `tests/test_judges_registration.py`, `tests/test_judges_labeling.py`, `tests/test_pools.py` | Tests for the judge layer (incl. the token-efficiency judge, align-then-register, and labeling) and the shared pool vocabulary. **Original.** |
| `src/ail/compare/contract.py` | Phase-2 comparison output contract (pydantic v2, `extra="forbid"`): `ComparisonResult`, `MetricDelta`, `GuardrailCheck`, `Recommendation`. **Original.** |
| `src/ail/compare/harness.py` | Candidate-vs-baseline comparison harness: runs an `AgentAdapter` with/without an `Intervention`, computes L0 deltas via the repo's own `ail.metrics`, applies the correctness non-regression guardrail via the repo's own `ail.judges` (base correctness judge — interim until MemAlign-aligned), and emits a `PROMOTE`/`BLOCK` recommendation. Consumes only this repo's own contracts (`ail.ingest`, `ail.metrics`, `ail.judges`, `ail.groundtruth`) + pydantic. **Original.** |
| `src/ail/compare/monitoring.py` | `configure_monitoring_warehouse`: set an experiment's monitoring SQL warehouse (`mlflow.monitoring.sqlWarehouseId` tag / `MLFLOW_TRACING_SQL_WAREHOUSE_ID`) over the public `mlflow.MlflowClient`. **Original.** |
| `tests/test_compare.py` | Tests for the comparison harness (the three decision cases, fail-closed scoring, the L1 guardrail, suite immutability) and the monitoring-warehouse helper. **Original.** |

### Wave 1a ground-truth schema — clean-room note

`src/ail/groundtruth/` was authored **without reading, cloning, fetching,
browsing, or grepping** SkillForge's `eval/schema.py` (`GroundTruthV5`) or any
`ai-dev-kit` `grp/` code. The package re-derives an *equivalent contract from
scratch*: it borrows only the **conceptual** properties we admired — every case
carries required provenance (`sources`), states a `regression_intent`, and is
approved by a human with **no LLM synthesis of expected outputs** — and
implements them in this repository's own Pydantic style (`extra="forbid"`,
frozen models) re-derived from the public `pydantic` v2 API. The
anti-co-adaptation invariant (`Expectations` is filled only by the human-gate
`approve` stage) is original to this implementation and asserted by tests
(`test_no_expected_output_synthesis_surface_in_package`,
`test_only_the_human_gate_writes_expectations`). MLflow usage is the public
`mlflow` Tracking API only (`start_run`, `log_params`, `set_tags`, `log_text`),
resolved/verified against **3.14.0**; no `mlflow.genai`/judge surface is used.

Cost prices in `l0_deterministic.py` are *data*, not code: base input/output
rates are attributed to the Claude API pricing reference (the `claude-api`
skill's cached model table, dated 2026-06-04) on each `PriceBookEntry.source`,
cache rates are derived from Anthropic's documented prompt-caching multipliers,
and any uncovered model is flagged rather than guessed.

### L2 judges layer (`src/ail/judges/`) — clean-room note

The L2 judge layer was authored **without reading, cloning, fetching, browsing,
or grepping** `ai-dev-kit`'s `optimize/judges.py` / `optimize/alignment.py` (the
`HARVEST`-tagged sources for `make_judge` scorers and MemAlign in
`docs/ARCHITECTURE.md` §7) or any other non-permissive source. Despite the §7
harvest map marking those `HARVEST`, the modules here are **original** and
re-derived solely from this repository's own interfaces/tests and the **public,
OSS Apache-2.0 `mlflow.genai` API**:

- `mlflow.genai.judges.make_judge` (name/instructions/`{{ template }}` variables,
  `feedback_value_type` incl. `Literal[...]` constrained categorical / bounded
  graded scales, `inference_params`), `Judge.align`, and
  `mlflow.genai.judges.optimizers.MemAlignOptimizer` (`reflection_lm`,
  `retrieval_k`, `embedding_model`, `embedding_dim`).
- `mlflow.genai.scorers` registration API — `Scorer.register` / `Scorer.start` /
  `Scorer.stop`, `list_scorers` / `delete_scorer`, and `ScorerSamplingConfig`
  (`sample_rate`, `filter_string`) — for scheduled (ongoing) scorers. The
  Databricks runtime backend is the public **`databricks-agents`** package
  (added as the optional `agents` extra), not copied code.
- The public MLflow **assessment-logging** API used by the labeling helper —
  `mlflow.log_feedback` / `mlflow.log_expectation` and
  `mlflow.entities.assessment_source.AssessmentSource` /
  `AssessmentSourceType.HUMAN` — and `MlflowClient.set_experiment_tag` for the
  best-effort `aligned` provenance tag. Public OSS surface only.

The three rubrics (`correctness`/`modularity`/`groundedness`), the agreement
metric (rate + floor + Cohen's kappa + the fail-closed insufficient-data rule),
the disjoint-pool handles, and the registration orchestration are all this
repository's own design. No upstream judge/alignment code was consulted, so the
layer carries no upstream license obligation and is Apache-2.0-ready. This note
makes good on the `docs/L2_JUDGES_CONTRACT.md` claim that provenance is recorded
here — earlier it was asserted but not actually written down.

## ⚠️ License flags (resolve before adding a top-level LICENSE)

1. **skillforge has no declared license file.** The repo (private,
   Databricks-Field-Eng-internal) ships **no `LICENSE`/`COPYING`/`NOTICE`** at
   its root; only `package.json` claims `"license": "MIT"`, with no
   accompanying MIT license text and no `license` field in `pyproject.toml`.
   This is an **ambiguous / undeclared license**. Because of this, Wave 1a did
   **not** copy `GroundTruthV5`; `src/ail/groundtruth/` is a clean-room original
   schema written without reading the SkillForge source (see the Wave 1a
   clean-room note above). The standing rule remains: **do not copy SkillForge
   code until its license is clarified in writing** — a bare `package.json`
   "MIT" string without license text is not a reliable grant. A different-vendor
   review should verify the non-derivation of the ground-truth package, as for
   the ingestion modules.

2. **Confirm before relicensing.** With the three ingestion modules now
   clean-room, the previously-blocking DB-licensed code is gone from these
   paths. A clean-room reimplementation that preserves only an interface is the
   standard remedy, but a human should confirm no other module derives from a
   non-permissive source before a top-level Apache-2.0 LICENSE is added, and an
   independent (different-vendor) review should verify the non-derivation of the
   three modules above.

## Reference-only (not copied)

- ai-dev-kit `.test/src/skill_test/trace/source.py` + `parser.py` — the
  Claude-Code-coupled local-JSONL/autolog path. **Deliberately NOT used**: the
  reimplemented `mlflow_source.py` reads only the public MLflow Traces API and
  is producer-agnostic.
- SkillForge `eval/schema.py` (`GroundTruthV5`) — **conceptual reference only,
  not read.** The required-provenance + regression-intent + human-approved
  contract was re-derived from scratch in `src/ail/groundtruth/schema.py`.
- ai-dev-kit `grp/` capture→approve→promote pipeline — **conceptual reference
  only, not read.** The four-stage human-gated pipeline was reimplemented as
  original work in `src/ail/groundtruth/`.
- SkillForge `/forge`, `/forge-author` skills — methodology reference only.
- ai-dev-kit builder app — UI reference only.
- DSPy `RLM` — design inspiration only (deferred).

## Adopted third-party dependencies (NOT vendored)

These are used through their published package and public API; their source is
**not** copied into this repository. Attribution also lives in `NOTICE`.

| Dependency | License | Used by | How |
|---|---|---|---|
| **HALO** (`halo-engine`, [context-labs/halo](https://github.com/context-labs/halo)) | **MIT** | `src/ail/l3/` | The L3 recursive trace reviewer **adopts** HALO as the trace-specialized Recursive LM engine (byte-offset index + bounded navigation tools + recursive subagents + compaction) so arbitrarily large traces (this corpus reaches 943K tokens) can be reviewed without exceeding a single judge call's context. We **do not reimplement or vendor the engine**: it is the optional `l3` extra (`pip install 'ail[l3]'`), lazy-imported. AIL contributes only original glue around it — see the L3 row below. |
| **DSPy** (`dspy`, [stanfordnlp/dspy](https://github.com/stanfordnlp/dspy)) | **MIT** | `src/ail/judges/alignment.py` | The optimizer backend that MLflow's `MemAlignOptimizer` requires for judge alignment. AIL does **not** call dspy directly — it reaches it only through MLflow's public `mlflow.genai.judges.optimizers` API. dspy is **not vendored**: it is the optional `align` extra (`pip install 'ail[align]'`), lazy-imported inside `build_memalign_optimizer`, so the base (unaligned) judges and CI never require it. (Distinct from the DSPy `RLM` *design-inspiration* reference noted above — this is the actual library dependency.) |

### L3 recursive trace reviewer (`src/ail/l3/`) — clean-room note

`src/ail/l3/` is 100% original work authored for this repository. It contains
**no** code copied from `context-labs/halo` (or any other source); it only
*calls* HALO's published public API (`engine.main.run_engine`, `EngineConfig`,
`ModelConfig`, `ModelProviderConfig`, `AgentConfig`, `AgentMessage`,
`AgentOutputItem`, and the `SpanRecord` JSONL input shape) as a dependency, and
consumes this repo's own ingestion types (`ail.ingest`) and the public
`mlflow` / `mlflow.genai` assessment APIs (`mlflow.log_feedback`,
`AssessmentSource`). The OpenInference/OTLP `SpanRecord` mapping, the structured
verdict contract, the free-text-report parser, the own-trace token isolation,
and the verdict-on-subject-trace attachment are all this repository's design.

| Module | Description |
|---|---|
| `src/ail/l3/contract.py` | L3 verdict output contract (pydantic v2, `extra="forbid"`): `HaloReviewVerdict`, `RedundancyFinding`, `FailureMode`. **Original.** |
| `src/ail/l3/adapter.py` | MLflow trace -> OpenInference/OTLP `SpanRecord` JSONL adapter (`mlflow_trace_to_otlp_jsonl`, `normalized_trace_to_span_records`). **Original.** |
| `src/ail/l3/parser.py` | Parse HALO's free-text `<final/>`-terminated report into a `HaloReviewVerdict`. **Original.** |
| `src/ail/l3/reviewer.py` | `review_trace`: run HALO under its **own** MLflow trace (token isolation), parse, attach the verdict to the subject trace as an `LLM_JUDGE` feedback assessment linked by `reviewer_trace_id`. **Original.** |
| `src/ail/l3/selection.py` | Pick which traces to review (top-N by tokens / above a token threshold) — L3 is expensive, so review the biggest/most-interesting, not every trace. **Original.** |
| `tests/test_l3_adapter.py`, `tests/test_l3_parser.py`, `tests/test_l3_reviewer.py` | Tests for the L3 layer (HALO engine + model + feedback logging mocked; the live e2e is `@pytest.mark.live`). **Original.** |

## Rules

1. Any file copied from another repository gets a header naming its upstream
   source path and commit, and is recorded in this file in the same PR. (No such
   files exist today: the three ingestion modules are original clean-room work.)
2. No upstream license obligation is dropped. If an upstream license is
   incompatible with this repo's intended license, the code is reimplemented
   clean-room (as here) or added as a properly-licensed dependency, never
   copied.
3. Clean-room reimplementations must be written without accessing the
   non-permissive source, working only from the project's own interfaces/tests
   and the public documentation/source of permissively-licensed packages.
