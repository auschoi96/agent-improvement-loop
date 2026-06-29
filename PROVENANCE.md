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

## Outstanding source (not yet incorporated)

| Upstream repo | Module | Upstream license | Status |
|---|---|---|---|
| `databricks-field-eng/skillforge` @ `3569232e` | `python/skillforge/eval/schema.py` (`GroundTruthV5`) | **Undeclared** — no `LICENSE` file in repo; `package.json` says `"MIT"` (unverified, no license text) | **NOT incorporated** (Wave 1a) — see flag below |

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
| `src/ail/judges/pools.py` | Frozen-evaluation-wall pool types (`Pool`, `AlignmentSet`, `HumanAnchor`, `assert_pools_disjoint`). **Original.** |
| `src/ail/judges/scorers.py` | L2 `make_judge` scorer factory (correctness/modularity/groundedness). **Original.** |
| `src/ail/judges/alignment.py` | MemAlign `judge.align` wrapper + `build_memalign_optimizer`. **Original.** |
| `src/ail/judges/agreement.py` | Judge-vs-human agreement metric with a configurable floor. **Original.** |
| `src/ail/judges/contract.py` | L2 judges output contract (pydantic v2 schema). **Original.** |
| `docs/L2_JUDGES_CONTRACT.md` | Prose spec of the L2 judges layer. **Original.** |
| `tests/test_judges.py` | Tests for the L2 judges layer (judge/LLM calls mocked offline). **Original.** |

### L2 judges layer — clean-room against public MLflow GenAI APIs only

The `src/ail/judges/` package (Wave 1c) is **original clean-room work**. It was
written **without reading, cloning, fetching, browsing, or grepping**
`databricks-solutions/ai-dev-kit` or `databricks-field-eng/skillforge` in any
way — their judge/alignment ideas were referenced *conceptually* only (the
HARVEST entries for `make_judge`/MemAlign in `docs/ARCHITECTURE.md` §7 describe
those upstreams as inspiration; **no code was harvested** for this package). All
code was re-derived solely from (a) this repository's own interfaces and (b) the
**public** OSS MLflow GenAI API:

- **`mlflow`** (OSS, Apache-2.0), pin `mlflow>=3.14,<4`, resolved/verified
  against **3.14.0**:
  `mlflow.genai.judges.make_judge` (and its `feedback_value_type` / template-
  variable contract), `mlflow.genai.judges.Judge.align(traces=, optimizer=)`,
  `mlflow.genai.judges.optimizers.MemAlignOptimizer`,
  `mlflow.genai.judges.CategoricalRating`, `mlflow.entities.assessment.Feedback`
  (its `.value`), and `mlflow.log_metric` / `mlflow.log_dict`.
- **pydantic v2** for the output contract (`AgreementReport` / `AlignmentReport`).

The scorer rubrics in `scorers.py` are original prose authored for this project.
MemAlign's `dspy` requirement is isolated behind a lazy import so the package
imports — and the offline test suite runs — without `dspy` or any live model.

Cost prices in `l0_deterministic.py` are *data*, not code: base input/output
rates are attributed to the Claude API pricing reference (the `claude-api`
skill's cached model table, dated 2026-06-04) on each `PriceBookEntry.source`,
cache rates are derived from Anthropic's documented prompt-caching multipliers,
and any uncovered model is flagged rather than guessed.

## ⚠️ License flags (resolve before adding a top-level LICENSE)

1. **skillforge has no declared license file.** The repo (private,
   Databricks-Field-Eng-internal) ships **no `LICENSE`/`COPYING`/`NOTICE`** at
   its root; only `package.json` claims `"license": "MIT"`, with no
   accompanying MIT license text and no `license` field in `pyproject.toml`.
   This is an **ambiguous / undeclared license**. `GroundTruthV5` is **not**
   incorporated in this PR (it is Wave 1a). **Do not copy SkillForge code until
   its license is clarified in writing** — a bare `package.json` "MIT" string
   without license text is not a reliable grant for a Python module copied out
   of the repo.

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
- SkillForge `/forge`, `/forge-author` skills — methodology reference only.
- ai-dev-kit builder app — UI reference only.
- DSPy `RLM`, HALO — design inspiration only (deferred).

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
