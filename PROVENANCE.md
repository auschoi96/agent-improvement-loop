# Provenance & License Reconciliation

This project **harvests** code from other repositories. Every harvested file
must be attributed here with its source path and the upstream license, and any
license incompatibility must be flagged before the code is merged. Until this
reconciliation is complete, the repository carries **no top-level LICENSE**.

A license-provenance check is an explicit **Wave 0** deliverable.

## Harvest sources

| Upstream repo | Module harvested | Upstream license | Status |
|---|---|---|---|
| `databricks-solutions/ai-dev-kit` @ `c4947868` | `.test/src/skill_test/trace/mlflow_integration.py` → `src/ail/ingest/mlflow_source.py` | **Databricks "DB license"** (proprietary; `LICENSE.md`) | harvested in this PR |
| `databricks-solutions/ai-dev-kit` @ `c4947868` | `.test/src/skill_test/agent/executor.py` → `src/ail/ingest/adapters/claude_code.py` | **Databricks "DB license"** (proprietary; `LICENSE.md`) | harvested in this PR |
| `databricks-field-eng/skillforge` @ `3569232e` | `python/skillforge/eval/schema.py` (`GroundTruthV5`) | **Undeclared** — no `LICENSE` file in repo; `package.json` says `"MIT"` (unverified, no license text) | **NOT yet harvested** (Wave 1a) — see flag below |

Commit pins verified by direct clone/read on 2026-06-28:
- `ai-dev-kit` HEAD `c4947868f06fbfbb8cb666cbfba15888127b8a3a` (2026-06-25).
- `skillforge` HEAD `3569232e72bfaf93d7b38cd22db95612af35e979` (2026-05-28).

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

Cost prices in `l0_deterministic.py` are *data*, not code: base input/output
rates are attributed to the Claude API pricing reference (the `claude-api`
skill's cached model table, dated 2026-06-04) on each `PriceBookEntry.source`,
cache rates are derived from Anthropic's documented prompt-caching multipliers,
and any uncovered model is flagged rather than guessed.

## ⚠️ License flags (resolve before adding a top-level LICENSE)

1. **ai-dev-kit is under the Databricks "DB license", NOT an OSI open-source
   license.** Full text in this repo's git history is the upstream
   `LICENSE.md`. Key terms relevant to harvesting:
   - **Scope is restricted to use "in connection with your use of the
     Databricks Services."** This is *not* a permissive grant — it is bounded
     by the Databricks MCSA.
   - **Redistribution is permitted** but with obligations: recipients must get
     a copy of the license; **modified files must carry prominent notices that
     they were changed**; copyright/attribution notices must be retained; any
     `NOTICE` file must be reproduced.
   - **Liability is capped at $1,000 and warranties are disclaimed.**
   - **Databricks can terminate the license at any time on notice**, after
     which copies must be deleted.
   - **Implication:** this repo cannot be relicensed as MIT/Apache while it
     contains DB-licensed code. Either (a) keep the repo under the DB license
     and propagate it (with the "changed files" notices — satisfied by the
     `CHANGES FROM UPSTREAM` headers on every harvested file), or (b)
     reimplement the harvested modules clean-room before any permissive
     release. **For an internal Databricks Field-Eng repo used with the
     Databricks Services, option (a) is compatible.** This must be confirmed by
     the human before a public/OSS release.

2. **skillforge has no declared license file.** The repo (private,
   Databricks-Field-Eng-internal) ships **no `LICENSE`/`COPYING`/`NOTICE`** at
   its root; only `package.json` claims `"license": "MIT"`, with no
   accompanying MIT license text and no `license` field in `pyproject.toml`.
   This is an **ambiguous / undeclared license**. `GroundTruthV5` is **not**
   harvested in this PR (it is Wave 1a). **Do not copy SkillForge code until
   its license is clarified in writing** — a bare `package.json` "MIT" string
   without license text is not a reliable grant for a Python module copied out
   of the repo.

## Reference-only (not copied)

- ai-dev-kit `.test/src/skill_test/trace/source.py` + `parser.py` — the
  Claude-Code-coupled local-JSONL/autolog path. **Deliberately NOT harvested**
  (the architecture's SKIP/REPLACE row): `mlflow_source.py` reads only the
  MLflow Traces API and is producer-agnostic.
- SkillForge `/forge`, `/forge-author` skills — methodology reference only.
- ai-dev-kit builder app — UI reference only.
- DSPy `RLM`, HALO — design inspiration only (deferred).

## Rules

1. Every copied file gets a header comment naming its upstream source path and
   commit. **Done** for both harvested files (see the `HARVEST` docstring at
   the top of `mlflow_source.py` and `claude_code.py`), each with an explicit
   `CHANGES FROM UPSTREAM` section — which also satisfies the DB license's
   "modified files must carry prominent notices that they were changed"
   obligation.
2. No upstream license obligation is dropped. If an upstream license is
   incompatible with this repo's intended license, the code is reimplemented or
   the dependency is added instead of copied.
3. This table is updated in the same PR that introduces the harvested code.
