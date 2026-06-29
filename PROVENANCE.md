# Provenance & License Reconciliation

This project **harvests** code from other repositories. Every harvested file
must be attributed here with its source path and the upstream license, and any
license incompatibility must be flagged before the code is merged. Until this
reconciliation is complete, the repository carries **no top-level LICENSE**.

A license-provenance check is an explicit **Wave 0** deliverable.

## Harvest sources

| Upstream repo | Module harvested | Upstream license | Status |
|---|---|---|---|
| `databricks-solutions/ai-dev-kit` | `.test/src/skill_test/` (GEPA runner, judges, alignment, trace ingestion, GRP, Claude Code adapter) | TODO — confirm during Wave 0 | pending |
| `databricks-field-eng/skillforge` | `eval/schema.py` (`GroundTruthV5`) | TODO — confirm during Wave 0 | pending |

## Reference-only (not copied)

- SkillForge `/forge`, `/forge-author` skills — methodology reference only.
- ai-dev-kit builder app — UI reference only.
- DSPy `RLM`, HALO — design inspiration only (deferred).

## Rules

1. Every copied file gets a header comment naming its upstream source path and
   commit.
2. No upstream license obligation is dropped. If an upstream license is
   incompatible with this repo's intended license, the code is reimplemented or
   the dependency is added instead of copied.
3. This table is updated in the same PR that introduces the harvested code.
