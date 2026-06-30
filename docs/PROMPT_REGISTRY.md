# UC prompt registry (the human-promote step)

This is the **explicit, human-run promote step** of the loop: the bridge between a
GEPA candidate that a human has reviewed and a *versioned, provenance-stamped* prompt
in the **Unity-Catalog-backed MLflow Prompt Registry**.

- **Module:** `src/ail/optimize/prompt_registry.py` (offline-tested; mocks the
  registry client — no live call on import or in tests)
- **Registry:** Databricks-managed MLflow, registry URI `databricks-uc`
- **Default location:** `austin_choi_omni_agent_catalog.agent_improvement_loop`
  (configurable per call), prompt name `token_efficient_execution`

```
GEPA (Stage 5)                          prompt_registry.py (this step)
gepa_runner.run_gepa_optimization()     register_gepa_candidate(candidate.json)
        │                                       │  fail-closed gate:
        ▼                                       │  refuse unless evolved beat seed
artifacts/gepa_candidate.json  ──── human ────► │  on the HELD-OUT split
(CANDIDATE — never promoted)   reviews it       ▼
                                        mlflow.genai.register_prompt(name, body, tags)
                                        → new VERSION under catalog.schema.name,
                                          tagged with WHY it was promoted
```

## What the prompt registry is

The MLflow Prompt Registry version-controls prompt **bodies** the way the model
registry version-controls models: each `register_prompt(name, template, …)` call
creates a new immutable **version** under a name, with optional **aliases**
(`champion`, `production`, …) and per-version **tags**. On this workspace it is
UC-backed (registry URI `databricks-uc`), so a prompt is a governed UC object named
`catalog.schema.name` and is searchable with
`mlflow.genai.search_prompts(filter_string="catalog = '…' AND schema = '…'")`.

This loop's optimized artifact is the **token-efficiency skill body** — the same
markdown the Phase-2 lever injects into a candidate's system prompt
(`src/ail/optimize/lever.py`). Versioning that body here gives every promoted skill
a durable, queryable history with the evidence behind each promotion attached.

## Register-on-human-promote workflow

GEPA **never auto-promotes**. `run_gepa_optimization` writes a *candidate* artifact
(`artifacts/gepa_candidate*.json`, `human_gate_required=True`) and stops — see
`docs/GEPA_OPTIMIZATION.md`. This module is the separate step a human runs **after
reviewing that candidate**. It is not imported or called by `gepa_runner` or the
comparison harness.

```python
from ail.optimize.prompt_registry import register_gepa_candidate, register_seed_prompt

# 1. (Once) version the seed body the candidates are measured against.
register_seed_prompt(profile="dais-demo")

# 2. After reviewing artifacts/gepa_candidate.json, promote the evolved body.
#    Refuses unless it beat the seed on the held-out split (see fail-closed guard).
registered = register_gepa_candidate(
    "artifacts/gepa_candidate.json",
    profile="dais-demo",
)
print(registered.name, registered.version, registered.uri)

# 3. Promotion to "production" is a SEPARATE, explicit act — pass alias= to set it.
#    Registering never sets a production/champion alias on its own.
register_gepa_candidate("artifacts/gepa_candidate.json", alias="champion", profile="dais-demo")
```

`register_prompt_body` is the low-level primitive (register any body + provenance);
`register_seed_prompt` and `register_gepa_candidate` are the two convenience entry
points. All of them accept an injectable `client` so tests (and other tools) never
hit a live workspace; a live `databricks-uc` client is built only when `client` is
omitted and a human invokes the call at runtime.

## Provenance: why a version was promoted

Each registered version carries `ail.prompt.*` **tags** so a version records the
evidence behind it, not just the text. For a GEPA candidate they are read straight
from the `GepaOptimizationResult`:

| Tag | Meaning |
| --- | --- |
| `ail.prompt.source` | `seed` or `gepa-evolved` |
| `ail.prompt.suite_content_hash` | the frozen Task Suite the run was scored against |
| `ail.prompt.changed` | did the evolved body differ from the seed |
| `ail.prompt.gepa_best_val_score` | GEPA's best validation score |
| `ail.prompt.gepa_num_candidates` | how many candidates GEPA explored |
| `ail.prompt.holdout_evolved_promote` / `…_seed_promote` | held-out PROMOTE/total per arm |
| `ail.prompt.holdout_evolved_savings_pct` / `…_seed_savings_pct` | realized held-out token savings per arm |
| `ail.prompt.holdout_savings_delta_pct` | evolved − seed (the anti-overfit headline) |
| `ail.prompt.candidate_artifact` | pointer back to the `gepa_candidate*.json` |
| `ail.prompt.improving` | did it pass the fail-closed gate |
| `ail.prompt.forced` / `ail.prompt.registration_reason` | present only on a forced, non-improving registration |

## Fail-closed guard: it refuses to register a non-improvement

`register_gepa_candidate` will **refuse** (raise `NonImprovingCandidateError`)
rather than register a candidate that did not actually beat the seed. The decision
is `candidate_improvement(result)`, which requires **all** of:

1. `changed is True` — the evolved body differs from the seed;
2. the candidate carries a live held-out validation for **both** the evolved and
   seed bodies; and
3. `holdout_savings_delta_pct` (evolved − seed realized token savings on the
   held-out split) is **strictly positive**.

Held-out savings are summed over PROMOTE tasks only (`Phase2Artifact`), so a positive
delta is the honest signal that the evolved body helped on tasks GEPA never trained
on — not a train-set or self-reported number. A candidate that is identical to seed,
has no held-out validation, or does not beat seed is refused with a reason.

`force=True` overrides the refusal for deliberate exceptions, but it does **not**
launder the result: the version is tagged `ail.prompt.forced=true` with the
`ail.prompt.registration_reason`, and its commit message is prefixed
`FORCE-registered non-improving GEPA candidate`, so a forced version can never
silently masquerade as an improvement.

## How this unlocks `mlflow.genai.optimize_prompts`

The live evolution loop deliberately calls `gepa.optimize` directly rather than
`mlflow.genai.optimize_prompts`, partly because `optimize_prompts` **requires the
prompts it optimizes to live in the Prompt Registry** while our artifact was a
free-text blob with no registry round-trip (`docs/GEPA_OPTIMIZATION.md`, "Why
`gepa.optimize`, not `mlflow.genai.optimize_prompts`"). Versioning the seed and the
evolved bodies here removes exactly that precondition: once the bodies are registered
prompts under `catalog.schema.name`, the `optimize_prompts` path becomes usable as a
future option (it can load a `PromptVersion`, call `.format`, and write back new
versions) — without changing the existing, fail-closed `gepa_runner` loop or its
human gate.

## Configuration

| Knob | Default | Notes |
| --- | --- | --- |
| `catalog` | `austin_choi_omni_agent_catalog` | UC catalog |
| `schema` | `agent_improvement_loop` | UC schema |
| `name` | `token_efficient_execution` | leaf name; the on-disk skill *slug* is `token-efficient-execution`, but a UC object name is a SQL identifier so the registered prompt uses the underscore form. Pass a full `catalog.schema.name` to override the prefix. |
| `profile` | unset | Databricks CLI profile, used only when a live client is built |
| `alias` | unset | set an alias (e.g. `champion`) **only** when passed explicitly |
| `client` | live `databricks-uc` | inject a fake to stay fully offline (tests do) |

The registered prompt **template is the skill body verbatim**. MLflow treats only
`{{double-brace}}` tokens as template variables; the skill bodies use none, so they
register as plain text prompts.
