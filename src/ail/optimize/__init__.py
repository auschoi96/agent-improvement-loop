"""The optimization lever layer: framework-generated assets + their wiring.

This is the ``optimize/`` step of the loop (``docs/ARCHITECTURE.md`` §4): given a
diagnosed weakness, build a helper asset for the agent under improvement and wire
it as a candidate the frozen-suite comparison harness can evaluate against the
baseline.

The first lever (Phase 2) targets the evidence-confirmed dominant weakness —
redundant file reads and repeated shell/`cd` boilerplate (token_efficiency scored
2.17/5 across 29 traces; L0 found same-path re-reads and up to 27× repeated `cd`
per session). It ships:

* :mod:`ail.optimize.assets` — the generated **skill** asset
  (``token-efficient-execution``) and its loader;
* :mod:`ail.optimize.lever` — the :class:`~ail.optimize.lever.SkillInjectionIntervention`
  and the :data:`~ail.optimize.lever.BASELINE` / :data:`~ail.optimize.lever.CANDIDATE`
  configs that wire it into :func:`ail.compare.compare_candidate`;
* :mod:`ail.optimize.phase2` — the runner that drives a baseline-vs-candidate
  comparison across the frozen Task Suite with a deterministic **L1 programmatic**
  correctness guardrail (no LLM judge), and emits the
  :class:`~ail.optimize.phase2.Phase2Artifact`;
* :mod:`ail.optimize.fixtures` — the live-fixture loader (``seed/`` + ``verify/``)
  and per-arm isolated-workspace lifecycle the runner uses to run real,
  file-mutating tasks honestly (per-arm isolation + tamper-proof verification).
"""

from __future__ import annotations

from ail.optimize.assets import (
    TOKEN_EFFICIENCY_SKILL,
    SkillAsset,
    load_skill_asset,
    skill_asset_path,
)
from ail.optimize.fixtures import (
    TaskFixture,
    fixture_dir,
    load_fixture,
    phase2_fixtures_root,
)
from ail.optimize.gepa_runner import (
    DEFAULT_COMPONENT,
    DEFAULT_REFLECTION_LM,
    GEPA_SCHEMA_VERSION,
    FrozenSuiteGepaAdapter,
    GepaConfig,
    GepaEvalBatch,
    GepaOptimizationResult,
    HeldOutLeakError,
    SuiteSplit,
    candidate_lever_config,
    fitness_from_outcome,
    run_gepa_optimization,
    split_suite,
)
from ail.optimize.lever import (
    BASELINE,
    CANDIDATE,
    LeverConfig,
    SkillInjectionIntervention,
    token_efficiency_intervention,
    token_efficiency_skill,
)
from ail.optimize.phase2 import (
    L1Outcome,
    Phase2Artifact,
    TaskOutcome,
    VerifySpec,
    case_from_task,
    make_command_check,
    run_phase2_comparison,
)

__all__ = [
    # assets
    "SkillAsset",
    "load_skill_asset",
    "skill_asset_path",
    "TOKEN_EFFICIENCY_SKILL",
    # live fixtures
    "TaskFixture",
    "load_fixture",
    "fixture_dir",
    "phase2_fixtures_root",
    # lever
    "SkillInjectionIntervention",
    "LeverConfig",
    "BASELINE",
    "CANDIDATE",
    "token_efficiency_skill",
    "token_efficiency_intervention",
    # phase2 runner + artifact
    "run_phase2_comparison",
    "Phase2Artifact",
    "TaskOutcome",
    "VerifySpec",
    "L1Outcome",
    "case_from_task",
    "make_command_check",
    # GEPA optimization loop (stage 5)
    "run_gepa_optimization",
    "GepaConfig",
    "GepaOptimizationResult",
    "FrozenSuiteGepaAdapter",
    "GepaEvalBatch",
    "SuiteSplit",
    "split_suite",
    "candidate_lever_config",
    "fitness_from_outcome",
    "HeldOutLeakError",
    "GEPA_SCHEMA_VERSION",
    "DEFAULT_REFLECTION_LM",
    "DEFAULT_COMPONENT",
]
