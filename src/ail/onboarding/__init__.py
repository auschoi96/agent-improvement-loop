"""In-app onboarding wizard — the server-side of the "Add an agent" flow (slice 1).

This package is the Python **source of truth** the app's authenticated
``server/plugins/onboarding`` routes invoke (``docs/ONBOARDING_WIZARD.md`` slice 1).
It reuses the framework's existing capabilities rather than reimplementing them:

* the fixed goal catalog + goal→scorer mapping + data-gate requirements
  (:mod:`ail.onboarding.goals`, reusing :func:`ail.readiness.compute_readiness`);
* fresh-experiment validation / creation (:mod:`ail.onboarding.experiment`);
* agent registration (:mod:`ail.onboarding.service`, reusing
  :func:`ail.publish_versions.publish_registry`).

Nothing here is trusted from the browser and nothing fabricates a write: an
experiment is only "created" when MLflow created it, an agent is only "registered"
when the registry write succeeded (see the module docstrings for the fail-closed
contract). The single CLI entry ``python -m ail.onboarding.service`` is the bridge
the Node route drives (JSON action on stdin, typed JSON result on stdout).
"""

from __future__ import annotations

from ail.onboarding.experiment import (
    ExperimentAccessError,
    ExperimentClient,
    ExperimentCreation,
    ExperimentPermissionError,
    ExperimentValidation,
    create_experiment,
    validate_experiment,
)
from ail.onboarding.goals import (
    GOAL_CATALOG,
    GoalKey,
    RequirementsResult,
    ScorerKind,
    UnknownGoalError,
    build_judge_config,
    build_requirements,
)
from ail.onboarding.service import (
    CreationResult,
    OnboardingOutcome,
    RegisterResult,
    ValidationResult,
    register_agent,
    run_action,
    run_create,
    run_register,
    run_requirements,
    run_validate,
)

__all__ = [
    # goals
    "GOAL_CATALOG",
    "GoalKey",
    "ScorerKind",
    "RequirementsResult",
    "UnknownGoalError",
    "build_requirements",
    "build_judge_config",
    # experiment
    "ExperimentClient",
    "ExperimentValidation",
    "ExperimentCreation",
    "ExperimentAccessError",
    "ExperimentPermissionError",
    "validate_experiment",
    "create_experiment",
    # service
    "OnboardingOutcome",
    "ValidationResult",
    "CreationResult",
    "RegisterResult",
    "register_agent",
    "run_requirements",
    "run_validate",
    "run_create",
    "run_register",
    "run_action",
]
