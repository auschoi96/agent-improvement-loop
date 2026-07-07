"""The advisory-memory lever config: :data:`MEMORY_CANDIDATE` for the spike.

The Phase A-0 sibling of :data:`ail.optimize.lever.CANDIDATE`. It reuses the same
:class:`~ail.optimize.lever.LeverConfig` shape and the existing
:data:`~ail.optimize.lever.BASELINE` (no asset), so the memory spike drives the
**unchanged** Phase-2 comparison machinery
(:func:`ail.optimize.phase2.run_phase2_comparison`): BASELINE vs
:data:`MEMORY_CANDIDATE` on the frozen suite. ``lever.py`` is left untouched — this
config lives here, additive-only.

**Fail-closed at import.** :data:`MEMORY_CANDIDATE` is built at import from the
default RLM report path. When that report is absent (CI, a fresh worktree) the
intervention carries **no** learnings and is a no-op equal to the baseline; import
never fails on a missing source. Point the runner at the real report (and tune
``k``) via :func:`build_memory_candidate`.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from ail.memory.intervention import MemoryInjectionIntervention
from ail.memory.source import DEFAULT_TOP_K, load_memory_learnings
from ail.optimize.lever import LeverConfig, SkillInjectionIntervention

__all__ = [
    "MEMORY_CANDIDATE_NAME",
    "memory_injection_intervention",
    "build_memory_candidate",
    "MEMORY_CANDIDATE",
]

#: Stable name of the advisory-memory candidate config (recorded for provenance).
MEMORY_CANDIDATE_NAME = "candidate-advisory-memory"

_DESCRIPTION = (
    "Advisory memory enabled: injects the top-k RLM learnings (ranked by cross-trace "
    "recurrence) as an advisory 'learnings from prior sessions' block in the system "
    "prompt. Empty/absent memory is a no-op identical to the baseline."
)


def memory_injection_intervention(
    path: str | Path | None = None, *, k: int = DEFAULT_TOP_K
) -> MemoryInjectionIntervention:
    """Build the :class:`~ail.memory.intervention.MemoryInjectionIntervention`.

    Loads the top-``k`` advisory learnings from the RLM report at ``path`` (the
    default location when ``None``); an absent report yields no learnings and a
    no-op intervention.
    """
    return MemoryInjectionIntervention(
        name="advisory-memory-rlm",
        learnings=load_memory_learnings(path, k=k),
    )


def build_memory_candidate(
    path: str | Path | None = None, *, k: int = DEFAULT_TOP_K
) -> LeverConfig:
    """Build the ``MEMORY_CANDIDATE`` :class:`~ail.optimize.lever.LeverConfig`.

    Reuses ``LeverConfig`` so the memory spike drives the unchanged Phase-2 runner.
    ``LeverConfig.intervention`` is *nominally* typed to ``SkillInjectionIntervention``,
    but the harness it feeds (:func:`ail.compare.compare_candidate`) accepts any
    :class:`~ail.compare.harness.Intervention`, and ``MemoryInjectionIntervention``
    is a sibling of that base. ``lever.py`` is intentionally left untouched (the
    spike is additive-only), so the ``cast`` bridges that nominal gap at this one
    site without widening the shared config's annotation.
    """
    intervention = memory_injection_intervention(path, k=k)
    return LeverConfig(
        name=MEMORY_CANDIDATE_NAME,
        intervention=cast(SkillInjectionIntervention, intervention),
        description=_DESCRIPTION,
    )


#: The advisory-memory candidate config (built from the default report at import).
MEMORY_CANDIDATE = build_memory_candidate()
