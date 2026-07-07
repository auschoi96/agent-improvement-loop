"""Phase A-0 advisory-memory spike: inject accumulated learnings as context.

A minimal "prove value before complexity" experiment: does injecting the RLM's
accumulated learnings as **advisory** context help the agent on the frozen
``phase2-mini`` suite? It runs entirely inside the existing controlled Phase-2
harness — **no** Lakebase, memory-writer job, store, deployer injection, or
organic traces (all deferred).

The pieces mirror the Phase-2 token lever exactly:

* :class:`~ail.memory.intervention.MemoryInjectionIntervention` — the sibling of
  :class:`ail.optimize.lever.SkillInjectionIntervention`: appends an advisory
  learnings block to the candidate's ``system_prompt`` and nothing else; empty
  memory is a fail-closed no-op equal to the baseline.
* :mod:`ail.memory.source` — reads the RLM roll-up
  (``artifacts/rlm_batch_report.json``), reusing the
  :class:`~ail.l3.contract.RankedAsset` schema, and formats the top-k (by
  recurrence) as advisory lines.
* :data:`~ail.memory.config.MEMORY_CANDIDATE` — the :class:`LeverConfig` the
  runner drives against the existing :data:`~ail.optimize.lever.BASELINE`.
* :func:`~ail.memory.provenance.assert_memory_disjoint_from_suite` — the
  teaching-to-the-test guard, reusing the :mod:`ail.pools` disjointness wall.
"""

from __future__ import annotations

from ail.memory.config import (
    MEMORY_CANDIDATE,
    MEMORY_CANDIDATE_NAME,
    build_memory_candidate,
    memory_injection_intervention,
)
from ail.memory.intervention import MemoryInjectionIntervention
from ail.memory.provenance import (
    assert_memory_disjoint_from_suite,
    memory_provenance_ids,
    task_suite_ids,
)
from ail.memory.source import (
    DEFAULT_REPORT_PATH,
    DEFAULT_TOP_K,
    build_memory_learnings,
    format_advisory_line,
    load_memory_learnings,
    load_ranked_assets,
    select_top_k,
)

__all__ = [
    "MemoryInjectionIntervention",
    "MEMORY_CANDIDATE",
    "MEMORY_CANDIDATE_NAME",
    "build_memory_candidate",
    "memory_injection_intervention",
    "DEFAULT_REPORT_PATH",
    "DEFAULT_TOP_K",
    "load_ranked_assets",
    "select_top_k",
    "format_advisory_line",
    "build_memory_learnings",
    "load_memory_learnings",
    "assert_memory_disjoint_from_suite",
    "memory_provenance_ids",
    "task_suite_ids",
]
