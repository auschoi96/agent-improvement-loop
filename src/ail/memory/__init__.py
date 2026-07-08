"""Advisory memory: the read/injection spike and the write system-of-record.

This package holds two complementary halves of advisory memory.

The read/injection spike (Phase A-0) proves value before complexity: does
injecting the RLM's accumulated learnings as advisory context help the agent on
the frozen ``phase2-mini`` suite? It mirrors the Phase-2 token lever exactly and
runs entirely inside the existing controlled harness. Its pieces are
``MemoryInjectionIntervention`` (appends an advisory learnings block to the
candidate's system prompt; empty memory is a fail-closed no-op equal to the
baseline), ``ail.memory.source`` (reads the RLM roll-up and formats the top-k by
recurrence), ``ail.memory.config`` (the ``LeverConfig`` the runner drives against
the baseline), and ``ail.memory.provenance`` (the teaching-to-the-test guard
reusing the ``ail.pools`` disjointness wall).

The write side is the distiller's system-of-record half. A scheduled Databricks
Job distills recent RLM/HALO and L2 judge feedback into short guideline rows and
writes the survivors to a Unity Catalog Delta table. Its schema lives in
``ail.memory.schema`` (``MEMORY_TABLE``, ``MemoryRow``, and the watermark table
backing idempotent re-runs). The same provenance wall drops any row whose source
traces overlap the frozen eval pools, so eval-derived guidance can never
contaminate the memory store.
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
from ail.memory.schema import (
    MEMORY_COLUMNS,
    MEMORY_TABLE,
    WATERMARK_TABLE,
    MemoryRow,
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
    "MEMORY_COLUMNS",
    "MEMORY_TABLE",
    "WATERMARK_TABLE",
    "MemoryRow",
]
