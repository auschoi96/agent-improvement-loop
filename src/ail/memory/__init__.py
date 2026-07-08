"""Advisory memory — distill recent agent-evaluation feedback into governed
"memory guideline" rows written to a Unity Catalog Delta table.

This is the WRITE / system-of-record half only. A scheduled Databricks Job
(:mod:`ail.memory.distiller`, wired by ``resources/memory_distiller.job.yml``)
reads recent RLM/HALO ``rlm_*`` assessments and the L2 MLflow-judge assessments
(``correctness`` / ``modularity`` / ``groundedness`` / ``token_efficiency``) off
the trace store, drives the Claude Agent SDK to distill them into short guideline
rows, and writes the surviving rows to
``<catalog>.<schema>.agent_memory`` (:data:`ail.memory.schema.MEMORY_TABLE`).

Two load-bearing guarantees live here:

* **The provenance wall** (:mod:`ail.memory.provenance`) — a memory row whose
  ``source_trace_ids`` overlap the frozen Task-Suite or Human-Anchor pools is
  DROPPED (never written), reusing :func:`ail.pools.assert_pools_disjoint` so
  eval-set-derived guidance can never contaminate the memory store.
* **Watermarked idempotency** (:mod:`ail.memory.watermark`) — only feedback new
  since the last successful run is distilled, so a re-run over the same window
  writes no duplicate rows.

The Lakebase sync and the read / injection side are deliberately out of scope.
"""

from ail.memory.schema import (
    MEMORY_COLUMNS,
    MEMORY_TABLE,
    WATERMARK_TABLE,
    MemoryRow,
)

__all__ = [
    "MEMORY_COLUMNS",
    "MEMORY_TABLE",
    "WATERMARK_TABLE",
    "MemoryRow",
]
