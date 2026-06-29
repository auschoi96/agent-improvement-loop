"""L3 — recursive trace deep-review (the adopted HALO path).

This package is the **L3 — RLM deep review** tier of the layered metrics design
(``docs/ARCHITECTURE.md`` §3, §11). It reads *whole* long traces — this corpus
reaches 943K tokens, far past any single judge call's context — to **discover**
token waste and failure modes a fixed scorer would miss, used to decide *what to
fix*, never to score the leaderboard.

It does this by **adopting HALO** (``halo-engine``, MIT — see ``PROVENANCE.md`` /
``NOTICE``), a trace-specialized Recursive LM that indexes a flat trace file by
byte offset and navigates it with bounded tools, recursive subagents, and
compaction. AIL contributes a *thin* layer around the engine, not a
reimplementation:

* :mod:`ail.l3.adapter` — MLflow trace → OpenInference/OTLP ``SpanRecord`` JSONL.
* :mod:`ail.l3.contract` — the structured verdict (:class:`HaloReviewVerdict`).
* :mod:`ail.l3.parser` — HALO's free-text ``<final/>`` report → verdict.
* :mod:`ail.l3.reviewer` — run HALO under its **own** trace (token isolation),
  parse, and attach the verdict to the subject trace as an ``LLM_JUDGE``
  feedback assessment linked by ``reviewer_trace_id``.
* :mod:`ail.l3.selection` — choose which (biggest/most-interesting) traces to
  review, because L3 is expensive.

The HALO engine itself is the optional ``l3`` extra (``pip install 'ail[l3]'``)
and is lazy-imported, so importing this package never requires it.
"""

from __future__ import annotations

from ail.l3.adapter import (
    OtlpExport,
    mlflow_trace_to_otlp_jsonl,
    normalized_trace_to_span_records,
    write_span_records_jsonl,
)
from ail.l3.contract import (
    SCHEMA_VERSION,
    FailureMode,
    HaloReviewVerdict,
    RedundancyFinding,
)
from ail.l3.parser import parse_halo_report, strip_final_marker
from ail.l3.reviewer import (
    FEEDBACK_NAME,
    REVIEW_PROMPT_TEMPLATE,
    build_engine_config,
    review_trace,
    run_halo_review,
)
from ail.l3.selection import (
    TraceSelection,
    select_from_experiment,
    select_traces_to_review,
)

__all__ = [
    # contract
    "SCHEMA_VERSION",
    "HaloReviewVerdict",
    "RedundancyFinding",
    "FailureMode",
    # adapter
    "OtlpExport",
    "normalized_trace_to_span_records",
    "write_span_records_jsonl",
    "mlflow_trace_to_otlp_jsonl",
    # parser
    "parse_halo_report",
    "strip_final_marker",
    # reviewer
    "FEEDBACK_NAME",
    "REVIEW_PROMPT_TEMPLATE",
    "build_engine_config",
    "run_halo_review",
    "review_trace",
    # selection
    "TraceSelection",
    "select_traces_to_review",
    "select_from_experiment",
]
