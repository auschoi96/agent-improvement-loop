"""The L0 metrics **output contract** — a stable, typed, JSON-shaped schema.

This module is a first-class deliverable. A downstream UI app reads *exactly*
these shapes, so they are versioned (:data:`SCHEMA_VERSION`) and changed only
deliberately. Everything is a pydantic v2 model so the contract is
self-validating and round-trips through JSON (`model_dump_json` /
`model_validate_json`) without custom (de)serialization.

Design rules:

* **Deterministic fields only.** Every value is derived mechanically from a
  :class:`~ail.ingest.base.NormalizedTrace` (token counts, timestamps, tool
  spans) — no model, no judgement, nothing an agent can inflate to look good.
* **Cost is honest about uncertainty.** Token counts are facts;
  dollar costs depend on a price mapping that may be incomplete or approximate.
  Cost carries a ``priced`` flag and ``flags`` so an unknown price is surfaced,
  never fabricated (see :class:`CostBreakdown` / :class:`CostAggregate`).
* **Two views of tool reuse.** :attr:`ToolRedundancy.redundancy_rate` is the
  strict, un-gameable rate of byte-identical repeated calls;
  :attr:`ToolRedundancy.repeated_calls` is the softer "same target / same shell
  prologue" diagnostic that names *what* was repeated.

The JSON shape is documented in ``docs/L0_METRICS_CONTRACT.md``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

#: Version of the L0 output contract. Bump the minor for additive,
#: backward-compatible fields; bump the major for breaking shape changes.
SCHEMA_VERSION = "l0.metrics/v1"


class _Contract(BaseModel):
    """Base for every contract model: forbid unknown fields so drift is loud."""

    model_config = ConfigDict(extra="forbid")


class TokenBreakdown(_Contract):
    """Token accounting for one trace, or summed across a group of traces.

    ``total_tokens`` is the producer-preferred total (it can differ from
    ``input + output`` when the producer reports a total that folds cache
    tokens in differently). Cache fields are kept separate because they are
    billed at different rates and matter to the cost estimate.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_total_tokens: int = 0


class TokenStats(_Contract):
    """Distribution of per-trace ``total_tokens`` across a set of traces.

    The dominant L0 finding for the reference corpus is that token usage is
    *bimodal* — a low median with a long tail of very large sessions — so the
    aggregate carries the distribution, not just the sum.
    """

    count: int = 0
    min: int = 0
    median: float = 0.0
    mean: float = 0.0
    p90: float = 0.0
    max: int = 0


class CostBreakdown(_Contract):
    """Estimated USD cost for one trace, split by token class.

    ``priced`` is ``False`` when the trace's model is not in the price book; in
    that case all dollar amounts are ``0.0`` and ``flags`` explains why — the
    cost is **not** fabricated. ``priced`` being ``True`` means every dollar
    figure was computed from a known price (recorded in
    :class:`PriceBookEntry`).
    """

    input_usd: float = 0.0
    output_usd: float = 0.0
    cache_write_usd: float = 0.0
    cache_read_usd: float = 0.0
    total_usd: float = 0.0
    priced: bool = False
    flags: list[str] = Field(default_factory=list)


class CostAggregate(_Contract):
    """Summed USD cost across a group of traces, with priced/unpriced counts.

    ``total_usd`` sums only the traces that could be priced; ``unpriced_traces``
    counts those excluded for lack of a price so the total is never silently
    understated without a signal.
    """

    input_usd: float = 0.0
    output_usd: float = 0.0
    cache_write_usd: float = 0.0
    cache_read_usd: float = 0.0
    total_usd: float = 0.0
    priced_traces: int = 0
    unpriced_traces: int = 0
    flags: list[str] = Field(default_factory=list)


class RepeatedCall(_Contract):
    """A tool-call "identity" that occurred more than once within a trace.

    ``signature_kind`` says how identity was decided:

    * ``"path"`` — file tools (Read/Edit/Write/…) keyed on ``file_path``
      (catches "the same path was read N times").
    * ``"shell"`` — shell tools keyed on the normalized command prologue
      (catches "the same ``cd``/setup boilerplate was re-run N times").
    * ``"args"`` — every other tool keyed on its exact arguments.

    ``identity`` is the human-readable key (the file path, the normalized
    prologue, or an args preview).
    """

    tool: str
    identity: str
    count: int
    signature_kind: str


class ToolRedundancy(_Contract):
    """Tool-call reuse for one trace, or summed across a group.

    Two distinct signals:

    * :attr:`redundancy_rate` — the **strict** rate of byte-identical repeated
      calls (``redundant_tool_calls / total_tool_calls``). Un-gameable: it only
      counts calls whose tool name *and* full arguments are identical.
    * :attr:`repeated_calls` — the **diagnostic** list of repeated identities
      (same file path / same shell prologue / same args), which surfaces the
      softer "re-read the same file" and "re-ran the setup boilerplate" patterns
      even when the full commands differ.
    """

    total_tool_calls: int = 0
    distinct_tool_calls: int = 0
    redundant_tool_calls: int = 0
    redundancy_rate: float = 0.0
    repeated_calls: list[RepeatedCall] = Field(default_factory=list)


class TraceMetrics(_Contract):
    """Deterministic L0 metrics for a single trace."""

    trace_id: str
    producer: str | None = None
    model: str | None = None
    session_id: str | None = None
    status: str = "UNKNOWN"
    request_time: str | None = None  # ISO-8601
    duration_seconds: float | None = None
    tokens: TokenBreakdown
    cost: CostBreakdown
    total_tool_calls: int = 0
    tool_counts: dict[str, int] = Field(default_factory=dict)
    redundancy: ToolRedundancy


class GroupMetrics(_Contract):
    """Metrics summed over a slice of the corpus (one model / producer / status)."""

    key: str
    n_traces: int = 0
    tokens: TokenBreakdown
    cost: CostAggregate
    total_tool_calls: int = 0


class AggregateMetrics(_Contract):
    """Corpus-wide L0 metrics across all traces in a report."""

    n_traces: int = 0
    tokens: TokenBreakdown
    token_stats: TokenStats
    cost: CostAggregate
    total_tool_calls: int = 0
    redundancy: ToolRedundancy
    status_counts: dict[str, int] = Field(default_factory=dict)


class PriceBookEntry(_Contract):
    """A single model's USD price per million tokens, with provenance.

    ``source`` and ``confidence`` are first-class so consumers know how much to
    trust a dollar figure. ``confidence`` is ``"high" | "medium" | "low"``.
    Cache rates follow Anthropic's documented multipliers (cache read ≈ 0.1×
    input; cache write ≈ 1.25× input at the 5-minute TTL).
    """

    model: str
    input_usd_per_mtok: float
    output_usd_per_mtok: float
    cache_write_usd_per_mtok: float
    cache_read_usd_per_mtok: float
    source: str
    confidence: str
    notes: str = ""


class L0MetricsReport(_Contract):
    """Top-level L0 report: the stable artifact a UI app consumes.

    ``traces`` is the per-trace detail; ``aggregate`` and the ``by_*``
    breakdowns are pre-computed rollups so a reader need not re-aggregate.
    ``pricebook`` and ``pricing_flags`` make every dollar figure auditable —
    which prices were used, their provenance, and any caveat (unpriced models,
    cache-TTL assumption).
    """

    schema_version: str = SCHEMA_VERSION
    experiment_id: str | None = None
    generated_at: str | None = None  # ISO-8601
    n_traces: int = 0
    aggregate: AggregateMetrics
    by_model: list[GroupMetrics] = Field(default_factory=list)
    by_producer: list[GroupMetrics] = Field(default_factory=list)
    by_status: list[GroupMetrics] = Field(default_factory=list)
    traces: list[TraceMetrics] = Field(default_factory=list)
    pricebook: list[PriceBookEntry] = Field(default_factory=list)
    pricing_flags: list[str] = Field(default_factory=list)
