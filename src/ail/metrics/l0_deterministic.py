"""Compute L0 deterministic metrics from normalized traces.

This is original work for the agent-improvement-loop project (see
``PROVENANCE.md``). It consumes :class:`~ail.ingest.base.NormalizedTrace`
records produced by the Wave 0 ingestion seam and produces the typed
:mod:`ail.metrics.contract` report. No code is copied from any upstream repo;
the only external surface used is pydantic and the project's own ingest types.

What "L0" means here: every number is mechanically derived from trace metadata
that the producer already emitted — token counts, timestamps, tool spans. There
is no model in the loop, so the numbers cannot be inflated by the agent under
test. The one place judgement enters is **cost**: dollars require a
model→price mapping. That mapping is explicit, versioned, and configurable, and
any model it does not cover is flagged rather than guessed (see
:data:`DEFAULT_PRICEBOOK` and :func:`compute_cost`).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

from ail.ingest.base import NormalizedTrace, TokenUsage, ToolCall
from ail.metrics.contract import (
    AggregateMetrics,
    CostAggregate,
    CostBreakdown,
    GroupMetrics,
    L0MetricsReport,
    PriceBookEntry,
    RepeatedCall,
    TokenBreakdown,
    TokenStats,
    ToolRedundancy,
    TraceMetrics,
)

# ---------------------------------------------------------------------------
# Price book
# ---------------------------------------------------------------------------
#
# Prices are USD per **million** tokens. Base input/output rates are sourced
# from the Claude API pricing reference (the `claude-api` skill's cached model
# table, dated 2026-06-04) — they are NOT invented here. Cache rates follow
# Anthropic's documented prompt-caching multipliers: cache *read* ≈ 0.1x the
# input rate, cache *write* ≈ 1.25x the input rate at the default 5-minute
# ephemeral TTL (a 1-hour TTL would be 2.0x; MLflow traces do not record which
# TTL was used, so 5-minute is assumed and flagged — see ``compute_l0``).
#
# Any model NOT in this book is left UNPRICED and flagged. Override or extend
# the book by passing your own ``pricebook`` to :func:`compute_l0`.

_PRICE_SOURCE = "claude-api skill model pricing table (cached 2026-06-04)"
_CACHE_READ_MULT = 0.1
_CACHE_WRITE_MULT_5M = 1.25


def _entry(model: str, in_rate: float, out_rate: float, confidence: str) -> PriceBookEntry:
    return PriceBookEntry(
        model=model,
        input_usd_per_mtok=in_rate,
        output_usd_per_mtok=out_rate,
        cache_write_usd_per_mtok=round(in_rate * _CACHE_WRITE_MULT_5M, 4),
        cache_read_usd_per_mtok=round(in_rate * _CACHE_READ_MULT, 4),
        source=_PRICE_SOURCE,
        confidence=confidence,
        notes="cache rates derived from documented 5-min-TTL multipliers (write 1.25x, read 0.1x)",
    )


#: Default model→price mapping. Keys are canonical model ids (see
#: :func:`_canonicalize_model`). Confidence is ``"high"`` for the base
#: input/output rates taken directly from the pricing table; the cache-rate
#: caveat (derived multipliers, TTL assumed) lives in each entry's ``notes``.
DEFAULT_PRICEBOOK: dict[str, PriceBookEntry] = {
    "claude-opus-4-8": _entry("claude-opus-4-8", 5.0, 25.0, "high"),
    "claude-opus-4-7": _entry("claude-opus-4-7", 5.0, 25.0, "high"),
    "claude-opus-4-6": _entry("claude-opus-4-6", 5.0, 25.0, "high"),
    "claude-opus-4-5": _entry("claude-opus-4-5", 5.0, 25.0, "high"),
    "claude-sonnet-4-6": _entry("claude-sonnet-4-6", 3.0, 15.0, "high"),
    "claude-sonnet-4-5": _entry("claude-sonnet-4-5", 3.0, 15.0, "high"),
    "claude-haiku-4-5": _entry("claude-haiku-4-5", 1.0, 5.0, "high"),
    "claude-fable-5": _entry("claude-fable-5", 10.0, 50.0, "high"),
}

# Provider-routing prefixes that denote the SAME SKU as the bare model id.
# ``anthropic.`` is the Amazon Bedrock model-id prefix (per the Claude API
# reference, Bedrock ids take an ``anthropic.`` prefix) — same model, same price.
_PROVIDER_PREFIXES = ("anthropic.",)

#: Explicit alias -> canonical-model map. Only entries we can SOURCE as the same
#: SKU (same price) belong here. Source: the Claude API model table's "full ID"
#: column, where a dated snapshot is the same model as its bare alias.
#: NOTE: speed-tier variants like ``-fast`` are deliberately absent — fast mode
#: is billed at *premium* pricing (per the Claude API fast-mode notes), so it is
#: a different SKU and must fall through to unpriced+flagged, never be mapped
#: here without a cited price of its own.
_MODEL_ALIASES: dict[str, str] = {
    "claude-haiku-4-5-20251001": "claude-haiku-4-5",
    "claude-opus-4-5-20251101": "claude-opus-4-5",
    "claude-sonnet-4-5-20250929": "claude-sonnet-4-5",
}


def _canonicalize_model(model: str) -> str:
    """Lowercase, strip whitespace and a known provider-routing prefix.

    This performs **only** safe, same-SKU canonicalization: case folding and the
    documented ``anthropic.`` Bedrock prefix. It does NOT strip arbitrary date or
    speed-tier suffixes — dated snapshots resolve through the explicit
    :data:`_MODEL_ALIASES` table, and anything else (e.g. a ``-fast`` tier or an
    unknown vendor id) is left to fall through to unpriced in
    :func:`lookup_price`.
    """
    m = model.strip().lower()
    for prefix in _PROVIDER_PREFIXES:
        if m.startswith(prefix):
            m = m[len(prefix) :]
    return m


def lookup_price(model: str | None, pricebook: dict[str, PriceBookEntry]) -> PriceBookEntry | None:
    """Return the price-book entry for ``model``, or ``None`` if not covered.

    Resolution order: exact (canonicalized) key, then the explicit sourced
    :data:`_MODEL_ALIASES` table. A model that matches neither is **not** priced
    — no fuzzy suffix-stripping that could map a differently-billed variant onto
    a base rate.
    """
    if not model:
        return None
    canon = _canonicalize_model(model)
    if canon in pricebook:
        return pricebook[canon]
    aliased = _MODEL_ALIASES.get(canon)
    if aliased is not None:
        return pricebook.get(aliased)
    return None


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------


def compute_cost(
    usage: TokenUsage,
    model: str | None,
    pricebook: dict[str, PriceBookEntry],
) -> CostBreakdown:
    """Estimate USD cost for one trace's token usage.

    Costs are computed from the *component* token counts (input, output, cache
    creation, cache read) priced separately — never from a single blended
    total. If ``model`` is missing or has no price-book entry, returns an
    unpriced :class:`CostBreakdown` (``priced=False``, all dollars ``0.0``) with
    a flag explaining why — the cost is not fabricated.
    """
    if not model:
        return CostBreakdown(
            priced=False, flags=["trace has no recorded model; cost not estimated"]
        )
    entry = lookup_price(model, pricebook)
    if entry is None:
        return CostBreakdown(
            priced=False,
            flags=[f"model '{model}' is not in the price book; cost not estimated"],
        )

    def _usd(tokens: int, rate: float) -> float:
        return round(tokens / 1_000_000 * rate, 6)

    input_usd = _usd(usage.input_tokens, entry.input_usd_per_mtok)
    output_usd = _usd(usage.output_tokens, entry.output_usd_per_mtok)
    cache_write_usd = _usd(usage.cache_creation_input_tokens, entry.cache_write_usd_per_mtok)
    cache_read_usd = _usd(usage.cache_read_input_tokens, entry.cache_read_usd_per_mtok)
    total = round(input_usd + output_usd + cache_write_usd + cache_read_usd, 6)
    flags: list[str] = []
    if usage.cache_tokens > 0:
        flags.append("cache-write cost assumes 5-min TTL (1.25x); trace does not record the TTL")
    return CostBreakdown(
        input_usd=input_usd,
        output_usd=output_usd,
        cache_write_usd=cache_write_usd,
        cache_read_usd=cache_read_usd,
        total_usd=total,
        priced=True,
        flags=flags,
    )


# ---------------------------------------------------------------------------
# Tool-call redundancy
# ---------------------------------------------------------------------------

_FILE_TOOLS = frozenset({"Read", "Edit", "Write", "MultiEdit", "NotebookEdit"})
_SHELL_TOOLS = frozenset({"Bash", "BashOutput"})
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE)
_FIELD_SEP = "\x1f"  # unit separator: safe in-band delimiter for composite keys


def _canonical_args(arguments: Any) -> str:
    """Deterministically serialize tool arguments for byte-identity matching."""
    try:
        return json.dumps(arguments, sort_keys=True, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(arguments)


def exact_signature(tc: ToolCall) -> str:
    """Strict identity: tool name + byte-identical canonical arguments.

    Two calls share this signature only if they are the same tool with exactly
    the same arguments — the basis of the un-gameable redundancy rate.
    """
    return f"{tc.name}{_FIELD_SEP}{_canonical_args(tc.arguments)}"


def normalize_command(command: str) -> str:
    """Reduce a shell command to its recurring setup prologue.

    Returns the first non-empty line with volatile per-session temp/scratch
    UUIDs collapsed to ``<id>``. Coding agents re-issue the same working-dir /
    environment prologue (``cd <repo>`` / ``export …``) before otherwise
    different commands; collapsing to the prologue is what surfaces that
    "boilerplate re-run" as a repeated identity.
    """
    stripped = command.strip()
    if not stripped:
        return ""
    first_line = stripped.splitlines()[0]
    return _UUID.sub("<id>", first_line)[:160]


def identity_signature(tc: ToolCall) -> tuple[str, str, str]:
    """Semantic identity of a tool call: ``(signature, kind, display)``.

    * file tools → keyed on ``file_path`` (kind ``"path"``)
    * shell tools → keyed on the normalized command prologue (kind ``"shell"``)
    * everything else → keyed on exact arguments (kind ``"args"``)
    """
    args = tc.arguments if isinstance(tc.arguments, dict) else {}
    if tc.name in _FILE_TOOLS and args.get("file_path"):
        path = str(args["file_path"])
        return (f"{tc.name}{_FIELD_SEP}path{_FIELD_SEP}{path}", "path", path)
    if tc.name in _SHELL_TOOLS and args.get("command"):
        norm = normalize_command(str(args["command"]))
        return (f"{tc.name}{_FIELD_SEP}shell{_FIELD_SEP}{norm}", "shell", norm)
    return (exact_signature(tc), "args", _canonical_args(args)[:160])


def compute_redundancy(tool_calls: Sequence[ToolCall], *, top_repeats: int = 20) -> ToolRedundancy:
    """Compute strict redundancy rate + the repeated-identity diagnostic.

    The rate is over *exact* signatures (un-gameable); ``repeated_calls`` lists
    semantic identities (same path / same shell prologue / same args) that
    occurred at least twice, sorted by count descending, capped at
    ``top_repeats``.
    """
    total = len(tool_calls)
    exact_counts = Counter(exact_signature(tc) for tc in tool_calls)
    distinct = len(exact_counts)
    redundant = total - distinct
    rate = round(redundant / total, 6) if total else 0.0

    id_counts: Counter[str] = Counter()
    representative: dict[str, tuple[str, str]] = {}  # sig -> (tool, display)
    kind_of: dict[str, str] = {}
    for tc in tool_calls:
        sig, kind, display = identity_signature(tc)
        id_counts[sig] += 1
        if sig not in representative:
            representative[sig] = (tc.name, display)
            kind_of[sig] = kind

    repeated = [
        RepeatedCall(
            tool=representative[sig][0],
            identity=representative[sig][1],
            count=count,
            signature_kind=kind_of[sig],
        )
        for sig, count in id_counts.items()
        if count >= 2
    ]
    repeated.sort(key=lambda r: (-r.count, r.tool, r.identity))

    return ToolRedundancy(
        total_tool_calls=total,
        distinct_tool_calls=distinct,
        redundant_tool_calls=redundant,
        redundancy_rate=rate,
        repeated_calls=repeated[:top_repeats],
    )


# ---------------------------------------------------------------------------
# Per-trace metrics
# ---------------------------------------------------------------------------


def _token_breakdown(usage: TokenUsage) -> TokenBreakdown:
    return TokenBreakdown(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        cache_creation_input_tokens=usage.cache_creation_input_tokens,
        cache_read_input_tokens=usage.cache_read_input_tokens,
        cache_total_tokens=usage.cache_tokens,
    )


def compute_trace_metrics(
    trace: NormalizedTrace,
    *,
    pricebook: dict[str, PriceBookEntry] | None = None,
    top_repeats: int = 20,
) -> TraceMetrics:
    """Compute the full L0 metric record for a single normalized trace."""
    book = pricebook if pricebook is not None else DEFAULT_PRICEBOOK
    return TraceMetrics(
        trace_id=trace.trace_id,
        producer=trace.producer,
        model=trace.model,
        session_id=trace.session_id,
        status=str(trace.status),
        request_time=trace.request_time.isoformat() if trace.request_time else None,
        duration_seconds=trace.duration_seconds,
        tokens=_token_breakdown(trace.token_usage),
        cost=compute_cost(trace.token_usage, trace.model, book),
        total_tool_calls=trace.total_tool_calls,
        tool_counts=trace.tool_counts,
        redundancy=compute_redundancy(trace.tool_calls, top_repeats=top_repeats),
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _sum_tokens(metrics: Iterable[TraceMetrics]) -> TokenBreakdown:
    out = TokenBreakdown()
    for m in metrics:
        t = m.tokens
        out.input_tokens += t.input_tokens
        out.output_tokens += t.output_tokens
        out.total_tokens += t.total_tokens
        out.cache_creation_input_tokens += t.cache_creation_input_tokens
        out.cache_read_input_tokens += t.cache_read_input_tokens
        out.cache_total_tokens += t.cache_total_tokens
    return out


def _sum_cost(metrics: Iterable[TraceMetrics]) -> CostAggregate:
    agg = CostAggregate()
    for m in metrics:
        if m.cost.priced:
            agg.input_usd = round(agg.input_usd + m.cost.input_usd, 6)
            agg.output_usd = round(agg.output_usd + m.cost.output_usd, 6)
            agg.cache_write_usd = round(agg.cache_write_usd + m.cost.cache_write_usd, 6)
            agg.cache_read_usd = round(agg.cache_read_usd + m.cost.cache_read_usd, 6)
            agg.total_usd = round(agg.total_usd + m.cost.total_usd, 6)
            agg.priced_traces += 1
        else:
            agg.unpriced_traces += 1
    if agg.unpriced_traces:
        agg.flags.append(
            f"{agg.unpriced_traces} trace(s) unpriced (model missing or not in price book); "
            "their tokens are counted but excluded from total_usd"
        )
    return agg


def _percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return round(s[lo] * (1 - frac) + s[hi] * frac, 2)


def _token_stats(metrics: Sequence[TraceMetrics]) -> TokenStats:
    totals = [m.tokens.total_tokens for m in metrics]
    if not totals:
        return TokenStats()
    return TokenStats(
        count=len(totals),
        min=min(totals),
        median=_percentile(totals, 0.5),
        mean=round(sum(totals) / len(totals), 2),
        p90=_percentile(totals, 0.9),
        max=max(totals),
    )


def _aggregate_redundancy(metrics: Sequence[TraceMetrics], *, top_repeats: int) -> ToolRedundancy:
    total = sum(m.redundancy.total_tool_calls for m in metrics)
    distinct = sum(m.redundancy.distinct_tool_calls for m in metrics)
    redundant = sum(m.redundancy.redundant_tool_calls for m in metrics)
    rate = round(redundant / total, 6) if total else 0.0
    # Top repeated identities across the whole corpus (per-trace identities, not
    # merged across traces — the same path in two sessions is different work).
    all_repeats: list[RepeatedCall] = [r for m in metrics for r in m.redundancy.repeated_calls]
    all_repeats.sort(key=lambda r: (-r.count, r.tool, r.identity))
    return ToolRedundancy(
        total_tool_calls=total,
        distinct_tool_calls=distinct,
        redundant_tool_calls=redundant,
        redundancy_rate=rate,
        repeated_calls=all_repeats[:top_repeats],
    )


def _group(metrics: Sequence[TraceMetrics], key_fn: Any, *, none_key: str) -> list[GroupMetrics]:
    buckets: dict[str, list[TraceMetrics]] = {}
    for m in metrics:
        key = key_fn(m)
        buckets.setdefault(str(key) if key is not None else none_key, []).append(m)
    groups = [
        GroupMetrics(
            key=key,
            n_traces=len(members),
            tokens=_sum_tokens(members),
            cost=_sum_cost(members),
            total_tool_calls=sum(m.total_tool_calls for m in members),
        )
        for key, members in buckets.items()
    ]
    groups.sort(key=lambda g: (-g.tokens.total_tokens, g.key))
    return groups


def compute_l0(
    traces: Sequence[NormalizedTrace],
    *,
    pricebook: dict[str, PriceBookEntry] | None = None,
    experiment_id: str | None = None,
    generated_at: str | None = None,
    top_repeats: int = 20,
) -> L0MetricsReport:
    """Compute the full L0 metrics report for a set of normalized traces.

    Args:
        traces: Normalized traces (from any :class:`~ail.ingest.base.TraceSource`).
        pricebook: Optional model→price override. Defaults to
            :data:`DEFAULT_PRICEBOOK`. Models absent from the book are flagged,
            never guessed.
        experiment_id: Recorded on the report for provenance.
        generated_at: ISO-8601 timestamp recorded on the report (caller-supplied
            so the computation itself stays deterministic).
        top_repeats: Cap on ``repeated_calls`` lists.

    Returns:
        A :class:`~ail.metrics.contract.L0MetricsReport`.
    """
    book = pricebook if pricebook is not None else DEFAULT_PRICEBOOK
    metrics = [compute_trace_metrics(t, pricebook=book, top_repeats=top_repeats) for t in traces]
    metrics.sort(key=lambda m: (-m.tokens.total_tokens, m.trace_id))

    status_counts = dict(Counter(m.status for m in metrics))

    aggregate = AggregateMetrics(
        n_traces=len(metrics),
        tokens=_sum_tokens(metrics),
        token_stats=_token_stats(metrics),
        cost=_sum_cost(metrics),
        total_tool_calls=sum(m.total_tool_calls for m in metrics),
        redundancy=_aggregate_redundancy(metrics, top_repeats=top_repeats),
        status_counts=status_counts,
    )

    pricing_flags: list[str] = []
    unpriced_models = sorted(
        {m.model or "(no model recorded)" for m in metrics if not m.cost.priced}
    )
    if unpriced_models:
        pricing_flags.append(
            "Unpriced models (tokens counted, cost omitted): " + ", ".join(unpriced_models)
        )
    if aggregate.tokens.cache_total_tokens > 0:
        pricing_flags.append(
            "Cache-write cost assumes the 5-min ephemeral TTL (1.25x input); "
            "MLflow traces do not record the cache TTL (1-hr would be 2.0x)."
        )
    pricing_flags.append(
        "Base input/output prices: " + _PRICE_SOURCE + ". Verify against live pricing before "
        "using dollar figures for billing decisions."
    )

    return L0MetricsReport(
        experiment_id=experiment_id,
        generated_at=generated_at,
        n_traces=len(metrics),
        aggregate=aggregate,
        by_model=_group(metrics, lambda m: m.model, none_key="<unknown-model>"),
        by_producer=_group(metrics, lambda m: m.producer, none_key="<unknown-producer>"),
        by_status=_group(metrics, lambda m: m.status, none_key="UNKNOWN"),
        traces=metrics,
        pricebook=list(book.values()),
        pricing_flags=pricing_flags,
    )
