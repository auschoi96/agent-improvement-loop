"""Tests for L0 deterministic metrics (offline, no network).

The compute path is exercised both with hand-built normalized traces and via
the real MLflow normalization path (the ``synthetic_trace`` fixture), so the
metrics are tested against the exact shape produced for live traces.
"""

from __future__ import annotations

from typing import Any

from ail.ingest.base import NormalizedTrace, TokenUsage, ToolCall, TraceStatus
from ail.metrics.contract import SCHEMA_VERSION, L0MetricsReport
from ail.metrics.l0_deterministic import (
    DEFAULT_PRICEBOOK,
    compute_cost,
    compute_l0,
    compute_redundancy,
    compute_trace_metrics,
    exact_signature,
    lookup_price,
    normalize_command,
)


def _trace(
    *,
    trace_id: str = "t",
    model: str | None = "claude-opus-4-8",
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_creation: int = 0,
    cache_read: int = 0,
    total: int | None = None,
    tool_calls: list[ToolCall] | None = None,
    status: TraceStatus = TraceStatus.OK,
    producer: str | None = "claude_code",
    duration_ms: int | None = 1000,
) -> NormalizedTrace:
    return NormalizedTrace(
        trace_id=trace_id,
        status=status,
        producer=producer,
        model=model,
        token_usage=TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
            _total_tokens=total,
        ),
        tool_calls=tool_calls or [],
        execution_duration_ms=duration_ms,
    )


class TestPricing:
    def test_known_model_priced(self) -> None:
        cost = compute_cost(
            TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000),
            "claude-opus-4-8",
            DEFAULT_PRICEBOOK,
        )
        assert cost.priced is True
        assert cost.input_usd == 5.0
        assert cost.output_usd == 25.0
        assert cost.total_usd == 30.0
        assert cost.flags == []

    def test_cache_priced_with_documented_multipliers(self) -> None:
        cost = compute_cost(
            TokenUsage(cache_creation_input_tokens=1_000_000, cache_read_input_tokens=1_000_000),
            "claude-opus-4-8",
            DEFAULT_PRICEBOOK,
        )
        assert cost.cache_write_usd == 6.25  # 1.25x input ($5)
        assert cost.cache_read_usd == 0.5  # 0.1x input ($5)
        # cache present -> TTL-assumption flag is raised, never silently assumed
        assert any("TTL" in f for f in cost.flags)

    def test_unknown_model_is_flagged_not_fabricated(self) -> None:
        cost = compute_cost(TokenUsage(input_tokens=1000), "gpt-4o", DEFAULT_PRICEBOOK)
        assert cost.priced is False
        assert cost.total_usd == 0.0
        assert any("not in the price book" in f for f in cost.flags)

    def test_missing_model_is_flagged(self) -> None:
        cost = compute_cost(TokenUsage(input_tokens=1000), None, DEFAULT_PRICEBOOK)
        assert cost.priced is False
        assert any("no recorded model" in f for f in cost.flags)

    def test_model_aliasing_is_explicit_and_sourced(self) -> None:
        # exact id resolves
        assert lookup_price("claude-opus-4-8", DEFAULT_PRICEBOOK) is not None
        # provider-routing prefix (Bedrock) is the same SKU -> resolves
        assert lookup_price("Anthropic.claude-opus-4-8", DEFAULT_PRICEBOOK) is not None
        # a dated snapshot resolves via the explicit, sourced alias table
        assert lookup_price("claude-haiku-4-5-20251001", DEFAULT_PRICEBOOK) is not None
        # an unknown model is not priced
        assert lookup_price("not-a-model", DEFAULT_PRICEBOOK) is None

    def test_fast_speed_tier_is_not_silently_priced_as_base(self) -> None:
        # fast mode is a different (premium) SKU; without a cited fast price it
        # must be left UNPRICED, never mapped onto the base claude-opus-4-8 rate.
        assert lookup_price("claude-opus-4-8-fast", DEFAULT_PRICEBOOK) is None
        cost = compute_cost(
            TokenUsage(input_tokens=1_000_000), "claude-opus-4-8-fast", DEFAULT_PRICEBOOK
        )
        assert cost.priced is False
        assert cost.total_usd == 0.0
        assert any("not in the price book" in f for f in cost.flags)

    def test_custom_pricebook_overrides_default(self) -> None:
        cost = compute_cost(TokenUsage(input_tokens=1_000_000), "claude-opus-4-8", {})
        assert cost.priced is False  # empty book -> unpriced, not the default $5


class TestRedundancy:
    def test_no_tools(self) -> None:
        r = compute_redundancy([])
        assert r.total_tool_calls == 0
        assert r.redundancy_rate == 0.0
        assert r.repeated_calls == []

    def test_strict_rate_only_counts_byte_identical(self) -> None:
        # same path, different offset -> distinct exact signatures, but same path identity
        calls = [
            ToolCall(id="1", name="Read", arguments={"file_path": "/a", "offset": 0}),
            ToolCall(id="2", name="Read", arguments={"file_path": "/a", "offset": 50}),
            ToolCall(id="3", name="Read", arguments={"file_path": "/a", "offset": 99}),
        ]
        r = compute_redundancy(calls)
        assert r.total_tool_calls == 3
        assert r.distinct_tool_calls == 3  # all exact-distinct
        assert r.redundancy_rate == 0.0
        # ...but the path-identity diagnostic catches the re-read
        path_repeats = [rc for rc in r.repeated_calls if rc.signature_kind == "path"]
        assert len(path_repeats) == 1
        assert path_repeats[0].count == 3
        assert path_repeats[0].identity == "/a"
        assert path_repeats[0].tool == "Read"

    def test_exact_duplicates_drive_rate(self) -> None:
        calls = [
            ToolCall(id="1", name="Bash", arguments={"command": "ls"}),
            ToolCall(id="2", name="Bash", arguments={"command": "ls"}),
        ]
        r = compute_redundancy(calls)
        assert r.distinct_tool_calls == 1
        assert r.redundant_tool_calls == 1
        assert r.redundancy_rate == 0.5

    def test_shell_prologue_identity(self) -> None:
        # different full commands sharing the same cd prologue -> shell identity match
        calls = [
            ToolCall(id="1", name="Bash", arguments={"command": "cd /repo\nnpm test"}),
            ToolCall(id="2", name="Bash", arguments={"command": "cd /repo\nnpm run build"}),
            ToolCall(id="3", name="Bash", arguments={"command": "cd /repo\nnpm run lint"}),
        ]
        r = compute_redundancy(calls)
        assert r.redundancy_rate == 0.0  # full commands differ -> no exact dup
        shell = [rc for rc in r.repeated_calls if rc.signature_kind == "shell"]
        assert len(shell) == 1
        assert shell[0].count == 3
        assert shell[0].identity == "cd /repo"

    def test_normalize_command_collapses_uuid_and_takes_first_line(self) -> None:
        cmd = "cd /tmp/abcdef01-2345-6789-abcd-ef0123456789/work\nnpm test"
        assert normalize_command(cmd) == "cd /tmp/<id>/work"
        assert normalize_command("") == ""

    def test_exact_signature_is_order_independent(self) -> None:
        a = ToolCall(id="1", name="X", arguments={"a": 1, "b": 2})
        b = ToolCall(id="2", name="X", arguments={"b": 2, "a": 1})
        assert exact_signature(a) == exact_signature(b)


class TestTraceMetrics:
    def test_basic_trace(self) -> None:
        m = compute_trace_metrics(
            _trace(
                input_tokens=1000,
                output_tokens=200,
                total=1200,
                tool_calls=[
                    ToolCall(id="1", name="Read", arguments={"file_path": "/a"}),
                    ToolCall(id="2", name="Bash", arguments={"command": "ls"}),
                ],
            )
        )
        assert m.tokens.total_tokens == 1200
        assert m.cost.priced is True
        assert m.total_tool_calls == 2
        assert m.tool_counts == {"Read": 1, "Bash": 1}
        assert m.status == "OK"
        assert m.redundancy.redundancy_rate == 0.0


class TestComputeL0:
    def _corpus(self) -> list[NormalizedTrace]:
        return [
            _trace(
                trace_id="big",
                model="claude-opus-4-8",
                input_tokens=900_000,
                output_tokens=40_000,
                total=940_000,
            ),
            _trace(
                trace_id="small",
                model="claude-sonnet-4-6",
                input_tokens=10_000,
                output_tokens=2_000,
                total=12_000,
            ),
            _trace(
                trace_id="nomodel",
                model=None,
                input_tokens=5_000,
                output_tokens=1_000,
                total=6_000,
                producer=None,
            ),
        ]

    def test_aggregate_and_breakdowns(self) -> None:
        rep = compute_l0(self._corpus(), experiment_id="exp1", generated_at="2026-06-29T00:00:00Z")
        assert rep.schema_version == SCHEMA_VERSION
        assert rep.n_traces == 3
        assert rep.experiment_id == "exp1"
        assert rep.aggregate.tokens.total_tokens == 940_000 + 12_000 + 6_000
        assert rep.aggregate.status_counts == {"OK": 3}
        # traces sorted by tokens desc
        assert [m.trace_id for m in rep.traces] == ["big", "small", "nomodel"]
        # cost: 2 priced, 1 unpriced (model None)
        assert rep.aggregate.cost.priced_traces == 2
        assert rep.aggregate.cost.unpriced_traces == 1
        assert any("unpriced" in f for f in rep.aggregate.cost.flags)
        # by_model breakdown has the unknown-model bucket
        keys = {g.key for g in rep.by_model}
        assert "claude-opus-4-8" in keys and "claude-sonnet-4-6" in keys
        assert any("unknown-model" in k for k in keys)
        # pricing flag names the unpriced model and the source
        assert any("Unpriced models" in f for f in rep.pricing_flags)
        assert any("claude-api skill" in f for f in rep.pricing_flags)

    def test_token_stats(self) -> None:
        rep = compute_l0(self._corpus(), generated_at="x")
        assert rep.aggregate.token_stats.max == 940_000
        assert rep.aggregate.token_stats.min == 6_000
        assert rep.aggregate.token_stats.count == 3

    def test_contract_round_trips_through_json(self) -> None:
        rep = compute_l0(self._corpus(), experiment_id="exp1", generated_at="x")
        js = rep.model_dump_json()
        back = L0MetricsReport.model_validate_json(js)
        assert back.n_traces == rep.n_traces
        assert back.aggregate.tokens.total_tokens == rep.aggregate.tokens.total_tokens
        assert back.traces[0].trace_id == "big"

    def test_empty_corpus(self) -> None:
        rep = compute_l0([], generated_at="x")
        assert rep.n_traces == 0
        assert rep.aggregate.token_stats.count == 0
        assert rep.aggregate.cost.total_usd == 0.0


class TestViaRealNormalization:
    """Compute L0 off the real MLflow normalization path (synthetic fixture)."""

    def test_from_normalized_fixture(self, synthetic_trace: Any) -> None:
        from ail.ingest.mlflow_source import normalize_trace

        nt = normalize_trace(synthetic_trace)
        rep = compute_l0([nt], experiment_id="660599403165942", generated_at="x")
        assert rep.n_traces == 1
        m = rep.traces[0]
        assert m.model == "claude-opus-4-8"
        assert m.tokens.total_tokens == 1200
        assert m.cost.priced is True
        assert m.cost.total_usd > 0
        assert m.tool_counts == {"Read": 1, "Bash": 1}
