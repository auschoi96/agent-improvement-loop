"""Unit tests for the unified per-agent-version publish (Tier A, Phase B).

These exercise the pure artifact -> records builders and the SQL row/DDL mapping
against the **committed** Phase-2 seed artifact, plus the readiness-gated display
status both ways (collecting vs proven). No network/warehouse access — the write
path is exercised through the same fake-client pattern as ``test_publish.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from databricks.sdk.service.sql import StatementState

import ail.publish_versions as pv
from ail.compare.contract import Recommendation
from ail.optimize.phase2 import L1Outcome, Phase2Artifact
from ail.publish_versions import (
    REGISTRY_COLUMNS,
    VERSION_COMPARISON_COLUMNS,
    VERSION_L0_COLUMNS,
    VERSION_READINESS_COLUMNS,
    VersionComparisonStatus,
    _comparison_rows,
    _metric_delta,
    _readiness_row,
    _reconstruct_redundant,
    _registry_row,
    _version_l0_row,
    build_phase2_version_bundle,
    publish_registry,
    publish_version_bundle,
)
from ail.readiness import ReadinessThresholds, ReadinessTier
from ail.registry import DEFAULT_REGISTRY

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = REPO_ROOT / "artifacts" / "phase2_token_lever.json"

AGENT = "claude_code"
BASE_V = "v0-baseline-no-skill"
CAND_V = "v1-token-efficiency-skill"


def _load_artifact() -> Phase2Artifact:
    return Phase2Artifact.model_validate_json(ARTIFACT_PATH.read_text(encoding="utf-8"))


def _bundle(**kwargs):  # type: ignore[no-untyped-def]
    return build_phase2_version_bundle(
        _load_artifact(),
        agent_name=AGENT,
        baseline_version=BASE_V,
        candidate_version=CAND_V,
        experiment_id="660599403165942",
        **kwargs,
    )


# -- fake warehouse client (records statements) ----------------------------


class _FakeStatus:
    def __init__(self, state: StatementState) -> None:
        self.state = state
        self.error = None


class _FakeResp:
    def __init__(self, state: StatementState) -> None:
        self.statement_id = "stmt-1"
        self.status = _FakeStatus(state)


class _FakeStatementExecution:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def execute_statement(self, *, warehouse_id, statement, wait_timeout=None):  # type: ignore[no-untyped-def]
        self.statements.append(statement)
        return _FakeResp(StatementState.SUCCEEDED)

    def get_statement(self, statement_id):  # type: ignore[no-untyped-def]
        return _FakeResp(StatementState.SUCCEEDED)


class _FakeClient:
    def __init__(self) -> None:
        self.statement_execution = _FakeStatementExecution()


# -- aggregation reproduces the real 35.4% headline ------------------------


def test_aggregate_reproduces_realized_headline() -> None:
    artifact = _load_artifact()
    bundle = _bundle()
    base, cand = bundle.aggregates
    assert base.agent_version == BASE_V and cand.agent_version == CAND_V

    # PROMOTE-only aggregate matches the artifact's realized headline.
    assert base.total_tokens == artifact.realized_baseline_tokens == 2570.0
    assert cand.total_tokens == artifact.realized_candidate_tokens == 1660.0
    assert base.n_traces == cand.n_traces == artifact.n_promote == 2
    assert base.n_traces_total == cand.n_traces_total == artifact.n_tasks == 5

    # Extensive sums over the 2 PROMOTE tasks (ts-fix-01, ts-impl-02).
    assert base.total_tool_calls == 64.0  # 35 + 29
    assert cand.total_tool_calls == 33.0  # 16 + 17
    assert base.tokens_per_trace == 1285.0
    assert cand.tokens_per_trace == 830.0
    # cost is unpriced in the seed -> $0 floor, flagged honestly (not asserted true).
    assert base.total_cost_usd == 0.0 and base.cost_priced is False


def test_redundancy_reconstructed_from_rate_times_total() -> None:
    # redundancy_rate * total_tool_calls recovers the integer count exactly.
    assert _reconstruct_redundant(0.103448, 29) == 3
    assert _reconstruct_redundant(0.0, 35) == 0
    base, cand = _bundle().aggregates
    assert base.redundant_tool_calls == 3.0
    assert base.redundancy_rate == pytest.approx(3 / 64)
    assert cand.redundant_tool_calls == 0.0
    assert cand.redundancy_rate == 0.0


def test_headline_delta_is_the_real_minus_35_4_pct() -> None:
    cmp = _bundle().comparison
    assert cmp.headline_metric == "total_tokens"
    assert cmp.headline_baseline == 2570.0
    assert cmp.headline_candidate == 1660.0
    assert cmp.headline_delta_pct == pytest.approx(-35.4086, abs=1e-3)
    assert cmp.headline_improved is True


def test_comparison_emits_the_five_view_metrics() -> None:
    cmp = _bundle().comparison
    metrics = [d.metric for d in cmp.deltas]
    assert metrics == [
        "total_tokens",
        "tokens_per_trace",
        "total_tool_calls",
        "redundancy_rate",
        "total_usd",
    ]
    tool = next(d for d in cmp.deltas if d.metric == "total_tool_calls")
    assert tool.baseline == 64.0 and tool.candidate == 33.0 and tool.improved is True


# -- readiness gating: never green unless the wall clears ------------------


def test_seed_is_controlled_proof_collecting_not_green() -> None:
    cmp = _bundle().comparison
    # 5 organic traces < baseline floor (10) => readiness is COLLECTING...
    assert cmp.readiness.tier is ReadinessTier.COLLECTING
    assert cmp.readiness.can_prove_improvement is False
    # ...so even though the controlled proof holds and correctness was held, the
    # status is NOT the green PROVEN — it is the honest amber dual-signal state.
    assert cmp.correctness_held is True
    assert cmp.frozen_suite_present is True
    assert cmp.status is VersionComparisonStatus.CONTROLLED_PROOF_COLLECTING
    assert cmp.status is not VersionComparisonStatus.PROVEN


def test_status_is_proven_only_when_readiness_clears() -> None:
    # Lower the trace floors so the (real) 5 traces clear the prove gate.
    cmp = _bundle(
        thresholds=ReadinessThresholds(baseline_min_traces=1, prove_min_traces=1)
    ).comparison
    assert cmp.readiness.tier is ReadinessTier.READY_TO_PROVE
    assert cmp.readiness.can_prove_improvement is True
    assert cmp.status is VersionComparisonStatus.PROVEN


def test_regression_never_reads_as_a_win() -> None:
    artifact = _load_artifact()
    artifact.outcomes[0].l1_outcome = L1Outcome.REGRESSED
    bundle = build_phase2_version_bundle(
        artifact,
        agent_name=AGENT,
        baseline_version=BASE_V,
        candidate_version=CAND_V,
        thresholds=ReadinessThresholds(baseline_min_traces=1, prove_min_traces=1),
    )
    # Even with the readiness wall cleared, a regression blocks the green status.
    assert bundle.comparison.correctness_held is False
    assert bundle.comparison.status is VersionComparisonStatus.REGRESSED


def test_no_promote_is_collecting() -> None:
    artifact = _load_artifact()
    for o in artifact.outcomes:
        o.recommendation = Recommendation.BLOCK
    artifact.n_promote = 0
    bundle = build_phase2_version_bundle(
        artifact, agent_name=AGENT, baseline_version=BASE_V, candidate_version=CAND_V
    )
    # No PROMOTE task => empty counted set => no headline improvement => collecting.
    assert bundle.comparison.status is VersionComparisonStatus.COLLECTING


def _set_promote_total_tokens_candidate(artifact: Phase2Artifact, fn) -> None:  # type: ignore[no-untyped-def]
    """Rewrite each PROMOTE task's candidate total_tokens via ``fn(baseline)``.

    Lets a test drive the version-level objective headline (the extensive sum over
    the counted PROMOTE set) to an arbitrary direction without inventing a whole
    artifact.
    """
    for o in artifact.outcomes:
        if o.recommendation is Recommendation.PROMOTE and o.comparison is not None:
            d = o.comparison.delta_for("total_tokens")
            assert d is not None
            d.candidate = fn(d.baseline)


def test_objective_regression_reads_as_regressed_not_collecting() -> None:
    # Candidate is strictly WORSE on the objective (tokens INCREASE) on the counted
    # set. That is a measured FAILURE — it must surface as REGRESSED (negative),
    # never neutral/COLLECTING ('not enough data yet'). Pre-fix this fell through
    # to COLLECTING; this test fails against that and passes after the fix.
    artifact = _load_artifact()
    _set_promote_total_tokens_candidate(artifact, lambda baseline: baseline + 100.0)
    # Even with the readiness wall cleared, an objective regression is not a win.
    bundle = build_phase2_version_bundle(
        artifact,
        agent_name=AGENT,
        baseline_version=BASE_V,
        candidate_version=CAND_V,
        thresholds=ReadinessThresholds(baseline_min_traces=1, prove_min_traces=1),
    )
    cmp = bundle.comparison
    assert cmp.headline_improved is False
    assert cmp.status is VersionComparisonStatus.REGRESSED
    assert cmp.status is not VersionComparisonStatus.COLLECTING


def test_objective_tie_stays_collecting() -> None:
    # A genuine tie (objective unchanged) with improvement unprovable is honestly
    # COLLECTING — no regression, just no win yet. Guards against over-correcting
    # the regression fix into flagging no-change as a regression.
    artifact = _load_artifact()
    _set_promote_total_tokens_candidate(artifact, lambda baseline: baseline)
    bundle = build_phase2_version_bundle(
        artifact, agent_name=AGENT, baseline_version=BASE_V, candidate_version=CAND_V
    )
    cmp = bundle.comparison
    assert cmp.headline_improved is False
    assert cmp.status is VersionComparisonStatus.COLLECTING


# -- _metric_delta semantics -----------------------------------------------


def test_metric_delta_pct_none_when_baseline_zero() -> None:
    d = _metric_delta("total_usd", "usd", 0.0, 0.0)
    assert d.delta_pct is None
    assert d.improved is False  # a tie is not an improvement


def test_metric_delta_strict_improvement() -> None:
    d = _metric_delta("total_tokens", "tokens", 100.0, 60.0)
    assert d.delta_absolute == -40.0
    assert d.delta_pct == -40.0
    assert d.improved is True


# -- flat rows / DDL match declared column order ---------------------------


def test_row_builders_match_column_orders() -> None:
    bundle = _bundle()
    assert len(_registry_row(DEFAULT_REGISTRY.get("claude_code"), generated_at="t")) == len(
        REGISTRY_COLUMNS
    )
    assert len(_version_l0_row(bundle.aggregates[0])) == len(VERSION_L0_COLUMNS)
    for row in _comparison_rows(bundle.comparison):
        assert len(row) == len(VERSION_COMPARISON_COLUMNS)
    assert len(_readiness_row(bundle.comparison)) == len(VERSION_READINESS_COLUMNS)


def test_ddl_creates_the_four_unified_tables() -> None:
    ddl = "\n".join(pv._ddl("cat", "sch"))
    for table in (
        "agent_registry",
        "agent_version_l0",
        "agent_version_comparison",
        "agent_version_readiness",
    ):
        assert f".{table} (" in ddl


# -- write path: idempotent, composite-key REPLACE WHERE -------------------


def test_publish_version_bundle_uses_composite_replace_predicates() -> None:
    client = _FakeClient()
    publish_version_bundle(_bundle(), client=client, warehouse_id="wh", catalog="cat", schema="sch")
    stmts = client.statement_execution.statements
    swaps = [s for s in stmts if "REPLACE WHERE" in s]
    # one per version (l0) + comparison + readiness = 4 swaps
    assert len(swaps) == 4
    assert any(
        "agent_name = 'claude_code' AND agent_version = 'v0-baseline-no-skill'" in s for s in swaps
    )
    assert any(
        "agent_name = 'claude_code' AND agent_version = 'v1-token-efficiency-skill'" in s
        for s in swaps
    )
    assert any(
        "baseline_version = 'v0-baseline-no-skill' AND candidate_version = "
        "'v1-token-efficiency-skill'" in s
        for s in swaps
    )


def test_publish_registry_writes_one_slice_per_agent() -> None:
    client = _FakeClient()
    n = publish_registry(
        DEFAULT_REGISTRY, client=client, warehouse_id="wh", catalog="cat", schema="sch"
    )
    assert n == 1
    swaps = [s for s in client.statement_execution.statements if "REPLACE WHERE" in s]
    assert len(swaps) == 1
    assert "agent_name = 'claude_code'" in swaps[0]
