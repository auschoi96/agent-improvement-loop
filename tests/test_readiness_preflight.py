"""Tests for the ``ail-readiness`` onboarding preflight (:mod:`ail.jobs.readiness_preflight`).

All offline. The CLI's only live dependency — gathering facts from MLflow — is
injected away: the gate/render tests drive :func:`evaluate` with a fake facts
source (a ``Callable`` returning canned :class:`~ail.readiness.ReadinessFacts`),
and the live :func:`gather_facts` path is exercised with a fake trace *source*
(an object exposing ``fetch_cohort_traces``) plus fake assessment objects, so no
test touches ``mlflow.search_traces``. This mirrors the injectable-client seam in
``ail.judges.registration`` / ``ail.jobs.publish_job``.

The headline invariant under test is the task's contract: the CLI maps facts to
the readiness module's gates faithfully (0 traces => all collecting; 10–29 =>
baseline ready, prove not; >=50 + frozen suite => prove ready) and **fails
closed** — an access error exits non-zero and prints no fabricated "ready".
"""

from __future__ import annotations

import pytest

from ail.jobs import readiness_preflight as rp
from ail.jobs.readiness_preflight import (
    PreflightAccessError,
    PreflightResult,
    evaluate,
    gather_facts,
    main,
    render,
)
from ail.judges import ScorePair, compute_agreement
from ail.readiness import (
    GateName,
    JudgeFact,
    ReadinessFacts,
    ReadinessTier,
)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


def _facts_source(facts: ReadinessFacts):
    """A fake facts source: ignores the experiment/cohort, returns canned facts."""

    def _source(experiment_id: str, cohort: object) -> ReadinessFacts:
        return facts

    return _source


def _trusted_judge(name: str = "token_efficiency", *, n_scored: int) -> JudgeFact:
    """A judge measured at perfect agreement (well above the floor)."""
    pairs = [ScorePair(item_id=str(i), human_value="yes", judge_value="yes") for i in range(10)]
    report = compute_agreement(pairs, judge_name=name)
    return JudgeFact.from_agreement_report(report, n_scored_traces=n_scored)


def _healthy_facts(*, trace_count: int) -> ReadinessFacts:
    """Facts where every gate — universal and quality — passes."""
    return ReadinessFacts(
        trace_count=trace_count,
        label_count=30,
        frozen_suite_present=True,
        n_scored_traces=trace_count,
        judge_runs=trace_count,
        judge_run_successes=trace_count,
        judges=(_trusted_judge(n_scored=trace_count),),
    )


def _gates(result: PreflightResult) -> dict[GateName, bool]:
    return {g.name: g.passed for g in result.gates}


# -- fakes for the live gather_facts path ----------------------------------


class _FakeAssessmentSource:
    def __init__(self, source_type: str) -> None:
        self.source_type = source_type


class _FakeAssessment:
    def __init__(self, name: str, source_type: str) -> None:
        self.name = name
        self.source = _FakeAssessmentSource(source_type)


class _FakeInfo:
    def __init__(self, assessments: list[_FakeAssessment]) -> None:
        self.assessments = assessments


class _FakeRaw:
    def __init__(self, assessments: list[_FakeAssessment]) -> None:
        self.info = _FakeInfo(assessments)


class _FakeTrace:
    def __init__(self, assessments: list[_FakeAssessment]) -> None:
        self.raw = _FakeRaw(assessments)


class _FakeTraceSource:
    """Stands in for :class:`MLflowTraceSource` — returns canned traces."""

    def __init__(self, traces: list[_FakeTrace]) -> None:
        self._traces = traces
        self.calls: list[str] = []

    def fetch_cohort_traces(self, cohort: object, *, experiment_id: str, **_: object):
        self.calls.append(experiment_id)
        return self._traces


class _ExplodingSource:
    """A trace source whose read fails — the auth/permission case."""

    def fetch_cohort_traces(self, *_: object, **__: object):
        raise PermissionError("UC trace store: CAN_USE required on the SQL warehouse")


# ---------------------------------------------------------------------------
# The four headline gate-mapping scenarios (task spec)
# ---------------------------------------------------------------------------


class TestGateMapping:
    def test_zero_traces_all_not_ready(self) -> None:
        result = evaluate("exp", facts_source=_facts_source(ReadinessFacts(trace_count=0)))
        gates = _gates(result)
        # Every gate is not-ready and both lenses are collecting.
        assert not any(gates.values())
        assert result.status.tier is ReadinessTier.COLLECTING
        assert result.quality_status.tier is ReadinessTier.COLLECTING
        assert result.status.can_prove_improvement is False

    @pytest.mark.parametrize("trace_count", [10, 20, 29])
    def test_baseline_ready_prove_not(self, trace_count: int) -> None:
        """10–29 traces: baseline/RLM ready, but proving an improvement is not."""
        result = evaluate(
            "exp", facts_source=_facts_source(ReadinessFacts(trace_count=trace_count))
        )
        gates = _gates(result)
        assert gates[GateName.TRACE_BASELINE] is True
        assert gates[GateName.TRACE_PROVE] is False
        # The headline (deterministic) lens can baseline but cannot yet prove.
        assert result.status.tier is ReadinessTier.BASELINE_ONLY
        assert result.status.can_prove_improvement is False

    def test_prove_ready_when_traces_and_wall_cleared(self) -> None:
        """>=50 traces + a frozen suite: the deterministic prove path is READY."""
        facts = ReadinessFacts(trace_count=50, frozen_suite_present=True)
        result = evaluate("exp", facts_source=_facts_source(facts))
        gates = _gates(result)
        assert gates[GateName.TRACE_PROVE] is True
        assert gates[GateName.FROZEN_SUITE] is True
        # No labels/judge => the headline (deterministic token-win) lens is ready
        # to prove, even though the MemAlign-judge lens honestly is not.
        assert result.status.tier is ReadinessTier.READY_TO_PROVE
        assert result.status.can_prove_improvement is True
        assert gates[GateName.HUMAN_LABELS] is False
        assert result.quality_status.tier is not ReadinessTier.READY_TO_PROVE

    def test_prove_not_ready_without_frozen_suite(self) -> None:
        """>=50 traces but no frozen suite: cannot prove (the wall is uncleared)."""
        facts = ReadinessFacts(trace_count=60, frozen_suite_present=False)
        result = evaluate("exp", facts_source=_facts_source(facts))
        gates = _gates(result)
        assert gates[GateName.TRACE_PROVE] is True
        assert gates[GateName.FROZEN_SUITE] is False
        assert result.status.can_prove_improvement is False

    def test_fully_healthy_is_ready_to_prove_both_lenses(self) -> None:
        result = evaluate("exp", facts_source=_facts_source(_healthy_facts(trace_count=60)))
        gates = _gates(result)
        assert all(gates.values())
        assert result.status.tier is ReadinessTier.READY_TO_PROVE
        assert result.quality_status.tier is ReadinessTier.READY_TO_PROVE


# ---------------------------------------------------------------------------
# Rendering + summary
# ---------------------------------------------------------------------------


class TestRender:
    def test_collecting_render_has_no_ready_marker_and_honest_messages(self) -> None:
        out = render(evaluate("exp", facts_source=_facts_source(ReadinessFacts(trace_count=0))))
        assert "NOT READY" in out
        assert "\n  Unlocked now:" in out
        # The readiness module's own "need N more" messages are surfaced verbatim.
        assert "need 10 more trace(s) to baseline" in out
        assert "need 20 more human label(s)" in out

    def test_summary_is_gate_driven(self) -> None:
        result = evaluate("exp", facts_source=_facts_source(ReadinessFacts(trace_count=20)))
        summary = rp._summary(result)
        assert "RLM+diagnosis: READY" in summary
        assert "MemAlign judge: need 20 more labels" in summary
        assert "prove a total_tokens win: need 30 more traces" in summary

    def test_healthy_summary_all_ready(self) -> None:
        out = render(evaluate("exp", facts_source=_facts_source(_healthy_facts(trace_count=60))))
        assert "Unlocked now: RLM+diagnosis: READY; MemAlign judge: READY" in out
        assert "prove a total_tokens win: READY" in out


# ---------------------------------------------------------------------------
# The live gather_facts path (offline, with injected fakes)
# ---------------------------------------------------------------------------


class TestGatherFacts:
    def test_counts_labels_scored_and_judges(self) -> None:
        traces = [
            # human label + a judge verdict -> 1 label, scored
            _FakeTrace(
                [
                    _FakeAssessment("token_efficiency", "HUMAN"),
                    _FakeAssessment("token_efficiency", "LLM_JUDGE"),
                ]
            ),
            # judge verdict only -> scored, no label
            _FakeTrace([_FakeAssessment("correctness", "LLM_JUDGE")]),
            # human label only -> 1 label, not scored
            _FakeTrace([_FakeAssessment("token_efficiency", "HUMAN")]),
            # bare trace -> nothing
            _FakeTrace([]),
        ]
        source = _FakeTraceSource(traces)
        cohort = rp.build_cohort(None, "exp")

        facts = gather_facts("exp", cohort, source=source, suite_present=True)

        assert facts.trace_count == 4
        assert facts.label_count == 2
        assert facts.n_scored_traces == 2
        assert facts.frozen_suite_present is True
        judges = {j.judge_name: j for j in facts.judges}
        assert set(judges) == {"token_efficiency", "correctness"}
        assert judges["token_efficiency"].n_scored_traces == 1
        # A preflight never measures agreement -> every discovered judge is distrusted.
        assert all(j.is_distrusted for j in facts.judges)
        assert source.calls == ["exp"]

    def test_access_failure_raises_actionable_error(self) -> None:
        cohort = rp.build_cohort(None, "exp")
        with pytest.raises(PreflightAccessError) as excinfo:
            gather_facts(
                "exp", cohort, source=_ExplodingSource(), profile="myprof", warehouse_id="wh-1"
            )
        msg = str(excinfo.value)
        assert "CAN_USE" in msg
        assert "myprof" in msg
        assert "wh-1" in msg


# ---------------------------------------------------------------------------
# main(): exit codes, fail-closed
# ---------------------------------------------------------------------------


class TestMain:
    def test_access_error_exits_nonzero_without_fake_ready(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def _boom(*_: object, **__: object) -> ReadinessFacts:
            raise PreflightAccessError("UC trace store needs CAN_USE on warehouse wh-9")

        monkeypatch.setattr(rp, "gather_facts", _boom)

        code = main(["exp-123", "--profile", "p", "--warehouse-id", "wh-9"])

        assert code == 1
        captured = capsys.readouterr()
        # The verdict table is never printed on error: no fabricated readiness.
        assert "READY" not in captured.out
        assert captured.out.strip() == ""
        assert "CAN_USE" in captured.err

    def test_happy_path_exits_zero_and_prints_table(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(rp, "gather_facts", lambda *a, **k: _healthy_facts(trace_count=60))

        code = main(["exp-123"])

        assert code == 0
        out = capsys.readouterr().out
        assert "Readiness preflight" in out
        assert "Unlocked now:" in out
