"""Tests for the GEPA agent-optimization loop (:mod:`ail.optimize.gepa_runner`).

Fully offline — **no live agent, network, or model**. The agent is a scripted
``SuiteAdapter`` (baseline vs seed-candidate vs evolved-candidate told apart by the
injected skill marker / an ``EVOLVED`` token in the system prompt), and
:func:`gepa.optimize` is replaced by a ``fake_optimize`` that exercises the real
fitness + reflection path against the train set and returns a scripted best
candidate. Nothing here imports gepa.

The load-bearing anti-overfit wall is proven two ways: the fitness function
(:meth:`FrozenSuiteGepaAdapter.evaluate`) **raises** on a held-out task id, and an
end-to-end run only ever evaluates train ids (the held-out split is scored solely
by the live harness afterwards). The loop is also shown to return a CANDIDATE — it
never auto-applies or promotes — and to emit a well-formed, round-trippable result.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from ail.compare import Recommendation
from ail.ingest.base import (
    AgentAdapter,
    AgentRunResult,
    AgentTask,
    NormalizedTrace,
    TokenUsage,
    TraceStatus,
)
from ail.optimize import (
    BASELINE,
    CANDIDATE,
    FrozenSuiteGepaAdapter,
    GepaConfig,
    GepaOptimizationResult,
    HeldOutLeakError,
    Phase2Artifact,
    VerifySpec,
    candidate_lever_config,
    fitness_from_outcome,
    run_gepa_optimization,
    split_suite,
    token_efficiency_skill,
)
from ail.optimize.phase2 import L1Outcome, TaskOutcome
from ail.task_suite.schema import Difficulty, Task, TaskCategory, TaskSuite

_STAMP = "2026-06-30T00:00:00+00:00"
_COMPONENT = "skill_body"
_EVOLVED_BODY = "EVOLVED: stay terse.\n\nNever re-read a file already read in-session."


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class SuiteAdapter(AgentAdapter):
    """Scripted agent: baseline / seed-candidate / evolved-candidate keyed on the prompt.

    Three arms are distinguished by the candidate's system prompt: no ``<skill``
    marker -> baseline; ``<skill`` present without ``EVOLVED`` -> the seed body;
    ``EVOLVED`` present -> the evolved body. Each ``plan`` maps a task prompt to the
    token count (and success flag) for each arm, so a test can make the evolved body
    strictly beat the seed body.
    """

    name = "scripted-suite"

    def __init__(self, plans: dict[str, dict[str, Any]]) -> None:
        self.plans = plans
        self.seen: list[AgentTask] = []

    def run(self, task: AgentTask) -> AgentRunResult:
        self.seen.append(task)
        plan = self.plans[task.prompt]
        sp = task.system_prompt or ""
        if "<skill" not in sp:
            arm, tokens, ok = "base", plan["baseline"], plan.get("baseline_ok", True)
        elif "EVOLVED" in sp:
            arm, tokens, ok = "evolved", plan["evolved"], plan.get("evolved_ok", True)
        else:
            arm, tokens, ok = "seed", plan["seed"], plan.get("seed_ok", True)
        trace = NormalizedTrace(
            trace_id=f"{arm}-{task.prompt}",
            status=TraceStatus.OK if ok else TraceStatus.ERROR,
            producer=self.name,
            model="claude-opus-4-8",
            token_usage=TokenUsage(input_tokens=tokens),
        )
        return AgentRunResult(
            trace=trace,
            output_text="done",
            success=ok,
            error=None if ok else f"{arm} crashed",
        )


def _task(task_id: str, prompt: str) -> Task:
    return Task(
        task_id=task_id,
        prompt=prompt,
        category=TaskCategory.REPEATED_TARGET_BOILERPLATE,
        source_trace_id=f"src-{task_id}",
        difficulty=Difficulty.MEDIUM,
    )


def _suite(*tasks: Task) -> TaskSuite:
    return TaskSuite(version="gepa-test-v1", tasks=tuple(tasks)).freeze()


def _five_task_suite() -> TaskSuite:
    return _suite(*[_task(f"ts-0{i}", f"p{i}") for i in range(1, 6)])


def _uniform_plans(
    *prompts: str, baseline: int, seed: int, evolved: int
) -> dict[str, dict[str, Any]]:
    return {p: {"baseline": baseline, "seed": seed, "evolved": evolved} for p in prompts}


def _adapter5(*, baseline: int = 100, seed: int = 70, evolved: int = 50) -> SuiteAdapter:
    """A :class:`SuiteAdapter` over the five ``p1..p5`` prompts with uniform arm tokens."""
    prompts = [f"p{i}" for i in range(1, 6)]
    return SuiteAdapter(_uniform_plans(*prompts, baseline=baseline, seed=seed, evolved=evolved))


def _all_pass_specs(*task_ids: str) -> dict[str, VerifySpec]:
    """An L1 verify spec that always passes (``true``) for each task id."""
    return {tid: VerifySpec(name=f"ok-{tid}", command=["true"]) for tid in task_ids}


def make_fake_optimize(
    captured: dict[str, Any],
    *,
    evolved_body: str | None = _EVOLVED_BODY,
    best_idx: int = 1,
    val_scores: list[float] | None = None,
) -> Any:
    """A stand-in for :func:`gepa.optimize`: exercises the real fitness, returns a result.

    Records the arguments GEPA was handed (so a test can prove only train tasks
    were passed), drives one real :meth:`FrozenSuiteGepaAdapter.evaluate` +
    :meth:`make_reflective_dataset` pass over the train set, and returns a scripted
    result whose ``best_candidate`` carries ``evolved_body``.
    """

    def fake_optimize(
        *,
        seed_candidate: dict[str, str],
        trainset: list[Task],
        valset: list[Task],
        adapter: FrozenSuiteGepaAdapter,
        reflection_lm: str,
        max_metric_calls: int,
        **kwargs: Any,
    ) -> Any:
        captured["seed_candidate"] = dict(seed_candidate)
        captured["trainset_ids"] = [t.task_id for t in trainset]
        captured["valset_ids"] = [t.task_id for t in valset]
        captured["adapter"] = adapter
        captured["reflection_lm"] = reflection_lm
        captured["max_metric_calls"] = max_metric_calls
        captured["kwargs"] = kwargs
        # Exercise the real fitness + reflection path on the TRAIN set only.
        eval_batch = adapter.evaluate(trainset, seed_candidate, capture_traces=True)
        captured["eval_scores"] = list(eval_batch.scores)
        captured["reflective"] = adapter.make_reflective_dataset(
            seed_candidate, eval_batch, [_COMPONENT]
        )
        best = {_COMPONENT: evolved_body} if evolved_body is not None else {}
        return SimpleNamespace(
            best_candidate=best,
            best_idx=best_idx,
            total_metric_calls=7,
            num_candidates=3,
            val_aggregate_scores=val_scores if val_scores is not None else [0.1, 0.42, 0.2],
        )

    return fake_optimize


# ---------------------------------------------------------------------------
# Train / held-out split (the anti-overfit wall, partition layer)
# ---------------------------------------------------------------------------


class TestSplit:
    def test_explicit_holdout_is_disjoint_and_covers_suite(self) -> None:
        suite = _five_task_suite()
        split = split_suite(suite, holdout_task_ids=["ts-04", "ts-05"])
        assert split.holdout_task_ids == frozenset({"ts-04", "ts-05"})
        assert split.train_task_ids == frozenset({"ts-01", "ts-02", "ts-03"})
        # Disjoint and (with no cap) a full cover of the suite.
        assert split.train_task_ids & split.holdout_task_ids == frozenset()
        assert split.train_task_ids | split.holdout_task_ids == suite.task_ids()
        split.assert_disjoint()  # does not raise

    def test_seeded_fraction_split_is_deterministic_and_disjoint(self) -> None:
        suite = _five_task_suite()
        a = split_suite(suite, holdout_fraction=0.4, seed=7)
        b = split_suite(suite, holdout_fraction=0.4, seed=7)
        assert a.holdout_task_ids == b.holdout_task_ids
        assert len(a.holdout_task_ids) == 2  # ceil(0.4 * 5)
        assert a.train_task_ids & a.holdout_task_ids == frozenset()
        assert a.train_task_ids and a.holdout_task_ids
        # A different seed can choose a different held-out set.
        c = split_suite(suite, holdout_fraction=0.4, seed=99)
        assert c.train_task_ids & c.holdout_task_ids == frozenset()

    def test_max_train_tasks_caps_train_without_touching_holdout(self) -> None:
        suite = _five_task_suite()
        split = split_suite(suite, holdout_task_ids=["ts-05"], max_train_tasks=2)
        assert len(split.train_tasks) == 2
        assert {t.task_id for t in split.unused_tasks} == {"ts-03", "ts-04"}
        assert split.holdout_task_ids == frozenset({"ts-05"})
        # The dropped tasks are unused, NOT moved to held-out.
        assert split.train_task_ids & split.holdout_task_ids == frozenset()
        assert "ts-05" not in {t.task_id for t in split.unused_tasks}

    def test_unknown_holdout_id_raises(self) -> None:
        with pytest.raises(ValueError, match="not in suite"):
            split_suite(_five_task_suite(), holdout_task_ids=["nope"])

    def test_empty_suite_raises(self) -> None:
        with pytest.raises(ValueError, match="empty Task Suite"):
            split_suite(_suite())

    def test_single_task_suite_cannot_be_split(self) -> None:
        # A single-task suite cannot yield a non-empty train AND held-out split:
        # the fraction floor keeps >= 1 train task, leaving the held-out side empty.
        with pytest.raises(ValueError, match="held-out split is empty"):
            split_suite(_suite(_task("ts-01", "p1")), holdout_fraction=0.5)


# ---------------------------------------------------------------------------
# The wall, structural: the fitness function refuses held-out tasks
# ---------------------------------------------------------------------------


class TestFitnessWall:
    def _adapter(self) -> tuple[FrozenSuiteGepaAdapter, TaskSuite]:
        suite = _five_task_suite()
        split = split_suite(suite, holdout_task_ids=["ts-04", "ts-05"])
        adapter = _adapter5()
        gepa_adapter = FrozenSuiteGepaAdapter(
            suite=suite,
            adapter=adapter,
            split=split,
            verify_specs=_all_pass_specs("ts-01", "ts-02", "ts-03", "ts-04", "ts-05"),
            config=GepaConfig(),
            generated_at=_STAMP,
        )
        return gepa_adapter, suite

    def test_evaluate_raises_on_held_out_task(self) -> None:
        gepa_adapter, suite = self._adapter()
        holdout_task = next(t for t in suite.tasks if t.task_id == "ts-05")
        with pytest.raises(HeldOutLeakError, match="held-out task 'ts-05'"):
            gepa_adapter.evaluate([holdout_task], {_COMPONENT: _EVOLVED_BODY})
        # The leak attempt is never recorded as an evaluated train id.
        assert "ts-05" not in gepa_adapter.evaluated_task_ids

    def test_evaluate_raises_on_non_suite_task(self) -> None:
        gepa_adapter, _ = self._adapter()
        stranger = _task("ts-99", "stranger")
        with pytest.raises(HeldOutLeakError, match="not in the train split"):
            gepa_adapter.evaluate([stranger], {_COMPONENT: _EVOLVED_BODY})

    def test_evaluate_records_only_train_ids(self) -> None:
        gepa_adapter, suite = self._adapter()
        train_tasks = [t for t in suite.tasks if t.task_id in {"ts-01", "ts-02", "ts-03"}]
        gepa_adapter.evaluate(train_tasks, {_COMPONENT: token_efficiency_skill().body})
        assert gepa_adapter.evaluated_task_ids == {"ts-01", "ts-02", "ts-03"}
        assert gepa_adapter.evaluated_task_ids & gepa_adapter.split.holdout_task_ids == set()


# ---------------------------------------------------------------------------
# Fitness IS the frozen-suite objective, computed by the harness, fail-closed
# ---------------------------------------------------------------------------


class TestFitnessFromOutcome:
    def _outcome(self, *, rec: Recommendation, pct: float | None) -> TaskOutcome:
        return TaskOutcome(task_id="t", recommendation=rec, token_delta_pct=pct)

    def test_promote_scores_reduction_fraction(self) -> None:
        assert fitness_from_outcome(self._outcome(rec=Recommendation.PROMOTE, pct=-40.0)) == 0.4

    def test_promote_with_no_reduction_scores_zero(self) -> None:
        assert fitness_from_outcome(self._outcome(rec=Recommendation.PROMOTE, pct=0.0)) == 0.0

    def test_block_scores_zero(self) -> None:
        assert fitness_from_outcome(self._outcome(rec=Recommendation.BLOCK, pct=-40.0)) == 0.0

    def test_missing_outcome_scores_zero(self) -> None:
        assert fitness_from_outcome(None) == 0.0

    def test_reduction_is_clamped_to_one(self) -> None:
        assert fitness_from_outcome(self._outcome(rec=Recommendation.PROMOTE, pct=-150.0)) == 1.0


class TestFitnessCallsHarness:
    def _adapter(
        self, plans: dict[str, dict[str, Any]], specs: dict[str, VerifySpec]
    ) -> FrozenSuiteGepaAdapter:
        # A 2-task suite so a non-empty train/held-out split exists; ts-01 is train.
        suite = _suite(_task("ts-01", "p1"), _task("ts-02", "p2"))
        split = split_suite(suite, holdout_task_ids=["ts-02"])
        adapter = SuiteAdapter(plans)
        return FrozenSuiteGepaAdapter(
            suite=suite, adapter=adapter, split=split, verify_specs=specs, generated_at=_STAMP
        )

    def test_runs_both_arms_through_the_harness_and_scores_reduction(self) -> None:
        ga = self._adapter(
            _uniform_plans("p1", "p2", baseline=100_000, seed=60_000, evolved=40_000),
            _all_pass_specs("ts-01", "ts-02"),
        )
        train = [t for t in ga.suite.tasks if t.task_id == "ts-01"]
        batch = ga.evaluate(train, {_COMPONENT: token_efficiency_skill().body}, capture_traces=True)
        # The harness ran BOTH arms (baseline + candidate) for the one train task.
        assert len(ga.adapter.seen) == 2  # type: ignore[attr-defined]
        assert "<skill" not in (ga.adapter.seen[0].system_prompt or "")  # type: ignore[attr-defined]
        assert "<skill" in (ga.adapter.seen[1].system_prompt or "")  # type: ignore[attr-defined]
        # -40% reduction with L1 pass -> PROMOTE -> fitness 0.4.
        assert batch.scores == [0.4]
        assert batch.trajectories is not None
        assert batch.trajectories[0]["l1_outcome"] == L1Outcome.PASSED.value

    def test_fail_closed_on_candidate_crash(self) -> None:
        plans = {
            # The candidate (seed body) crashes: ~0 tokens because it did nothing.
            "p1": {"baseline": 100_000, "seed": 500, "evolved": 500, "seed_ok": False},
            "p2": {"baseline": 1, "seed": 1, "evolved": 1},
        }
        ga = self._adapter(plans, _all_pass_specs("ts-01", "ts-02"))
        train = [t for t in ga.suite.tasks if t.task_id == "ts-01"]
        batch = ga.evaluate(train, {_COMPONENT: token_efficiency_skill().body})
        # Tokens "fell" only because the candidate crashed -> fail closed to 0.0.
        assert batch.scores == [0.0]

    def test_fail_closed_without_verification(self) -> None:
        ga = self._adapter(
            _uniform_plans("p1", "p2", baseline=100_000, seed=60_000, evolved=40_000),
            {},  # no L1 verify configured -> no correctness signal -> fail closed
        )
        train = [t for t in ga.suite.tasks if t.task_id == "ts-01"]
        batch = ga.evaluate(train, {_COMPONENT: token_efficiency_skill().body})
        assert batch.scores == [0.0]

    def test_missing_component_raises(self) -> None:
        ga = self._adapter(
            _uniform_plans("p1", "p2", baseline=100_000, seed=60_000, evolved=40_000),
            _all_pass_specs("ts-01", "ts-02"),
        )
        train = [t for t in ga.suite.tasks if t.task_id == "ts-01"]
        with pytest.raises(KeyError, match="missing component"):
            ga.evaluate(train, {"wrong_name": "body"})


# ---------------------------------------------------------------------------
# End-to-end loop: only-train, returns-candidate, well-formed, held-out report
# ---------------------------------------------------------------------------


def _run_loop(
    captured: dict[str, Any],
    *,
    adapter: SuiteAdapter,
    suite: TaskSuite,
    holdout: list[str],
    train_adapter: SuiteAdapter | None = None,
    evolved_body: str | None = _EVOLVED_BODY,
    config: GepaConfig | None = None,
) -> GepaOptimizationResult:
    specs = _all_pass_specs(*[t.task_id for t in suite.tasks])
    return run_gepa_optimization(
        suite=suite,
        adapter=adapter,
        verify_specs=specs,
        config=config or GepaConfig(max_metric_calls=5),
        holdout_task_ids=holdout,
        train_adapter=train_adapter,
        gepa_optimize=make_fake_optimize(captured, evolved_body=evolved_body),
        generated_at=_STAMP,
    )


class TestLoopOnlyEvaluatesTrain:
    def test_gepa_is_handed_only_train_tasks(self) -> None:
        captured: dict[str, Any] = {}
        suite = _five_task_suite()
        adapter = _adapter5()
        result = _run_loop(captured, adapter=adapter, suite=suite, holdout=["ts-04", "ts-05"])

        train_ids = set(result.train_task_ids)
        holdout_ids = set(result.holdout_task_ids)
        assert holdout_ids == {"ts-04", "ts-05"}
        # GEPA received ONLY train tasks, as both trainset and valset.
        assert set(captured["trainset_ids"]) == train_ids
        assert set(captured["valset_ids"]) == train_ids
        assert set(captured["trainset_ids"]).isdisjoint(holdout_ids)
        # The fitness function only ever evaluated train ids — the structural wall.
        gepa_adapter: FrozenSuiteGepaAdapter = captured["adapter"]
        assert gepa_adapter.evaluated_task_ids
        assert gepa_adapter.evaluated_task_ids <= train_ids
        assert gepa_adapter.evaluated_task_ids.isdisjoint(holdout_ids)

    def test_reflection_lm_is_normalized_for_gepa(self) -> None:
        captured: dict[str, Any] = {}
        suite = _five_task_suite()
        adapter = _adapter5()
        result = _run_loop(captured, adapter=adapter, suite=suite, holdout=["ts-05"])
        # The configured MLflow URI is recorded verbatim, but GEPA gets the litellm form.
        assert result.reflection_lm == "databricks:/databricks-claude-sonnet-4-6"
        assert captured["reflection_lm"] == "databricks/databricks-claude-sonnet-4-6"
        assert captured["max_metric_calls"] == 5

    def test_proxy_train_adapter_drives_inner_loop_live_adapter_validates(self) -> None:
        captured: dict[str, Any] = {}
        suite = _five_task_suite()
        plans = _uniform_plans(*[f"p{i}" for i in range(1, 6)], baseline=100, seed=70, evolved=50)
        live = SuiteAdapter(plans)
        proxy = SuiteAdapter(plans)
        _run_loop(captured, adapter=live, suite=suite, holdout=["ts-05"], train_adapter=proxy)
        # The GEPA inner loop used the proxy; the live adapter was used only for the
        # held-out validation (its only runs are over the single held-out task).
        assert proxy.seen and all(t.prompt != "p5" for t in proxy.seen)
        assert live.seen and all(t.prompt == "p5" for t in live.seen)


class TestReturnsCandidateNotPromotion:
    def test_returns_human_gated_candidate(self) -> None:
        captured: dict[str, Any] = {}
        suite = _five_task_suite()
        adapter = _adapter5()
        result = _run_loop(captured, adapter=adapter, suite=suite, holdout=["ts-04", "ts-05"])
        assert result.human_gate_required is True
        assert result.changed is True
        assert result.evolved_skill_body == _EVOLVED_BODY
        assert result.seed_skill_body == token_efficiency_skill().body
        assert isinstance(result.holdout_evolved, Phase2Artifact)
        assert isinstance(result.holdout_seed_baseline, Phase2Artifact)
        assert any("not auto-applied or promoted" in n.lower() for n in result.notes)

    def test_does_not_mutate_the_on_disk_skill_asset(self) -> None:
        captured: dict[str, Any] = {}
        before = token_efficiency_skill().body
        suite = _five_task_suite()
        adapter = _adapter5()
        _run_loop(captured, adapter=adapter, suite=suite, holdout=["ts-05"])
        # The loop proposes a CANDIDATE; it never writes the evolved body to the asset.
        assert token_efficiency_skill().body == before

    def test_falls_back_to_seed_when_gepa_returns_no_candidate(self) -> None:
        captured: dict[str, Any] = {}
        suite = _five_task_suite()
        adapter = _adapter5()
        result = _run_loop(
            captured, adapter=adapter, suite=suite, holdout=["ts-05"], evolved_body=None
        )
        assert result.changed is False
        assert result.evolved_skill_body == result.seed_skill_body

    def test_gepa_metadata_is_recorded(self) -> None:
        captured: dict[str, Any] = {}
        suite = _five_task_suite()
        adapter = _adapter5()
        result = _run_loop(captured, adapter=adapter, suite=suite, holdout=["ts-05"])
        assert result.gepa_total_metric_calls == 7
        assert result.gepa_num_candidates == 3
        assert result.gepa_best_val_score == 0.42  # val_aggregate_scores[best_idx=1]


class TestHeldOutReport:
    def test_reports_evolved_beating_seed_on_held_out(self) -> None:
        captured: dict[str, Any] = {}
        suite = _five_task_suite()
        # evolved (50%) reduces more than seed (30%) on the held-out tasks.
        adapter = _adapter5(baseline=100_000, seed=70_000, evolved=50_000)
        result = _run_loop(captured, adapter=adapter, suite=suite, holdout=["ts-04", "ts-05"])
        assert result.holdout_evolved is not None and result.holdout_seed_baseline is not None
        assert result.holdout_evolved.realized_token_savings_pct == 50.0
        assert result.holdout_seed_baseline.realized_token_savings_pct == 30.0
        # The honest anti-overfit headline: evolved beat seed on tasks GEPA never saw.
        assert result.holdout_savings_delta_pct == 20.0
        # Held-out artifacts only cover the held-out tasks.
        assert {o.task_id for o in result.holdout_evolved.outcomes} == {"ts-04", "ts-05"}


class TestWellFormedArtifact:
    def test_result_round_trips_through_json(self) -> None:
        captured: dict[str, Any] = {}
        suite = _five_task_suite()
        adapter = _adapter5()
        result = _run_loop(captured, adapter=adapter, suite=suite, holdout=["ts-05"])
        reloaded = GepaOptimizationResult.model_validate_json(result.model_dump_json())
        assert reloaded == result

    def test_candidate_lever_config_injects_the_body(self) -> None:
        cfg = candidate_lever_config("BODYTEXT-XYZ", name="cand-x")
        assert cfg.asset_enabled is True
        assert cfg.intervention is not None
        task = AgentTask(prompt="do it")
        injected = cfg.intervention.apply(task)
        assert "BODYTEXT-XYZ" in (injected.system_prompt or "")
        assert "<skill" in (injected.system_prompt or "")
        # Pure: the original task is untouched.
        assert task.system_prompt is None

    def test_default_seed_body_is_the_token_efficiency_skill(self) -> None:
        captured: dict[str, Any] = {}
        suite = _five_task_suite()
        adapter = _adapter5()
        result = _run_loop(captured, adapter=adapter, suite=suite, holdout=["ts-05"])
        assert result.seed_skill_body.strip()
        assert result.seed_skill_body == token_efficiency_skill().body


# ---------------------------------------------------------------------------
# Lever-config guardrails carried into held-out validation
# ---------------------------------------------------------------------------


class TestLeverConfigShape:
    def test_baseline_and_candidate_constants_are_distinct(self) -> None:
        # The GEPA candidate configs reuse the lever's intervention contract.
        assert BASELINE.intervention is None
        assert CANDIDATE.intervention is not None
