"""Tests for the requirements-intake engine (:mod:`ail.requirements`) + its gaps.

All offline. The extractor's LLM is a canned-string mock (the
:class:`ail.goals.compiler.GoalProposerLLM` seam), judge authoring and goal
persistence are injected spies / fake SQL clients, and the dynamic allowlist is
exercised in-process. No live model, MLflow, or warehouse call is ever made.

Coverage map (the slice-1 acceptance list):

* multi-dimension extraction + fail-closed on empty/garbage;
* L0-vs-judge routing (+ mis-mapped metric / duplicate-judge fail-closed);
* priority -> objective/guardrails composition;
* propose-then-confirm (nothing authored/persisted before confirm);
* GAP A round-trip (persist a confirmed goal -> the loop's load reads it back);
* GAP B (an authored judge validates via the dynamic allowlist; an unreadable
  registry fails closed);
* token_efficiency excluded from the auto-align cadence.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from ail.cohorts import Cohort
from ail.goals.allowlist import (
    AllowlistSourceError,
    is_judge,
    is_known_metric,
    judge_allowlist,
    sourced_judge_names,
)
from ail.goals.compiler import CompiledGoal, GoalTarget, Guardrail
from ail.requirements import (
    COMPILED_GOAL_COLUMNS,
    PlanExecution,
    RequirementDimension,
    RequirementsExtractionError,
    RequirementsNotConfirmedError,
    RequirementsRoutingError,
    build_plan,
    execute_plan,
    extract_dimensions,
    load_persisted_goal,
    persist_compiled_goal,
    plan_requirements,
)
from ail.requirements import persistence as persistence_mod

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def mock_llm(payload: Any, *, fence: bool = False, prose: bool = False):  # type: ignore[no-untyped-def]
    """A mock extractor LLM returning a fixed response (JSON list or raw string)."""
    text = json.dumps(payload) if not isinstance(payload, str) else payload
    if fence:
        text = f"```json\n{text}\n```"
    if prose:
        text = f"Here are the dimensions:\n{text}\nHope that helps!"

    def _llm(*, system: str, user: str) -> str:
        return text

    return _llm


THREE_DIMS = [
    {
        "name": "no hallucinated tool calls",
        "description": "never invent tool calls the user did not enable",
        "user_priority": 1,
        "metric": None,
    },
    {
        "name": "response conciseness",
        "description": "answers should be brief and to the point",
        "user_priority": 2,
        "metric": None,
    },
    {
        "name": "latency",
        "description": "responses should be fast",
        "user_priority": 3,
        "metric": "duration_seconds",
    },
]


def _spy_author(log: list[tuple[str, str]]):  # type: ignore[no-untyped-def]
    def _author(name: str, description: str, *, experiment_id: str) -> Any:
        log.append(("author", name))
        return SimpleNamespace(name=name, experiment_id=experiment_id)

    return _author


# ---------------------------------------------------------------------------
# Piece 1: multi-dimension extraction (+ fail-closed)
# ---------------------------------------------------------------------------


class TestExtraction:
    def test_extracts_multiple_distinct_dimensions(self) -> None:
        dims = extract_dimensions("...", llm=mock_llm(THREE_DIMS))
        assert [d.name for d in dims] == [
            "no hallucinated tool calls",
            "response conciseness",
            "latency",
        ]
        assert [d.user_priority for d in dims] == [1, 2, 3]
        assert dims[2].metric == "duration_seconds"
        assert dims[0].metric is None

    def test_single_dimension_is_valid(self) -> None:
        dims = extract_dimensions("just be concise", llm=mock_llm([THREE_DIMS[1]]))
        assert len(dims) == 1 and dims[0].name == "response conciseness"

    def test_fenced_and_prose_wrapped_json_recovered(self) -> None:
        assert len(extract_dimensions("x", llm=mock_llm(THREE_DIMS, fence=True))) == 3
        assert len(extract_dimensions("x", llm=mock_llm(THREE_DIMS, prose=True))) == 3

    def test_blank_metric_string_becomes_none(self) -> None:
        payload = [
            {
                "name": "helpfulness",
                "description": "be helpful",
                "user_priority": 1,
                "metric": "   ",
            }
        ]
        dims = extract_dimensions("x", llm=mock_llm(payload))
        assert dims[0].metric is None

    def test_empty_requirements_fails_closed(self) -> None:
        with pytest.raises(RequirementsExtractionError, match="non-empty"):
            extract_dimensions("   ", llm=mock_llm(THREE_DIMS))

    def test_non_json_output_fails_closed(self) -> None:
        with pytest.raises(RequirementsExtractionError, match="valid JSON array"):
            extract_dimensions("x", llm=mock_llm("I cannot help with that."))

    def test_non_array_output_fails_closed(self) -> None:
        with pytest.raises(RequirementsExtractionError, match="must be a JSON array"):
            extract_dimensions(
                "x", llm=mock_llm({"name": "a", "description": "b", "user_priority": 1})
            )

    def test_empty_array_fails_closed_never_fabricates(self) -> None:
        with pytest.raises(RequirementsExtractionError, match="empty dimensions array"):
            extract_dimensions("x", llm=mock_llm([]))

    def test_blank_name_item_fails_closed(self) -> None:
        payload = [{"name": "  ", "description": "b", "user_priority": 1}]
        with pytest.raises(RequirementsExtractionError, match="malformed"):
            extract_dimensions("x", llm=mock_llm(payload))

    def test_unknown_field_item_fails_closed(self) -> None:
        payload = [{"name": "a", "description": "b", "user_priority": 1, "surprise": 1}]
        with pytest.raises(RequirementsExtractionError, match="malformed"):
            extract_dimensions("x", llm=mock_llm(payload))


# ---------------------------------------------------------------------------
# Piece 2: routing + composition
# ---------------------------------------------------------------------------


class TestRoutingAndComposition:
    def test_routes_l0_vs_judge_and_composes_by_priority(self) -> None:
        plan = plan_requirements("...", Cohort.by_agent("claude_code"), llm=mock_llm(THREE_DIMS))
        # highest priority (1) is the objective — a quality dimension => a judge
        obj = plan.objective
        assert obj.name == "no hallucinated tool calls"
        assert obj.kind == "memalign_judge"
        assert obj.judge_name == "no_hallucinated_tool_calls"
        assert plan.goal.objective_metric == "no_hallucinated_tool_calls"
        assert plan.goal.direction == "maximize"
        # latency routed to a deterministic L0 metric (no judge)
        det = plan.deterministic_metrics
        assert [d.metric for d in det] == ["duration_seconds"]
        assert det[0].kind == "deterministic_l0"
        # the two quality dimensions are the judges that would be authored
        assert sorted(d.judge_name for d in plan.judges_to_author) == [
            "no_hallucinated_tool_calls",
            "response_conciseness",
        ]

    def test_guardrails_compose_from_non_primary_dimensions(self) -> None:
        plan = plan_requirements("...", "claude_code", llm=mock_llm(THREE_DIMS))
        gr = {(g.name, g.kind) for g in plan.goal.guardrails}
        # judged objective must ALSO be a judge guardrail (the readiness contract)
        assert ("no_hallucinated_tool_calls", "judge") in gr
        # the other quality dimension -> judge guardrail; latency -> metric guardrail
        assert ("response_conciseness", "judge") in gr
        assert ("duration_seconds", "metric") in gr
        assert plan.goal.requires_quality is True
        assert set(plan.goal.guardrail_names) == {
            "no_hallucinated_tool_calls",
            "response_conciseness",
        }

    def test_deterministic_primary_yields_no_judge_and_minimize(self) -> None:
        dims = [
            RequirementDimension(
                name="latency", description="fast", user_priority=1, metric="duration_seconds"
            ),
            RequirementDimension(
                name="cost", description="cheap", user_priority=2, metric="total_usd"
            ),
        ]
        plan = build_plan(dims, "claude_code")
        assert plan.goal.objective_metric == "duration_seconds"
        assert plan.goal.direction == "minimize"
        assert plan.goal.requires_quality is False
        assert list(plan.goal.guardrail_names) == []  # no judges
        assert {(g.name, g.kind) for g in plan.goal.guardrails} == {("total_usd", "metric")}

    def test_single_quality_dimension_composes_judge_objective(self) -> None:
        dims = [RequirementDimension(name="conciseness", description="brief", user_priority=1)]
        plan = build_plan(dims, "claude_code")
        assert plan.goal.objective_metric == "conciseness"
        # a judged objective is self-guardrailed so readiness requires the judge
        assert {(g.name, g.kind) for g in plan.goal.guardrails} == {("conciseness", "judge")}

    def test_priority_ordering_picks_lowest_number_as_objective(self) -> None:
        dims = [
            RequirementDimension(
                name="cost", description="cheap", user_priority=5, metric="total_usd"
            ),
            RequirementDimension(
                name="tokens", description="few", user_priority=2, metric="total_tokens"
            ),
        ]
        plan = build_plan(dims, "claude_code")
        assert plan.objective.name == "tokens"
        assert plan.goal.objective_metric == "total_tokens"

    def test_mismapped_metric_fails_closed(self) -> None:
        # A candidate metric that is not a real L0 metric is refused, not treated
        # as deterministic (and not silently turned into a judge).
        dims = [
            RequirementDimension(
                name="quality", description="good", user_priority=1, metric="correctness"
            )
        ]
        with pytest.raises(RequirementsRoutingError, match="not a deterministic L0 metric"):
            build_plan(dims, "claude_code")

    def test_duplicate_judge_name_fails_closed(self) -> None:
        dims = [
            RequirementDimension(name="conciseness", description="brief", user_priority=1),
            RequirementDimension(name="Conciseness!", description="also brief", user_priority=2),
        ]
        with pytest.raises(RequirementsRoutingError, match="normalize to judge name"):
            build_plan(dims, "claude_code")

    def test_empty_dimensions_fails_closed(self) -> None:
        with pytest.raises(RequirementsRoutingError, match="at least one dimension"):
            build_plan([], "claude_code")

    def test_composed_goal_is_unconfirmed_and_a_goalview(self) -> None:
        from ail.readiness import GoalView

        plan = plan_requirements("...", "claude_code", llm=mock_llm(THREE_DIMS))
        assert plan.goal.human_confirmed is False
        assert isinstance(plan.goal, GoalView)


# ---------------------------------------------------------------------------
# Propose-then-confirm: nothing authored / persisted before confirm
# ---------------------------------------------------------------------------


class TestProposeThenConfirm:
    def test_planning_authors_and_persists_nothing(self) -> None:
        log: list[tuple[str, str]] = []
        plan = plan_requirements("...", "claude_code", llm=mock_llm(THREE_DIMS))
        # Building the plan must not have called the (unused) author/persist seams,
        # and the goal must be unconfirmed.
        assert log == []
        assert plan.confirmed is False
        assert plan.goal.human_confirmed is False

    def test_execute_refuses_unconfirmed_without_side_effects(self) -> None:
        log: list[tuple[str, str]] = []
        plan = plan_requirements("...", "claude_code", llm=mock_llm(THREE_DIMS))
        persisted: list[CompiledGoal] = []
        with pytest.raises(RequirementsNotConfirmedError, match="not confirmed"):
            execute_plan(
                plan,
                experiment_id="e",
                author=_spy_author(log),
                persist=lambda g: persisted.append(g),
            )
        assert log == []  # no judge authored
        assert persisted == []  # nothing persisted

    def test_execute_after_confirm_authors_then_persists(self) -> None:
        log: list[tuple[str, str]] = []
        plan = plan_requirements("...", "claude_code", llm=mock_llm(THREE_DIMS)).confirm()

        def _persist(goal: CompiledGoal) -> None:
            log.append(("persist", goal.objective_metric))

        result = execute_plan(plan, experiment_id="e", author=_spy_author(log), persist=_persist)
        assert isinstance(result, PlanExecution)
        # both quality dimensions authored, in plan order
        assert result.authored_names == ("no_hallucinated_tool_calls", "response_conciseness")
        assert result.persisted is True
        # judges are authored BEFORE the goal is persisted
        kinds = [k for k, _ in log]
        assert kinds == ["author", "author", "persist"]
        # the persisted goal is the confirmed one
        assert plan.goal.human_confirmed is True

    def test_execute_without_persist_only_authors(self) -> None:
        log: list[tuple[str, str]] = []
        plan = plan_requirements("...", "claude_code", llm=mock_llm(THREE_DIMS)).confirm()
        result = execute_plan(plan, experiment_id="e", author=_spy_author(log))
        assert result.persisted is False
        assert [k for k, _ in log] == ["author", "author"]


# ---------------------------------------------------------------------------
# GAP B: dynamic allowlist
# ---------------------------------------------------------------------------


class TestDynamicAllowlist:
    def test_authored_judge_validates_only_inside_allowlist(self) -> None:
        # An authored dimension's judge is NOT a built-in; it validates only when the
        # dynamic allowlist admits it. The composer does this for us — prove it holds
        # for a direct CompiledGoal construction too.
        assert is_judge("no_hallucinated_tool_calls") is False
        with judge_allowlist(["no_hallucinated_tool_calls"]):
            assert is_judge("no_hallucinated_tool_calls") is True
            assert is_known_metric("no_hallucinated_tool_calls") is True
            goal = CompiledGoal(
                objective_metric="no_hallucinated_tool_calls",
                direction="maximize",
                target=GoalTarget(value=0.1, kind="relative"),
                guardrails=(
                    Guardrail(
                        name="no_hallucinated_tool_calls", kind="judge", must_not_regress=True
                    ),
                ),
                cohort="claude_code",
            )
            assert goal.objective_metric == "no_hallucinated_tool_calls"
        # reset on exit — no leak
        assert is_judge("no_hallucinated_tool_calls") is False

    def test_deterministic_l0_set_intact_under_dynamic_context(self) -> None:
        with judge_allowlist(["conciseness"]):
            # L0 metrics still route as L0, not as judges
            from ail.goals.allowlist import is_l0_metric

            assert is_l0_metric("duration_seconds") is True
            assert is_judge("duration_seconds") is False

    def test_sourced_judge_names_unions_registry_with_builtins(self) -> None:
        def lister(experiment_id: str) -> list[Any]:
            assert experiment_id == "exp-1"
            return [SimpleNamespace(name="conciseness"), SimpleNamespace(name="correctness")]

        names = sourced_judge_names(lister, experiment_id="exp-1")
        assert "conciseness" in names  # authored/registered
        assert "correctness" in names  # built-in preserved
        assert "token_efficiency" in names  # built-ins are the floor

    def test_unreadable_registry_fails_closed_not_allow_everything(self) -> None:
        def broken(experiment_id: str) -> list[Any]:
            raise RuntimeError("registry unreachable")

        with pytest.raises(AllowlistSourceError, match="could not re-source"):
            sourced_judge_names(broken, experiment_id="exp-1")
        # and a failed source did not widen the allowlist
        assert is_judge("conciseness") is False


# ---------------------------------------------------------------------------
# GAP A: persist a confirmed goal -> the loop's load reads it back
# ---------------------------------------------------------------------------


def _warehouse_cell(value: Any) -> Any:
    """Render a Python value the way the SQL result API returns it (strings/None)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


class TestGapAPersistence:
    def _confirmed_goal(self) -> CompiledGoal:
        with judge_allowlist(["no_hallucinated_tool_calls"]):
            return CompiledGoal(
                objective_metric="no_hallucinated_tool_calls",
                direction="maximize",
                target=GoalTarget(value=0.1, kind="relative"),
                guardrails=(
                    Guardrail(
                        name="no_hallucinated_tool_calls", kind="judge", must_not_regress=True
                    ),
                    Guardrail(name="duration_seconds", kind="metric", must_not_regress=True),
                ),
                cohort="claude_code",
            ).confirm()

    def test_refuses_to_persist_unconfirmed_goal(self, fake_sql_client) -> None:  # type: ignore[no-untyped-def]
        goal = self._confirmed_goal().model_copy(update={"human_confirmed": False})
        client = fake_sql_client(lambda s: None)
        with pytest.raises(ValueError, match="unconfirmed goal"):
            persist_compiled_goal(goal, agent_name="claude_code", client=client, warehouse_id="w")

    def test_persist_issues_ddl_and_agent_scoped_replace(self, fake_sql_client) -> None:  # type: ignore[no-untyped-def]
        goal = self._confirmed_goal()
        client = fake_sql_client(lambda s: None)
        n = persist_compiled_goal(
            goal, agent_name="claude_code", client=client, warehouse_id="w", catalog="c", schema="s"
        )
        assert n == 1
        executed = client.statement_execution.executed
        assert any(s.startswith("CREATE SCHEMA IF NOT EXISTS") for s in executed)
        assert any(
            "CREATE TABLE IF NOT EXISTS" in s and "agent_compiled_goals" in s for s in executed
        )
        # atomic REPLACE scoped to this agent only
        assert any(
            "REPLACE WHERE agent_name = 'claude_code'" in s and "INSERT INTO" in s for s in executed
        )

    def test_round_trip_persist_then_load(self, fake_sql_client) -> None:  # type: ignore[no-untyped-def]
        goal = self._confirmed_goal()
        # The row the writer serializes (via the SAME _goal_row + column order the
        # loader reads), rendered as the SQL API would return it (strings/None).
        row = [
            _warehouse_cell(v)
            for v in persistence_mod._goal_row(
                goal,
                agent_name="claude_code",
                requirements_text="be safe and fast",
                stamp="2026-07-08T00:00:00+00:00",
            )
        ]
        client = fake_sql_client({"SELECT": (COMPILED_GOAL_COLUMNS, [row])})

        loaded = load_persisted_goal(
            agent_name="claude_code", client=client, warehouse_id="w", catalog="c", schema="s"
        )
        assert loaded is not None
        assert loaded.objective_metric == goal.objective_metric
        assert loaded.direction == goal.direction
        assert loaded.target.value == goal.target.value
        assert loaded.target.kind == goal.target.kind
        assert loaded.human_confirmed is True
        assert loaded.cohort_name == "claude_code"
        assert {(g.name, g.kind) for g in loaded.guardrails} == {
            ("no_hallucinated_tool_calls", "judge"),
            ("duration_seconds", "metric"),
        }
        # GAP B closed on the read side: a goal referencing an authored judge
        # reconstructs, and the dynamic admission did not leak past load.
        assert is_judge("no_hallucinated_tool_calls") is False

    def test_load_missing_row_returns_none(self, fake_sql_client) -> None:  # type: ignore[no-untyped-def]
        client = fake_sql_client({"SELECT": (COMPILED_GOAL_COLUMNS, [])})
        assert (
            load_persisted_goal(agent_name="claude_code", client=client, warehouse_id="w") is None
        )

    def test_load_missing_table_returns_none_fail_soft(self) -> None:
        class _Exec:
            def execute_statement(self, **kwargs: Any) -> Any:
                raise RuntimeError("TABLE_OR_VIEW_NOT_FOUND")

        client = SimpleNamespace(statement_execution=_Exec())
        assert (
            load_persisted_goal(agent_name="claude_code", client=client, warehouse_id="w") is None
        )


# ---------------------------------------------------------------------------
# GAP A: the loop's goal-load prefers the persisted intake goal
# ---------------------------------------------------------------------------


class TestLoopGoalLoad:
    def _args(self) -> SimpleNamespace:
        return SimpleNamespace(
            agent="claude_code",
            experiment="e",
            warehouse_id="w",
            catalog="c",
            schema="s",
            objective_metric="total_tokens",
            goal_direction="minimize",
            goal_target=-0.30,
            goal_target_kind="relative",
            guardrail_judge=None,
            objective_baseline=None,
            goal_confirmed="true",
        )

    def test_prefers_confirmed_persisted_goal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from ail.jobs import companion_planner as cp

        persisted = CompiledGoal(
            objective_metric="total_usd",
            direction="minimize",
            target=GoalTarget(value=-0.2, kind="relative"),
            cohort="claude_code",
        ).confirm()
        monkeypatch.setattr(cp, "load_persisted_goal", lambda args: persisted)
        goal, source = cp.resolve_goal(self._args())
        assert source == "persisted-intake"
        assert goal.objective_metric == "total_usd"
        assert goal.human_confirmed is True

    def test_falls_back_to_args_when_no_persisted_goal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ail.jobs import companion_planner as cp

        monkeypatch.setattr(cp, "load_persisted_goal", lambda args: None)
        goal, source = cp.resolve_goal(self._args())
        assert source == "cli-args"
        assert goal.objective_metric == "total_tokens"
        assert goal.human_confirmed is True  # goal_confirmed="true"

    def test_ignores_unconfirmed_persisted_goal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from ail.jobs import companion_planner as cp

        unconfirmed = CompiledGoal(
            objective_metric="total_usd",
            direction="minimize",
            target=GoalTarget(value=-0.2, kind="relative"),
            cohort="claude_code",
        )
        monkeypatch.setattr(cp, "load_persisted_goal", lambda args: unconfirmed)
        goal, source = cp.resolve_goal(self._args())
        assert source == "cli-args"  # a stale unconfirmed write never bypasses the confirm gate
        assert goal.objective_metric == "total_tokens"

    def test_load_skipped_without_static_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from ail.jobs import companion_planner as cp

        monkeypatch.delenv("DATABRICKS_HOST", raising=False)
        monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
        # No static token => the read is skipped (offline), returns None.
        assert cp.load_persisted_goal(self._args()) is None


# ---------------------------------------------------------------------------
# Piece 5: token_efficiency excluded from the auto-align cadence
# ---------------------------------------------------------------------------


class TestAutoAlignExclusion:
    def test_token_efficiency_marked_not_auto_alignable(self) -> None:
        from ail.judges.scorers import CORRECTNESS, TOKEN_EFFICIENCY

        assert TOKEN_EFFICIENCY.auto_alignable is False
        # the other built-ins remain alignable
        assert CORRECTNESS.auto_alignable is True

    def test_authored_judge_is_auto_alignable_by_default(self) -> None:
        from ail.judges.authoring import build_judge_spec

        spec = build_judge_spec("no hallucinated tool calls", "never invent tools")
        assert spec.auto_alignable is True

    def test_cadence_excludes_non_auto_alignable_judges(self) -> None:
        from ail.judges.auto_align import AutoAlignState, auto_align_scorers

        class _Source:
            def iter_traces(self, **kwargs: Any):  # type: ignore[no-untyped-def]
                return iter(())

        class _Store:
            def read(self, judge_name: str) -> AutoAlignState:
                return AutoAlignState()

            def write(self, judge_name: str, state: AutoAlignState) -> None:  # pragma: no cover
                pass

        report = auto_align_scorers("exp", source=_Source(), store=_Store(), register=False)
        judged = {r.judge_name for r in report.results}
        assert "token_efficiency" not in judged
        assert judged == {"correctness", "modularity", "groundedness"}
        assert report.n_failed == 0
