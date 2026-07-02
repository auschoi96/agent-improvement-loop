"""Tests for the judge-authoring capability (:mod:`ail.judges.authoring`).

Offline by construction, mirroring :mod:`tests.test_judges`:

* **Rubric + spec building** is pure — no model, no MLflow — so
  :func:`~ail.judges.authoring.build_instructions` / ``build_judge_spec`` /
  ``normalize_judge_name`` run with no network.
* **The authored judge's shape** is proven with the *real* ``make_judge`` (via
  ``make_scorer``): it only constructs a judge, so asserting it is
  ``align``-shaped (declares a ``trace`` input field) needs no model.
* **Label-schema creation and registration** are monkeypatched, so the
  name-matched pairing and the reuse of the registration path are asserted
  without a workspace.

The one genuinely-live path (authoring against a real workspace) is gated behind
``@pytest.mark.live`` + ``AIL_LIVE_MLFLOW=1`` and self-skips otherwise.
"""

from __future__ import annotations

import os
from typing import Any, Literal

import pytest

from ail.judges import authoring
from ail.judges.authoring import (
    AuthoredJudge,
    author_judge,
    build_instructions,
    build_judge_spec,
    create_matching_label_schema,
    normalize_judge_name,
    refine_criteria,
)
from ail.judges.scorers import TOKEN_EFFICIENCY, make_scorer

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeLabelSchema:
    """Duck-typed stand-in for an MLflow ``LabelSchema`` (exposes ``.name``)."""

    def __init__(self, name: str) -> None:
        self.name = name


class _FakeRegisteredJudge:
    """A judge that records ``align`` (stands in for a registered/aligned judge)."""

    def __init__(self, name: str) -> None:
        self.name = name


class _FakeRegistration:
    """Duck-typed stand-in for :class:`ail.judges.registration.ScorerRegistration`."""

    def __init__(self, name: str, *, aligned: bool = False) -> None:
        self.scorer = _FakeRegisteredJudge(name)
        self.judge = _FakeRegisteredJudge(name)
        self.aligned = aligned
        self.report = type("Report", (), {"aligned": aligned})()


@pytest.fixture
def captured_label_schemas(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Neutralize ``create_label_schema`` (record its kwargs, return a fake schema)."""
    calls: list[dict[str, Any]] = []

    def fake_create(name: str, **kwargs: Any) -> _FakeLabelSchema:
        calls.append({"name": name, **kwargs})
        return _FakeLabelSchema(name)

    monkeypatch.setattr("mlflow.genai.label_schemas.create_label_schema", fake_create)
    return calls


@pytest.fixture
def offline_backend(
    monkeypatch: pytest.MonkeyPatch, captured_label_schemas: list[dict[str, Any]]
) -> dict[str, list[Any]]:
    """Make ``author_judge`` run offline: no Databricks config, no live registration.

    Returns a dict recording the ordered side effects so a test can assert the
    backend was configured *before* the label schema was created, and inspect the
    registration call.
    """
    events: dict[str, list[Any]] = {
        "configure": [],
        "register": [],
        "label": captured_label_schemas,
    }

    def fake_configure(**kwargs: Any) -> None:
        events["configure"].append(kwargs)

    def fake_create_aligned_scorer(spec: Any, **kwargs: Any) -> _FakeRegistration:
        events["register"].append({"spec": spec, **kwargs})
        return _FakeRegistration(spec.name)

    monkeypatch.setattr(authoring, "_configure_databricks", fake_configure)
    monkeypatch.setattr(authoring, "create_aligned_scorer", fake_create_aligned_scorer)
    return events


# ---------------------------------------------------------------------------
# normalize_judge_name — the pairing is guaranteed by one canonical name
# ---------------------------------------------------------------------------


class TestNormalizeJudgeName:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Answer Helpfulness", "answer_helpfulness"),
            ("  tool-selection.quality  ", "tool_selection_quality"),
            ("instruction_following", "instruction_following"),
            ("Is it Correct??", "is_it_correct"),
            ("multi   space", "multi_space"),
        ],
    )
    def test_canonicalizes_to_snake_case(self, raw: str, expected: str) -> None:
        assert normalize_judge_name(raw) == expected

    @pytest.mark.parametrize("bad", ["", "   ", "!!!", "123", "42_answers"])
    def test_rejects_unusable_names(self, bad: str) -> None:
        with pytest.raises(ValueError):
            normalize_judge_name(bad)


# ---------------------------------------------------------------------------
# build_instructions — concrete, gradeable, {{ trace }}-templated
# ---------------------------------------------------------------------------


class TestBuildInstructions:
    def test_graded_rubric_embeds_trace_and_is_bounded_with_rationale(self) -> None:
        instr = build_instructions("answer_helpfulness", "Did the agent answer usefully?")
        # HARD REQUIREMENT 1/3: the {{ trace }} template variable is embedded verbatim.
        assert "{{ trace }}" in instr
        # The user's criteria are carried into the rubric.
        assert "Did the agent answer usefully?" in instr
        # A bounded 1..5 scale with anchors.
        assert "1 = worst, 5 = best" in instr
        for grade in ("1 -", "2 -", "3 -", "4 -", "5 -"):
            assert grade in instr
        # A required one-line rationale that must name trace evidence.
        assert "one-line rationale" in instr
        assert "evidence in the trace" in instr

    def test_pass_fail_rubric_embeds_trace_and_requires_rationale(self) -> None:
        instr = build_instructions("safety_ok", "No unsafe content.", scale="pass_fail")
        assert "{{ trace }}" in instr
        assert "'pass'" in instr and "'fail'" in instr
        assert "rationale" in instr

    def test_rejects_blank_criteria(self) -> None:
        with pytest.raises(ValueError, match="criteria"):
            build_instructions("q", "   ")

    def test_rejects_unknown_scale(self) -> None:
        with pytest.raises(ValueError, match="scale"):
            build_instructions("q", "criteria", scale="1-10")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# build_judge_spec — the reusable ScorerSpec (offline)
# ---------------------------------------------------------------------------


class TestBuildJudgeSpec:
    def test_graded_spec_shape(self) -> None:
        spec = build_judge_spec("Answer Helpfulness", "Did the agent answer usefully?")
        assert spec.name == "answer_helpfulness"
        assert spec.feedback_value_type == Literal[1, 2, 3, 4, 5]
        # A bounded Literal loses make_judge's default mean; aggregations restored.
        assert spec.aggregations == ("mean", "median", "p90")
        assert "{{ trace }}" in spec.instructions
        # The raw description is preserved (stripped) as the spec description.
        assert spec.description == "Did the agent answer usefully?"

    def test_pass_fail_spec_shape(self) -> None:
        spec = build_judge_spec("safety_ok", "No unsafe content.", scale="pass_fail")
        assert spec.feedback_value_type == Literal["pass", "fail"]
        # A categorical guardrail keeps make_judge's default aggregation (None here).
        assert spec.aggregations is None
        assert "{{ trace }}" in spec.instructions

    def test_refine_runs_once_and_feeds_the_rubric(self) -> None:
        seen: list[dict[str, str]] = []

        def refiner(*, system: str, user: str) -> str:
            seen.append({"system": system, "user": user})
            return "SHARP: a helpful answer resolves the request completely"

        spec = build_judge_spec("helpfulness", "vague help thing", refine=True, refiner=refiner)
        assert len(seen) == 1
        assert seen[0]["user"] == "vague help thing"
        assert "SHARP: a helpful answer resolves the request completely" in spec.instructions
        # The stored description remains the human's original text, not the refined one.
        assert spec.description == "vague help thing"

    def test_no_refine_by_default_touches_no_model(self) -> None:
        def explode(*, system: str, user: str) -> str:  # pragma: no cover - must not run
            raise AssertionError("refiner must not be called when refine=False")

        spec = build_judge_spec("q_dim", "criteria", refiner=explode)
        assert "criteria" in spec.instructions


# ---------------------------------------------------------------------------
# The authored judge is structurally what align_judge / MemAlign accepts
# ---------------------------------------------------------------------------


class TestAuthoredJudgeIsAlignShaped:
    def test_built_judge_declares_a_trace_input_and_can_align(self) -> None:
        # Uses the REAL make_judge (via make_scorer): construction calls no model.
        spec = build_judge_spec("answer_helpfulness", "Did the agent answer usefully?")
        judge = make_scorer(spec, model="openai:/gpt-4.1-mini")
        # MemAlign learns from a {{ trace }} judge; score_anchor calls it with the
        # item's trace because it declares a 'trace' input field.
        field_names = {f.name for f in judge.get_input_fields()}
        assert "trace" in field_names
        # align_judge calls judge.align(traces=..., optimizer=...); the judge exposes it.
        assert callable(getattr(judge, "align", None))
        assert judge.name == spec.name

    def test_token_efficiency_stays_computed_inputs_not_trace(self) -> None:
        # RECONCILE: the deterministic-L0 token_efficiency scorer is deliberately
        # NOT a {{ trace }} judge (it judges an L0 summary). Authoring is the path
        # for human-defined QUALITY dimensions; it does not change this exclusion.
        assert "{{ trace }}" not in TOKEN_EFFICIENCY.instructions
        assert "{{ inputs }}" in TOKEN_EFFICIENCY.instructions


# ---------------------------------------------------------------------------
# refine_criteria — the optional single LLM pass (injectable seam)
# ---------------------------------------------------------------------------


class TestRefineCriteria:
    def test_returns_refined_text(self) -> None:
        out = refine_criteria("vague", refiner=lambda *, system, user: "  crisp criteria  ")
        assert out == "crisp criteria"

    def test_empty_model_output_falls_back_to_original(self) -> None:
        out = refine_criteria("original description", refiner=lambda *, system, user: "   ")
        assert out == "original description"

    def test_default_refiner_without_endpoint_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AIL_JUDGE_AUTHOR_LLM_ENDPOINT", raising=False)
        with pytest.raises(ValueError, match="endpoint"):
            refine_criteria("x")  # refiner=None, no endpoint configured


# ---------------------------------------------------------------------------
# create_matching_label_schema — name == judge name, type='feedback'
# ---------------------------------------------------------------------------


class TestCreateMatchingLabelSchema:
    def test_graded_schema_is_named_for_the_judge(
        self, captured_label_schemas: list[dict[str, Any]]
    ) -> None:
        from mlflow.genai.label_schemas import InputNumeric

        schema = create_matching_label_schema("answer_helpfulness", experiment_id="exp1")
        assert schema.name == "answer_helpfulness"
        (call,) = captured_label_schemas
        # HARD REQUIREMENT 2: name matches the judge name, type is 'feedback'.
        assert call["name"] == "answer_helpfulness"
        assert call["type"] == "feedback"
        assert isinstance(call["input"], InputNumeric)
        assert call["enable_comment"] is True
        assert call["experiment_id"] == "exp1"

    def test_pass_fail_schema_uses_passfail_input(
        self, captured_label_schemas: list[dict[str, Any]]
    ) -> None:
        from mlflow.genai.label_schemas import InputPassFail

        create_matching_label_schema("safety_ok", scale="pass_fail", overwrite=True)
        (call,) = captured_label_schemas
        assert isinstance(call["input"], InputPassFail)
        assert call["overwrite"] is True


# ---------------------------------------------------------------------------
# author_judge — the front door, reusing make_scorer + the registration path
# ---------------------------------------------------------------------------


class TestAuthorJudge:
    def test_no_register_builds_judge_and_matching_schema(
        self, offline_backend: dict[str, list[Any]]
    ) -> None:
        authored = author_judge(
            "Answer Helpfulness",
            "Did the agent answer usefully?",
            experiment_id="exp1",
            register=False,
        )
        assert isinstance(authored, AuthoredJudge)
        # THE load-bearing assertion: label-schema name == judge name.
        assert authored.label_schema.name == authored.spec.name == "answer_helpfulness"
        # The returned judge is the real, align-shaped make_scorer judge.
        assert "trace" in {f.name for f in authored.judge.get_input_fields()}
        assert "{{ trace }}" in authored.spec.instructions
        # No registration when register=False.
        assert authored.registration is None
        assert offline_backend["register"] == []
        # Backend was configured before the (mocked) label-schema write.
        assert len(offline_backend["configure"]) == 1

    def test_configures_backend_before_creating_label_schema(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        order: list[str] = []
        monkeypatch.setattr(
            authoring, "_configure_databricks", lambda **kw: order.append("configure")
        )

        def fake_create(name: str, **kwargs: Any) -> _FakeLabelSchema:
            order.append("label_schema")
            return _FakeLabelSchema(name)

        monkeypatch.setattr("mlflow.genai.label_schemas.create_label_schema", fake_create)
        author_judge("q_dim", "criteria", experiment_id="exp1", register=False)
        assert order == ["configure", "label_schema"]

    def test_register_reuses_create_aligned_scorer(
        self, offline_backend: dict[str, list[Any]]
    ) -> None:
        authored = author_judge(
            "Instruction Following",
            "Did the agent follow the user's instructions?",
            experiment_id="exp1",
            sampling_rate=0.25,
        )
        # HARD REQUIREMENT 4: registration goes through the existing path.
        (call,) = offline_backend["register"]
        assert call["spec"].name == "instruction_following"
        assert call["experiment_id"] == "exp1"
        assert call["sampling_rate"] == 0.25
        # The returned judge is the registered one from create_aligned_scorer.
        assert authored.registration is not None
        assert authored.judge is authored.registration.judge
        # Schema is still name-matched to the judge.
        assert authored.label_schema.name == "instruction_following"

    def test_register_forwards_alignment_set_and_optimizer(
        self, offline_backend: dict[str, list[Any]]
    ) -> None:
        sentinel_set = object()
        sentinel_opt = object()
        author_judge(
            "grounded_answer",
            "Is every claim supported by the retrieved context?",
            experiment_id="exp1",
            alignment_set=sentinel_set,  # type: ignore[arg-type]
            optimizer=sentinel_opt,  # type: ignore[arg-type]
        )
        (call,) = offline_backend["register"]
        assert call["alignment_set"] is sentinel_set
        assert call["optimizer"] is sentinel_opt

    def test_refine_flows_through_author_judge(self, offline_backend: dict[str, list[Any]]) -> None:
        authored = author_judge(
            "clarity",
            "vague clarity thing",
            experiment_id="exp1",
            register=False,
            refine=True,
            refiner=lambda *, system, user: "a clear answer is unambiguous and well-structured",
        )
        assert "a clear answer is unambiguous and well-structured" in authored.spec.instructions


# ---------------------------------------------------------------------------
# CLI — ail-author-judge (drives author_judge, prints the pairing)
# ---------------------------------------------------------------------------


class TestCli:
    def test_success_prints_matching_pairing(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from ail.jobs import author_judge as cli

        spec = build_judge_spec("answer_helpfulness", "Did the agent answer usefully?")
        judge = make_scorer(spec, model="openai:/gpt-4.1-mini")
        fake = AuthoredJudge(
            spec=spec,
            judge=judge,
            label_schema=_FakeLabelSchema(spec.name),
            registration=None,
        )
        monkeypatch.setattr(cli, "author_judge", lambda *a, **kw: fake)

        rc = cli.main(
            ["answer_helpfulness", "-d", "Did the agent answer usefully?", "--no-register"]
        )
        out = capsys.readouterr().out
        assert rc == 0
        assert "answer_helpfulness" in out
        assert "matches judge name: True" in out

    def test_invalid_name_exits_nonzero(self, capsys: pytest.CaptureFixture[str]) -> None:
        from ail.jobs import author_judge as cli

        # "!!!" normalizes to nothing → build_judge_spec raises ValueError before any
        # backend call, so main returns 2 without touching MLflow.
        rc = cli.main(["!!!", "-d", "something", "--no-register"])
        assert rc == 2
        assert "invalid request" in capsys.readouterr().err

    def test_refine_without_endpoint_exits_nonzero(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from ail.jobs import author_judge as cli

        monkeypatch.delenv("AIL_JUDGE_AUTHOR_LLM_ENDPOINT", raising=False)
        rc = cli.main(["clarity", "-d", "vague", "--no-register", "--refine"])
        assert rc == 2
        assert "invalid request" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Live path (gated) — author against a real workspace
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_live_author_judge_creates_matching_schema() -> None:
    """Author a judge against a real workspace and prove the name-matched pairing.

    Gated by ``AIL_LIVE_MLFLOW=1`` + ``AIL_LIVE_EXPERIMENT_ID`` and self-skips
    otherwise (no model call: register=False builds the judge and writes only the
    label schema).
    """
    if os.environ.get("AIL_LIVE_MLFLOW") != "1":
        pytest.skip("set AIL_LIVE_MLFLOW=1 to run the live judge-authoring path")
    experiment_id = os.environ.get("AIL_LIVE_EXPERIMENT_ID")
    if not experiment_id:
        pytest.skip("set AIL_LIVE_EXPERIMENT_ID to the target experiment")

    authored = author_judge(
        "ail_live_authoring_probe",
        "A throwaway probe dimension for the live authoring test.",
        experiment_id=experiment_id,
        register=False,
        overwrite_label_schema=True,
        profile=os.environ.get("DATABRICKS_CONFIG_PROFILE"),
    )
    assert authored.label_schema.name == authored.spec.name == "ail_live_authoring_probe"
    assert "{{ trace }}" in authored.spec.instructions
