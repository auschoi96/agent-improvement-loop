"""Tests for the configurable review rubric (:mod:`ail.l3.rubric`) and its prompt.

The rubric is the input config that fixes which guidelines HALO scores, on what
scale, and whether it must recommend assets. These tests pin the default (the
user's five guidelines), the validation invariants, and that the rendered prompt
is rubric-driven, unambiguous about autonomy, and free of unsubstituted sentinels.
"""

from __future__ import annotations

import pytest

from ail.l3.reviewer import build_review_prompt
from ail.l3.rubric import DEFAULT_RUBRIC, ReviewRubric, ScoredGuideline


class TestDefaultRubric:
    def test_default_has_the_four_scored_guidelines(self) -> None:
        assert DEFAULT_RUBRIC.guideline_ids() == (
            "tool_calling_efficiency",
            "token_efficiency",
            "tooling_purpose",
            "instruction_clarity",
        )

    def test_default_recommends_assets_on_1_to_5_scale(self) -> None:
        assert DEFAULT_RUBRIC.recommend_assets is True
        assert (DEFAULT_RUBRIC.score_min, DEFAULT_RUBRIC.score_max) == (1, 5)

    def test_clamp_score(self) -> None:
        assert DEFAULT_RUBRIC.clamp_score(0) == 1
        assert DEFAULT_RUBRIC.clamp_score(9) == 5
        assert DEFAULT_RUBRIC.clamp_score(3) == 3

    def test_guideline_lookup(self) -> None:
        assert DEFAULT_RUBRIC.guideline("token_efficiency").title.startswith("Token")
        assert DEFAULT_RUBRIC.guideline("nope") is None


class TestValidation:
    def test_empty_guidelines_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            ReviewRubric(rubric_id="r", guidelines=())

    def test_duplicate_ids_rejected(self) -> None:
        g = ScoredGuideline("dup", "Dup", "d")
        with pytest.raises(ValueError, match="unique"):
            ReviewRubric(rubric_id="r", guidelines=(g, g))

    def test_blank_rubric_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="rubric_id"):
            ReviewRubric(rubric_id="", guidelines=(ScoredGuideline("a", "A", "d"),))

    def test_inverted_scale_rejected(self) -> None:
        with pytest.raises(ValueError, match="score_min"):
            ReviewRubric(
                rubric_id="r",
                guidelines=(ScoredGuideline("a", "A", "d"),),
                score_min=5,
                score_max=1,
            )

    @pytest.mark.parametrize("reserved", ["review", "recommended_assets"])
    def test_reserved_guideline_id_rejected(self, reserved: str) -> None:
        # These collide with the fixed rlm_review / rlm_recommended_assets names.
        with pytest.raises(ValueError, match="reserved"):
            ReviewRubric(rubric_id="r", guidelines=(ScoredGuideline(reserved, "X", "d"),))


class TestPrompt:
    def test_prompt_is_rubric_driven_and_unambiguous(self) -> None:
        prompt = build_review_prompt("trace-abc", DEFAULT_RUBRIC)
        # The subject trace id and every guideline id reach the prompt.
        assert "trace-abc" in prompt
        for gid in DEFAULT_RUBRIC.guideline_ids():
            assert gid in prompt
        # Crisp on autonomy (dogfooding guideline 4): never pause, idle turn ends it.
        assert "AUTONOMOUSLY" in prompt
        assert "never end a turn idle" in prompt
        # The objective is stated so HALO judges toward fewer tokens / lower latency.
        assert DEFAULT_RUBRIC.objective in prompt
        # Demands the JSON verdict + the terminating marker.
        assert "```json" in prompt and "<final/>" in prompt
        # No unsubstituted sentinel tokens leak into the rendered prompt.
        assert "<<" not in prompt and ">>" not in prompt

    def test_prompt_includes_asset_schema_when_assets_requested(self) -> None:
        prompt = build_review_prompt("t", DEFAULT_RUBRIC)
        assert '"recommended_assets"' in prompt
        # The allowed asset-type vocabulary is offered (not the "other" fallback).
        assert "metric_view" in prompt and "prompt_change" in prompt
        assert "other" not in prompt.split("recommended_assets")[1].split("]")[0]

    def test_prompt_omits_assets_when_not_requested(self) -> None:
        rubric = ReviewRubric(
            rubric_id="scored-only/v1",
            guidelines=(ScoredGuideline("clarity", "Clarity", "Was it clear?"),),
            recommend_assets=False,
        )
        prompt = build_review_prompt("t", rubric)
        assert '"recommended_assets"' not in prompt
        assert "Recommended assets (" not in prompt
        # The remaining schema is still well-formed (redundancy follows guidelines).
        assert '],\n  "redundancy_findings"' in prompt
        assert "<<" not in prompt and ">>" not in prompt

    def test_prompt_reflects_custom_score_scale(self) -> None:
        rubric = ReviewRubric(
            rubric_id="scale/v1",
            guidelines=(ScoredGuideline("clarity", "Clarity", "d"),),
            score_min=0,
            score_max=10,
        )
        prompt = build_review_prompt("t", rubric)
        assert "from 0 (worst) to 10 (best)" in prompt
