"""Tests for the ground-truth contract (our own clean-room schema).

Cover the invariants the rest of the package leans on: unknown fields are
rejected, provenance is required, instances are frozen, and the promotion gate
is encoded as data on the case.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ail.groundtruth.schema import (
    SCHEMA_VERSION,
    Expectations,
    GroundTruthCase,
    GroundTruthError,
    GroundTruthSet,
    Pool,
    ReviewRecord,
    ReviewStatus,
    Source,
    SourceKind,
    TaskInput,
)


def _candidate(case_id: str = "gt-1") -> GroundTruthCase:
    return GroundTruthCase(
        case_id=case_id,
        task_input=TaskInput(prompt="add two numbers"),
        sources=[Source(kind=SourceKind.TRACE, ref="tr-1")],
    )


def _approved(case_id: str, pool: Pool) -> GroundTruthCase:
    """A fully human-approved (promotable) case, for building pool sets."""
    return _candidate(case_id).model_copy(
        update={
            "expectations": Expectations(expected_response="3"),
            "regression_intent": "guards arithmetic",
            "target_pool": pool,
            "review": ReviewRecord(status=ReviewStatus.APPROVED, reviewer="austin"),
        }
    )


class TestModelConfig:
    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            GroundTruthCase(
                case_id="x",
                task_input=TaskInput(prompt="p"),
                sources=[Source(kind=SourceKind.TRACE, ref="tr")],
                surprise="nope",  # type: ignore[call-arg]
            )

    def test_sources_required_non_empty(self) -> None:
        with pytest.raises(ValidationError):
            GroundTruthCase(
                case_id="x",
                task_input=TaskInput(prompt="p"),
                sources=[],
            )

    def test_case_is_frozen(self) -> None:
        case = _candidate()
        with pytest.raises(ValidationError):
            case.case_id = "mutated"  # type: ignore[misc]

    def test_default_schema_version(self) -> None:
        assert _candidate().schema_version == SCHEMA_VERSION


class TestExpectations:
    def test_empty_is_not_filled(self) -> None:
        assert Expectations().is_filled() is False

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"expected_response": "x"},
            {"must_include": ["x"]},
            {"must_not_include": ["x"]},
            {"rubric": "x"},
            {"assertions": ["x"]},
        ],
    )
    def test_any_authored_field_counts_as_filled(self, kwargs: dict[str, object]) -> None:
        assert Expectations(**kwargs).is_filled() is True

    def test_reviewer_notes_alone_is_not_filled(self) -> None:
        # Notes without an actual expectation do not count as a real expectation.
        assert Expectations(reviewer_notes="looks fine").is_filled() is False


class TestPromotionGate:
    def test_fresh_candidate_is_not_promotable(self) -> None:
        case = _candidate()
        assert case.is_promotable() is False
        blockers = case.promotion_blockers()
        assert any("not approved" in b for b in blockers)
        assert any("expectations are empty" in b for b in blockers)
        assert any("regression_intent is blank" in b for b in blockers)
        assert any("no target pool" in b for b in blockers)

    def test_fully_reviewed_case_is_promotable(self) -> None:
        case = _candidate().model_copy(
            update={
                "expectations": Expectations(expected_response="3"),
                "regression_intent": "guards basic arithmetic",
                "target_pool": Pool.ALIGNMENT_SET,
                "review": ReviewRecord(status=ReviewStatus.APPROVED, reviewer="austin"),
            }
        )
        assert case.promotion_blockers() == []
        assert case.is_promotable() is True


class TestGroundTruthSet:
    def test_rejects_mixed_pools(self) -> None:
        case = _approved("c1", Pool.HUMAN_ANCHOR)
        with pytest.raises(GroundTruthError):
            GroundTruthSet(pool=Pool.ALIGNMENT_SET, name="s", cases=[case])

    def test_rejects_duplicate_case_ids(self) -> None:
        a = _approved("dup", Pool.ALIGNMENT_SET)
        b = _approved("dup", Pool.ALIGNMENT_SET)
        with pytest.raises(GroundTruthError):
            GroundTruthSet(pool=Pool.ALIGNMENT_SET, name="s", cases=[a, b])

    def test_rejects_unapproved_case(self) -> None:
        # A candidate that has not cleared the human gate cannot be pooled — this
        # is the schema-level half of closing the store-bypass hole.
        candidate = _candidate("c1").model_copy(update={"target_pool": Pool.ALIGNMENT_SET})
        with pytest.raises(GroundTruthError):
            GroundTruthSet(pool=Pool.ALIGNMENT_SET, name="s", cases=[candidate])

    def test_rejects_target_pool_none(self) -> None:
        candidate = _candidate("c1")  # target_pool is None
        with pytest.raises(GroundTruthError):
            GroundTruthSet(pool=Pool.ALIGNMENT_SET, name="s", cases=[candidate])

    def test_task_suite_set_must_be_empty(self) -> None:
        case = _approved("c1", Pool.TASK_SUITE)
        with pytest.raises(GroundTruthError):
            GroundTruthSet(pool=Pool.TASK_SUITE, name="s", cases=[case])
        # …but an empty Task Suite set is fine (what the store returns on load).
        assert GroundTruthSet(pool=Pool.TASK_SUITE, name="s").cases == []

    def test_case_ids_helper(self) -> None:
        case = _approved("c1", Pool.ALIGNMENT_SET)
        gt = GroundTruthSet(pool=Pool.ALIGNMENT_SET, name="s", cases=[case])
        assert gt.case_ids() == {"c1"}


class TestJsonRoundTrip:
    def test_case_round_trips_through_json(self) -> None:
        case = _candidate().model_copy(
            update={
                "expectations": Expectations(must_include=["return"], rubric="must be a function"),
                "regression_intent": "guards the add() helper",
                "target_pool": Pool.HUMAN_ANCHOR,
                "review": ReviewRecord(status=ReviewStatus.APPROVED, reviewer="austin"),
            }
        )
        restored = GroundTruthCase.model_validate_json(case.model_dump_json())
        assert restored == case
        assert restored.expectations.must_include == ["return"]
        assert restored.target_pool is Pool.HUMAN_ANCHOR
