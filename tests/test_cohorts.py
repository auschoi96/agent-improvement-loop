"""Tests for the Cohort abstraction and per-cohort L0 metrics (offline).

Cohorts select traces purely from :attr:`NormalizedTrace.tags`, so these tests
build normalized traces directly with tags — including arbitrary user-defined
keys, which are the primary path. The MLflow read/write integration is tested
separately in ``test_cohort_ingest.py``.
"""

from __future__ import annotations

import pytest

from ail.cohorts import (
    TAG_AGENT,
    TAG_COHORT,
    Cohort,
    TagClause,
    TagFilter,
)
from ail.ingest.base import NormalizedTrace, TokenUsage
from ail.metrics.cohort import compute_cohort_l0, compute_l0_by_cohort


def _nt(
    trace_id: str,
    tags: dict[str, str],
    *,
    model: str = "claude-opus-4-8",
    input_tokens: int = 0,
    output_tokens: int = 0,
    total: int | None = None,
) -> NormalizedTrace:
    return NormalizedTrace(
        trace_id=trace_id,
        model=model,
        tags=tags,
        token_usage=TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            _total_tokens=total,
        ),
    )


class TestTagClause:
    def test_equality(self) -> None:
        clause = TagClause("env", frozenset({"prod"}))
        assert clause.matches({"env": "prod"}) is True
        assert clause.matches({"env": "staging"}) is False
        assert clause.matches({}) is False

    def test_value_in_set(self) -> None:
        clause = TagClause("env", frozenset({"prod", "staging"}))
        assert clause.matches({"env": "prod"}) is True
        assert clause.matches({"env": "staging"}) is True
        assert clause.matches({"env": "dev"}) is False

    def test_presence_only(self) -> None:
        clause = TagClause("team")  # empty values => presence-only
        assert clause.matches({"team": "anything"}) is True
        assert clause.matches({"team": ""}) is True
        assert clause.matches({"other": "x"}) is False


class TestTagFilter:
    def test_empty_filter_matches_everything(self) -> None:
        assert TagFilter().matches({}) is True
        assert TagFilter().matches({"anything": "goes"}) is True

    def test_multiple_keys_are_anded(self) -> None:
        f = TagFilter.from_mapping({"ail.agent": "claude_code", "env": "prod"})
        assert f.matches({"ail.agent": "claude_code", "env": "prod"}) is True
        assert f.matches({"ail.agent": "claude_code", "env": "dev"}) is False
        assert f.matches({"ail.agent": "claude_code"}) is False  # missing env

    def test_arbitrary_user_keys(self) -> None:
        # No ail.* convention — a user's own keys are first-class.
        f = TagFilter.from_mapping({"squad": "alpha", "ticket": {"JIRA-1", "JIRA-2"}})
        assert f.matches({"squad": "alpha", "ticket": "JIRA-2"}) is True
        assert f.matches({"squad": "alpha", "ticket": "JIRA-9"}) is False

    def test_none_means_presence(self) -> None:
        f = TagFilter.from_mapping({"reviewed": None})
        assert f.matches({"reviewed": "yes"}) is True
        assert f.matches({"reviewed": "no"}) is True
        assert f.matches({}) is False

    def test_empty_value_set_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="presence-only"):
            TagFilter.from_mapping({"k": []})

    def test_int_values_coerced_to_str(self) -> None:
        f = TagFilter.from_mapping({"version": {1, 2}})
        assert f.matches({"version": "1"}) is True
        assert f.matches({"version": "3"}) is False


class TestTagFilterPushdown:
    def test_single_value_clause_pushes_down(self) -> None:
        f = TagFilter.from_mapping({"ail.agent": "claude_code"})
        assert f.to_mlflow_filter() == "tags.`ail.agent` = 'claude_code'"

    def test_multiple_single_value_clauses_anded(self) -> None:
        f = TagFilter.from_mapping({"ail.agent": "claude_code", "env": "prod"})
        pushed = f.to_mlflow_filter()
        assert pushed is not None
        assert "tags.`ail.agent` = 'claude_code'" in pushed
        assert "tags.`env` = 'prod'" in pushed
        assert " AND " in pushed

    def test_multi_value_and_presence_are_not_pushed(self) -> None:
        # value-in-set and presence-only cannot be faithfully/safely pushed; only
        # the single-value clause survives into the prefilter.
        f = TagFilter.from_mapping(
            {"ail.agent": "claude_code", "env": {"prod", "staging"}, "team": None}
        )
        assert f.to_mlflow_filter() == "tags.`ail.agent` = 'claude_code'"

    def test_nothing_pushable_returns_none(self) -> None:
        assert TagFilter.from_mapping({"env": {"prod", "staging"}}).to_mlflow_filter() is None
        assert TagFilter.from_mapping({"team": None}).to_mlflow_filter() is None
        assert TagFilter().to_mlflow_filter() is None

    def test_unsafe_literal_is_not_pushed_but_still_matches(self) -> None:
        # A value with a single quote can't be embedded in the filter literal, so
        # it's left to the post-filter — but matching still works in memory.
        f = TagFilter.from_mapping({"label": "o'brien"})
        assert f.to_mlflow_filter() is None
        assert f.matches({"label": "o'brien"}) is True


class TestCohortConstructors:
    def test_from_tag(self) -> None:
        c = Cohort.from_tag("c", "env", "prod")
        assert c.matches(_nt("t", {"env": "prod"})) is True

    def test_from_tags(self) -> None:
        c = Cohort.from_tags("c", {"ail.agent": "codex", "env": "prod"})
        assert c.matches(_nt("t", {"ail.agent": "codex", "env": "prod"})) is True
        assert c.matches(_nt("t", {"ail.agent": "codex"})) is False

    def test_by_agent_uses_convention_and_defaults_name(self) -> None:
        c = Cohort.by_agent("claude_code")
        assert c.name == "claude_code"
        assert c.tag_filter == TagFilter.from_mapping({TAG_AGENT: "claude_code"})

    def test_by_cohort_tag_uses_convention_and_defaults_name(self) -> None:
        c = Cohort.by_cohort_tag("nightly")
        assert c.name == "nightly"
        assert c.tag_filter == TagFilter.from_mapping({TAG_COHORT: "nightly"})

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Cohort("", TagFilter())


class TestCohortSelect:
    def _corpus(self) -> list[NormalizedTrace]:
        return [
            _nt("a1", {"ail.agent": "claude_code", "env": "prod"}),
            _nt("a2", {"ail.agent": "claude_code", "env": "dev"}),
            _nt("b1", {"ail.agent": "codex", "env": "prod"}),
            _nt("u1", {"squad": "alpha"}),  # arbitrary user key only
            _nt("n1", {}),  # untagged
        ]

    def test_selects_right_subset(self) -> None:
        selected = Cohort.by_agent("claude_code").select(self._corpus())
        assert [t.trace_id for t in selected] == ["a1", "a2"]

    def test_anded_filter_narrows_further(self) -> None:
        c = Cohort.from_tags("prod-claude", {"ail.agent": "claude_code", "env": "prod"})
        assert [t.trace_id for t in c.select(self._corpus())] == ["a1"]

    def test_arbitrary_user_key_cohort(self) -> None:
        c = Cohort.from_tag("alpha-squad", "squad", "alpha")
        assert [t.trace_id for t in c.select(self._corpus())] == ["u1"]

    def test_empty_cohort_returns_empty(self) -> None:
        # No trace carries this tag -> the collecting / not-ready state.
        c = Cohort.by_agent("gemini")
        assert c.select(self._corpus()) == []

    def test_preserves_order(self) -> None:
        c = Cohort.from_tag("prod", "env", "prod")
        assert [t.trace_id for t in c.select(self._corpus())] == ["a1", "b1"]


class TestPerCohortL0:
    def _corpus(self) -> list[NormalizedTrace]:
        return [
            _nt(
                "a1", {"ail.agent": "claude_code"}, input_tokens=1000, output_tokens=200, total=1200
            ),
            _nt(
                "a2", {"ail.agent": "claude_code"}, input_tokens=3000, output_tokens=500, total=3500
            ),
            _nt("b1", {"ail.agent": "codex"}, input_tokens=10, output_tokens=5, total=15),
        ]

    def test_cohort_l0_aggregates_only_its_traces(self) -> None:
        rep = compute_cohort_l0(self._corpus(), Cohort.by_agent("claude_code"), generated_at="x")
        assert rep.n_traces == 2
        assert rep.aggregate.tokens.total_tokens == 1200 + 3500
        assert {m.trace_id for m in rep.traces} == {"a1", "a2"}
        assert rep.aggregate.cost.priced_traces == 2

    def test_empty_cohort_yields_empty_report_not_error(self) -> None:
        rep = compute_cohort_l0(self._corpus(), Cohort.by_agent("gemini"), generated_at="x")
        assert rep.n_traces == 0
        assert rep.aggregate.tokens.total_tokens == 0
        assert rep.aggregate.cost.total_usd == 0.0

    def test_by_cohort_returns_report_per_name(self) -> None:
        cohorts = [
            Cohort.by_agent("claude_code"),
            Cohort.by_agent("codex"),
            Cohort.by_agent("none"),
        ]
        reports = compute_l0_by_cohort(self._corpus(), cohorts, generated_at="x")
        assert set(reports) == {"claude_code", "codex", "none"}
        assert reports["claude_code"].n_traces == 2
        assert reports["codex"].n_traces == 1
        assert reports["none"].n_traces == 0  # collecting state

    def test_duplicate_cohort_names_rejected(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            compute_l0_by_cohort(
                self._corpus(),
                [Cohort.by_agent("claude_code"), Cohort.from_tag("claude_code", "env", "prod")],
                generated_at="x",
            )

    def test_accepts_an_iterator_of_traces(self) -> None:
        # compute_l0_by_cohort must materialize a one-shot iterator before reuse.
        reports = compute_l0_by_cohort(
            iter(self._corpus()),
            [Cohort.by_agent("claude_code"), Cohort.by_agent("codex")],
            generated_at="x",
        )
        assert reports["claude_code"].n_traces == 2
        assert reports["codex"].n_traces == 1
