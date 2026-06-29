"""Tests for the MLflow integration of cohorts (offline, no network).

Two paths:

* **Cohort-aware reads** — ``mlflow.search_traces`` is stubbed so the test
  drives real ``Trace`` objects (rebuilt from the recorded fixture, with tags
  swapped in) through the exact normalization + filtering path. The stub
  deliberately ignores the pushdown filter and returns *everything*, which
  proves the in-memory post-filter is the real source of truth; the pushdown
  string is asserted separately.
* **Tag writes** — ``apply_trace_tags`` is exercised with a MOCKED client. We
  never attempt a live write (it may hit a PermissionDenied wall) and the build
  never depends on one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call

from ail.cohorts import Cohort
from ail.ingest.mlflow_source import MLflowTraceSource, apply_trace_tags

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _trace_with_tags(trace_id: str, tags: dict[str, str]) -> Any:
    """Rebuild the recorded fixture ``Trace`` with a given id and tag set."""
    from mlflow.entities import Trace

    data = json.loads((FIXTURE_DIR / "synthetic_trace.json").read_text())
    data["info"]["trace_id"] = trace_id
    data["info"]["tags"] = tags
    return Trace.from_dict(data)


class _FakeSearch:
    """Stub for ``mlflow.search_traces``: records kwargs, returns a fixed list."""

    def __init__(self, traces: list[Any]) -> None:
        self.traces = traces
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> list[Any]:
        self.calls.append(kwargs)
        return self.traces


def _patch_source(monkeypatch: Any, traces: list[Any]) -> tuple[MLflowTraceSource, _FakeSearch]:
    import mlflow

    fake = _FakeSearch(traces)
    monkeypatch.setattr(mlflow, "search_traces", fake)
    # _configure only sets MLflow URIs (no network); skip it to avoid mutating
    # global MLflow state during the offline suite.
    monkeypatch.setattr(MLflowTraceSource, "_configure", lambda self: None)
    return MLflowTraceSource(), fake


def _corpus() -> list[Any]:
    return [
        _trace_with_tags("t-claude-prod", {"ail.agent": "claude_code", "env": "prod"}),
        _trace_with_tags("t-claude-dev", {"ail.agent": "claude_code", "env": "dev"}),
        _trace_with_tags("t-codex-prod", {"ail.agent": "codex", "env": "prod"}),
    ]


class TestCohortAwareReads:
    def test_equality_cohort_pushes_filter_and_post_filters(self, monkeypatch: Any) -> None:
        source, fake = _patch_source(monkeypatch, _corpus())
        out = source.fetch_cohort_traces(Cohort.by_agent("claude_code"), experiment_id="exp1")
        # post-filter keeps exactly the claude_code traces
        assert {t.trace_id for t in out} == {"t-claude-prod", "t-claude-dev"}
        # the equality clause was pushed into the backend search
        assert fake.calls[0]["filter_string"] == "tags.`ail.agent` = 'claude_code'"
        assert fake.calls[0]["locations"] == ["exp1"]

    def test_partial_pushdown_with_post_filter(self, monkeypatch: Any) -> None:
        source, fake = _patch_source(monkeypatch, _corpus())
        cohort = Cohort.from_tags(
            "prod-claude", {"ail.agent": "claude_code", "env": {"prod", "staging"}}
        )
        out = source.fetch_cohort_traces(cohort, experiment_id="exp1")
        # only the equality clause is pushed; the value-in-set env clause is
        # enforced in memory -> only claude_code AND env in {prod,staging}
        assert [t.trace_id for t in out] == ["t-claude-prod"]
        assert fake.calls[0]["filter_string"] == "tags.`ail.agent` = 'claude_code'"

    def test_user_filter_is_anded_with_pushdown(self, monkeypatch: Any) -> None:
        source, fake = _patch_source(monkeypatch, _corpus())
        source.fetch_cohort_traces(
            Cohort.by_agent("claude_code"), experiment_id="exp1", filter_string="status = 'OK'"
        )
        assert (
            fake.calls[0]["filter_string"] == "status = 'OK' AND tags.`ail.agent` = 'claude_code'"
        )

    def test_non_pushable_cohort_passes_user_filter_through(self, monkeypatch: Any) -> None:
        source, fake = _patch_source(monkeypatch, _corpus())
        # value-in-set only -> nothing pushable -> filter_string is None
        out = source.fetch_cohort_traces(
            Cohort.from_tag("envs", "env", {"prod", "staging"}), experiment_id="exp1"
        )
        assert {t.trace_id for t in out} == {"t-claude-prod", "t-codex-prod"}
        assert fake.calls[0]["filter_string"] is None

    def test_empty_cohort_returns_empty(self, monkeypatch: Any) -> None:
        source, _ = _patch_source(monkeypatch, _corpus())
        out = source.fetch_cohort_traces(Cohort.by_agent("gemini"), experiment_id="exp1")
        assert out == []

    def test_max_results_passed_through(self, monkeypatch: Any) -> None:
        source, fake = _patch_source(monkeypatch, _corpus())
        source.fetch_cohort_traces(
            Cohort.by_agent("claude_code"), experiment_id="exp1", max_results=5
        )
        assert fake.calls[0]["max_results"] == 5

    def test_iter_is_lazy_and_yields_normalized(self, monkeypatch: Any) -> None:
        source, _ = _patch_source(monkeypatch, _corpus())
        gen = source.iter_cohort_traces(Cohort.by_agent("codex"), experiment_id="exp1")
        results = list(gen)
        assert len(results) == 1
        assert results[0].trace_id == "t-codex-prod"
        assert results[0].tags["ail.agent"] == "codex"


class TestApplyTraceTags:
    def test_calls_client_for_each_trace_and_tag(self) -> None:
        client = MagicMock()
        written = apply_trace_tags(
            ["t1", "t2"], {"ail.agent": "claude_code", "env": "prod"}, client=client
        )
        assert written == 4  # 2 traces x 2 tags
        assert client.set_trace_tag.call_args_list == [
            call("t1", "ail.agent", "claude_code"),
            call("t1", "env", "prod"),
            call("t2", "ail.agent", "claude_code"),
            call("t2", "env", "prod"),
        ]

    def test_empty_inputs_make_no_calls(self) -> None:
        client = MagicMock()
        assert apply_trace_tags([], {"k": "v"}, client=client) == 0
        client.set_trace_tag.assert_not_called()

    def test_no_tags_makes_no_calls(self) -> None:
        client = MagicMock()
        assert apply_trace_tags(["t1"], {}, client=client) == 0
        client.set_trace_tag.assert_not_called()

    def test_values_coerced_to_str(self) -> None:
        client = MagicMock()
        apply_trace_tags(["t1"], {"version": 3}, client=client)  # type: ignore[dict-item]
        client.set_trace_tag.assert_called_once_with("t1", "version", "3")

    def test_injected_client_means_no_network(self) -> None:
        # A mocked client is used directly; no MLflow client is constructed, so
        # this never touches a workspace (the live write may be PermissionDenied).
        client = MagicMock()
        apply_trace_tags(["t1"], {"ail.cohort": "nightly"}, client=client)
        client.set_trace_tag.assert_called_once_with("t1", "ail.cohort", "nightly")
