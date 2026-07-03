"""The suite builder (L8): scaffold a draft from traces, then freeze fail-closed.

These pin the two guarantees that make the on-ramp safe. Scaffold (a) drafts N
representative tasks from a corpus, stratified across the L0 distribution, and (b)
**never fabricates a success check** — every check is the required placeholder a
human must replace. Freeze (c) **fails closed** on any remaining placeholder (naming
the tasks, freezing nothing, exiting non-zero), and only when every check is authored
does it (d) seal the suite and (e) confirm it reloads with a passing integrity check;
(f) tampering a frozen task then trips that integrity check.

No live calls: the trace source is a fake injected into ``scaffold`` — the same seam
``test_readiness_preflight`` uses.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ail.ingest.base import NormalizedTrace, TokenUsage, ToolCall
from ail.jobs.suite_freeze import (
    SuiteFreezeError,
    find_unauthored,
    freeze_suite,
    read_draft,
)
from ail.jobs.suite_freeze import (
    main as freeze_main,
)
from ail.jobs.suite_scaffold import (
    CHECKS_FILENAME,
    DEFAULT_COUNT,
    PLACEHOLDER_CHECK,
    build_draft_suite,
    load_checks,
    root_version_from_tasks_path,
    scaffold,
)
from ail.task_suite import TaskSuiteIntegrityError, load_task_suite

FIXED_CREATED_AT = "2026-07-03T00:00:00+00:00"


class _FakeSource:
    """A ``TraceSource``-shaped stub returning canned traces (no MLflow)."""

    def __init__(self, traces: list[NormalizedTrace]) -> None:
        self._traces = traces

    def fetch_traces(
        self, *, experiment_id: str, max_results: int | None = None, **_: object
    ) -> list[NormalizedTrace]:
        return list(self._traces if max_results is None else self._traces[:max_results])


def _trace(i: int, tokens: int, n_tools: int, preview: str | None) -> NormalizedTrace:
    calls = [
        ToolCall(id=f"{i}-{j}", name="Bash", arguments={"command": f"cd /repo{i} && run"})
        for j in range(n_tools)
    ]
    return NormalizedTrace(
        trace_id=f"trace-{i:02d}",
        token_usage=TokenUsage(input_tokens=tokens),
        tool_calls=calls,
        request_preview=preview,
    )


def _corpus() -> list[NormalizedTrace]:
    """20 traces spanning 1k..20k tokens; even ids carry a request preview, odd don't."""
    return [
        _trace(
            i,
            tokens=1000 * i,
            n_tools=(i % 5) + 1,
            preview=(f"Task {i}: do it" if i % 2 == 0 else None),
        )
        for i in range(1, 21)
    ]


def _out_path(root: Path, version: str = "draft") -> Path:
    return root / "eval" / "task_suite" / version / "tasks.yaml"


def _author_all_checks(tasks_path: Path) -> None:
    """Overwrite checks.yaml with a real (non-placeholder) check per drafted task."""
    suite = read_draft(tasks_path)
    checks = {t.task_id: f"assert the harness exits 0 for {t.task_id}" for t in suite.tasks}
    (tasks_path.parent / CHECKS_FILENAME).write_text(
        yaml.safe_dump({"checks": checks}), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# scaffold
# ---------------------------------------------------------------------------


def test_scaffold_produces_n_draft_tasks_with_placeholder_checks(tmp_path: Path) -> None:
    """(1) N draft tasks, frozen=False; (2) every check is the placeholder, none fabricated."""
    path = scaffold(
        "exp",
        out=_out_path(tmp_path),
        count=8,
        source=_FakeSource(_corpus()),
        created_at=FIXED_CREATED_AT,
    )
    assert path.name == "tasks.yaml" and path.is_file()

    suite = read_draft(path)
    assert suite.frozen is False
    assert suite.content_hash == ""
    assert len(suite.tasks) == 8

    checks = load_checks(path)
    assert len(checks) == 8
    # (2) scaffold NEVER fabricates a real check — every one is the placeholder,
    # and the placeholder never leaks into a task's prompt or notes.
    assert set(checks.values()) == {PLACEHOLDER_CHECK}
    for task in suite.tasks:
        assert PLACEHOLDER_CHECK not in task.prompt
        assert PLACEHOLDER_CHECK not in task.notes


def test_scaffold_stratifies_across_the_distribution_not_top_n(tmp_path: Path) -> None:
    """The sample spans the token distribution — not just the N largest sessions."""
    path = scaffold(
        "exp",
        out=_out_path(tmp_path),
        count=8,
        source=_FakeSource(_corpus()),
        created_at=FIXED_CREATED_AT,
    )
    picked = {t.source_trace_id for t in read_draft(path).tasks}
    top_8 = {f"trace-{i:02d}" for i in range(13, 21)}  # the 8 largest by token count
    assert picked != top_8
    # a below-median trace is represented (top-N would exclude all of these)
    assert any(int(tid.split("-")[1]) <= 10 for tid in picked)


def test_build_draft_suite_is_pure_and_deterministic() -> None:
    """The pure builder fabricates no check and freezes reproducibly (stable hash)."""
    suite, checks = build_draft_suite(
        _corpus(), count=5, version="v-test", created_at=FIXED_CREATED_AT
    )
    assert suite.frozen is False and suite.content_hash == ""
    assert len(suite.tasks) == 5 == len(checks)
    assert set(checks.values()) == {PLACEHOLDER_CHECK}

    again, _ = build_draft_suite(_corpus(), count=5, version="v-test", created_at=FIXED_CREATED_AT)
    assert suite.freeze().content_hash == again.freeze().content_hash


def test_scaffold_defaults_to_eight_tasks(tmp_path: Path) -> None:
    path = scaffold(
        "exp", out=_out_path(tmp_path), source=_FakeSource(_corpus()), created_at=FIXED_CREATED_AT
    )
    assert len(read_draft(path).tasks) == DEFAULT_COUNT


# ---------------------------------------------------------------------------
# freeze — fail-closed
# ---------------------------------------------------------------------------


def test_freeze_refuses_when_any_check_is_placeholder(tmp_path: Path) -> None:
    """(3) fail-closed: names the offending tasks, freezes nothing, exits non-zero."""
    path = scaffold(
        "exp",
        out=_out_path(tmp_path),
        count=4,
        source=_FakeSource(_corpus()),
        created_at=FIXED_CREATED_AT,
    )

    with pytest.raises(SuiteFreezeError) as excinfo:
        freeze_suite(path)
    message = str(excinfo.value)
    assert "ts-draft-001" in message  # names an offending task id
    assert "REFUSING TO FREEZE" in message

    # froze NOTHING: still an unfrozen draft, and therefore not loadable via the
    # sealed loader (which refuses an unfrozen artifact).
    assert read_draft(path).frozen is False
    root, version = root_version_from_tasks_path(path)
    with pytest.raises(TaskSuiteIntegrityError):
        load_task_suite(version, root=root)

    # the CLI surfaces the refusal as a non-zero exit code.
    assert freeze_main([str(path)]) == 2


def test_freeze_refuses_when_even_one_check_is_unauthored(tmp_path: Path) -> None:
    """A single remaining placeholder is enough to refuse (and only it is named)."""
    path = scaffold(
        "exp",
        out=_out_path(tmp_path),
        count=4,
        source=_FakeSource(_corpus()),
        created_at=FIXED_CREATED_AT,
    )
    suite = read_draft(path)
    # author every check except the last one
    authored = {t.task_id: f"real check {t.task_id}" for t in suite.tasks}
    last_id = suite.tasks[-1].task_id
    authored[last_id] = PLACEHOLDER_CHECK
    (path.parent / CHECKS_FILENAME).write_text(
        yaml.safe_dump({"checks": authored}), encoding="utf-8"
    )

    assert find_unauthored(suite, authored) == [last_id]
    with pytest.raises(SuiteFreezeError) as excinfo:
        freeze_suite(path)
    assert last_id in str(excinfo.value)


def test_freeze_refuses_when_checks_file_missing(tmp_path: Path) -> None:
    """No checks.yaml at all => every task is unauthored => fail-closed."""
    path = scaffold(
        "exp",
        out=_out_path(tmp_path),
        count=3,
        source=_FakeSource(_corpus()),
        created_at=FIXED_CREATED_AT,
    )
    (path.parent / CHECKS_FILENAME).unlink()
    with pytest.raises(SuiteFreezeError):
        freeze_suite(path)


# ---------------------------------------------------------------------------
# freeze — success + integrity
# ---------------------------------------------------------------------------


def test_freeze_succeeds_and_seals_when_all_authored(tmp_path: Path) -> None:
    """(4) all checks authored => frozen=True + content_hash; (5) reloads + passes integrity."""
    path = scaffold(
        "exp",
        out=_out_path(tmp_path),
        count=6,
        source=_FakeSource(_corpus()),
        created_at=FIXED_CREATED_AT,
    )
    _author_all_checks(path)

    frozen = freeze_suite(path)
    assert frozen.frozen is True
    assert frozen.content_hash  # non-empty seal

    # (5) the sealed artifact reloads through the existing loader and passes integrity.
    root, version = root_version_from_tasks_path(path)
    reloaded = load_task_suite(version, root=root)
    assert reloaded.frozen is True
    assert reloaded.content_hash == frozen.content_hash

    assert freeze_main([str(path)]) == 0  # idempotent success on an already-frozen suite


def test_tampering_a_frozen_task_trips_integrity(tmp_path: Path) -> None:
    """(6) editing a frozen task's content => TaskSuiteIntegrityError on reload."""
    path = scaffold(
        "exp",
        out=_out_path(tmp_path),
        count=5,
        source=_FakeSource(_corpus()),
        created_at=FIXED_CREATED_AT,
    )
    _author_all_checks(path)
    freeze_suite(path)

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["frozen"] is True
    data["tasks"][0]["prompt"] = data["tasks"][0]["prompt"] + "  (tampered)"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    root, version = root_version_from_tasks_path(path)
    with pytest.raises(TaskSuiteIntegrityError):
        load_task_suite(version, root=root)
