"""The runnable ``phase2-mini`` suite + its five live fixtures, end-to-end.

Fully offline — **no live agent, no network**. A ``FixtureSolvingAdapter`` *actually
edits files* in ``task.cwd`` (the per-arm workspace the harness seeds from each
fixture's ``seed/``), applying the intended fix for whichever fixture it detects.
Verification is the real, committed ``python -m pytest -q verify/`` command from
``run_plan.yaml`` (run under the test interpreter), restored from the pristine
``verify/`` — so the live PROMOTE/BLOCK path runs for real against the actual
fixtures and run plan.

Coverage (the task contract):

* **each fixture resolves** — ``load_fixture`` finds ``seed/`` + ``verify/`` for all five;
* **frozen suite integrity** — ``load_task_suite("phase2-mini")`` loads frozen and
  re-hashes to its source builder (no drift), and every task has a fixture + run-plan entry;
* **end-to-end** — an adapter that applies the correct edit and is cheaper PROMOTEs
  every task; one that breaks the candidate REGRESSES (BLOCK); one that does nothing
  fails closed on the baseline anchor (BLOCK). Realized savings count PROMOTE only.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from ail.compare import Recommendation
from ail.ingest.base import (
    AgentAdapter,
    AgentRunResult,
    AgentTask,
    NormalizedTrace,
    TokenUsage,
    TraceStatus,
)
from ail.optimize import L1Outcome, VerifySpec, load_fixture, run_phase2_comparison
from ail.task_suite import build_phase2_mini_suite, load_task_suite

_STAMP = "2026-06-29T00:00:00+00:00"
_SUITE = "phase2-mini"
_REPO_ROOT = Path(__file__).resolve().parents[1]

FIXTURE_TASK_IDS = (
    "ts-fix-01",
    "ts-impl-02",
    "ts-refactor-03",
    "ts-config-04",
    "ts-route-05",
)

# --------------------------------------------------------------------------- #
# Reference solutions: the intended fix per fixture, as {workspace-rel: content}.
# Wrapped in single-quoted triple strings because the file contents themselves
# contain double-quoted docstrings. These are what a correct agent run produces;
# the e2e PROMOTE test applies them and asserts verify passes.
# --------------------------------------------------------------------------- #

_TS_FIX_01_AREA = '''\
"""Area formulas for a few basic shapes."""

import math


def circle_area(radius):
    return math.pi * radius * radius


def rectangle_area(width, height):
    return width * height


def triangle_area(base, height):
    return 0.5 * base * height
'''

_TS_FIX_01_REGISTRY = '''\
"""Dispatch a shape name to its area function."""

from shapes.area import circle_area, rectangle_area, triangle_area

AREA_FUNCS = {
    "circle": circle_area,
    "rectangle": rectangle_area,
    "triangle": triangle_area,
}


def area(name, *args):
    """Compute the area of ``name`` given its dimensions."""
    return AREA_FUNCS[name](*args)
'''

_TS_IMPL_02_EVALUATE = '''\
"""Evaluate an arithmetic expression over the calc helpers."""

from calc.ops import apply, precedence
from calc.parser import tokenize

__all__ = ["evaluate"]


def evaluate(expression):
    """Evaluate ``expression`` (integers with +, -, *) to an int."""
    tokens = tokenize(expression)
    values = []
    operators = []

    def reduce_once():
        operator = operators.pop()
        right = values.pop()
        left = values.pop()
        values.append(apply(operator, left, right))

    for token in tokens:
        if isinstance(token, int):
            values.append(token)
        else:
            while operators and precedence(operators[-1]) >= precedence(token):
                reduce_once()
            operators.append(token)
    while operators:
        reduce_once()
    return values[0]
'''

_TS_REFACTOR_03_COMMON = '''\
"""Shared report helpers."""


def format_currency(amount):
    return f"${amount:,.2f}"
'''


def _report_module(docstring: str) -> str:
    return (
        f'"""{docstring}"""\n\n'
        "from common import format_currency\n\n\n"
        "def render(label, amount):\n"
        '    return f"{label}: {format_currency(amount)}"\n'
    )


_TS_CONFIG_04_YAML = (
    "# Application configuration.\napp_name: governed-demo\ntimeout_seconds: 30\nmax_retries: 3\n"
)

_TS_CONFIG_04_SETTINGS = '''\
"""Load and validate the application configuration."""

from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"

REQUIRED_KEYS = ("app_name", "timeout_seconds")


@dataclass(frozen=True)
class Settings:
    app_name: str
    timeout_seconds: int
    max_retries: int = 3


def _parse_config(text):
    config = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, sep, value = stripped.partition(":")
        if not sep:
            raise ValueError(f"malformed config line: {line!r}")
        key = key.strip()
        value = value.strip()
        if value.lstrip("-").isdigit():
            config[key] = int(value)
        else:
            config[key] = value
    return config


def validate(raw):
    missing = [key for key in REQUIRED_KEYS if key not in raw]
    if missing:
        raise ValueError(f"missing required config key(s): {', '.join(missing)}")
    max_retries = raw.get("max_retries", 3)
    if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 1:
        raise ValueError(f"max_retries must be a positive integer, got {max_retries!r}")
    return Settings(
        app_name=raw["app_name"],
        timeout_seconds=raw["timeout_seconds"],
        max_retries=max_retries,
    )


def load(path=DEFAULT_CONFIG_PATH):
    raw = _parse_config(Path(path).read_text(encoding="utf-8"))
    return validate(raw)
'''

_TS_ROUTE_05_HANDLERS = '''\
"""Request handlers."""

from api.models import NotFound, get


def index():
    """Health/index handler: a minimal status response."""
    return {"status": "ok"}


def get_user(id):
    """Return the requested user, or a NotFound result."""
    try:
        user_id = int(id)
    except (TypeError, ValueError):
        return NotFound(resource=f"user:{id}")
    user = get(user_id)
    if user is None:
        return NotFound(resource=f"user:{user_id}")
    return user
'''

_TS_ROUTE_05_INIT = '''\
"""Wire the handlers into a router via build_router()."""

from api.handlers import get_user, index
from api.router import Router

__all__ = ["build_router"]


def build_router():
    """Construct the application router with all routes registered."""
    router = Router()
    router.register("GET", "/health", index)
    router.register("GET", "/users/<id>", get_user)
    return router
'''

FIXTURE_SOLUTIONS: dict[str, dict[str, str]] = {
    "ts-fix-01": {
        "shapes/area.py": _TS_FIX_01_AREA,
        "shapes/registry.py": _TS_FIX_01_REGISTRY,
    },
    "ts-impl-02": {
        "calc/evaluate.py": _TS_IMPL_02_EVALUATE,
    },
    "ts-refactor-03": {
        "common.py": _TS_REFACTOR_03_COMMON,
        "report_a.py": _report_module("Revenue report."),
        "report_b.py": _report_module("Cost report."),
        "report_c.py": _report_module("Net report."),
    },
    "ts-config-04": {
        "app/config.yaml": _TS_CONFIG_04_YAML,
        "app/settings.py": _TS_CONFIG_04_SETTINGS,
    },
    "ts-route-05": {
        "api/handlers.py": _TS_ROUTE_05_HANDLERS,
        "api/__init__.py": _TS_ROUTE_05_INIT,
    },
}

# A workspace-root marker unique to each fixture's seed, so the adapter can tell
# which fixture it is running in (it only sees the seeded ``task.cwd``).
_MARKERS = {
    "shapes": "ts-fix-01",
    "calc": "ts-impl-02",
    "report_a.py": "ts-refactor-03",
    "app": "ts-config-04",
    "api": "ts-route-05",
}


def _detect_task_id(cwd: Path) -> str:
    for marker, task_id in _MARKERS.items():
        if (cwd / marker).exists():
            return task_id
    raise AssertionError(f"no known fixture marker under {cwd}")


def _apply_solution(cwd: Path) -> None:
    task_id = _detect_task_id(cwd)
    for rel, content in FIXTURE_SOLUTIONS[task_id].items():
        target = cwd / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #


@dataclass
class ArmPlan:
    """What the fake agent does for one arm: solve (or not) + token/exec outcome."""

    solve: bool = True
    tokens: int = 0
    success: bool = True
    output: str = "done"


class FixtureSolvingAdapter(AgentAdapter):
    """A FAKE adapter that applies the correct fix to ``task.cwd`` (no live agent).

    Tells the candidate arm from the baseline by the skill marker the CANDIDATE
    intervention injects into the system prompt, then — if that arm's plan says so
    — applies the detected fixture's reference solution to the per-arm workspace.
    """

    name = "fixture-solver"

    def __init__(self, *, baseline: ArmPlan, candidate: ArmPlan) -> None:
        self.baseline = baseline
        self.candidate = candidate
        self.seen: list[tuple[str | None, bool]] = []

    def run(self, task: AgentTask) -> AgentRunResult:
        is_candidate = "<skill" in (task.system_prompt or "")
        plan = self.candidate if is_candidate else self.baseline
        self.seen.append((task.cwd, is_candidate))
        cwd = Path(task.cwd) if task.cwd else Path.cwd()
        if plan.solve:
            _apply_solution(cwd)
        trace = NormalizedTrace(
            trace_id=("cand" if is_candidate else "base"),
            status=TraceStatus.OK if plan.success else TraceStatus.ERROR,
            producer=self.name,
            model="claude-opus-4-8",
            token_usage=TokenUsage(input_tokens=plan.tokens),
        )
        return AgentRunResult(
            trace=trace,
            output_text=plan.output,
            success=plan.success,
            error=None if plan.success else "arm failed",
        )


def _specs_from_run_plan() -> dict[str, VerifySpec]:
    """Build verify specs from the committed ``run_plan.yaml``.

    Validates the actual committed run plan, then swaps the leading ``python`` for
    the current interpreter so the check runs hermetically under the test venv
    (which has pytest), independent of what ``python`` resolves to on PATH.
    """
    raw = yaml.safe_load((_REPO_ROOT / "run_plan.yaml").read_text(encoding="utf-8"))
    specs: dict[str, VerifySpec] = {}
    for task_id, entry in raw.items():
        command = list(entry["command"])
        if command and command[0] == "python":
            command[0] = sys.executable
        specs[task_id] = VerifySpec(
            name=str(entry.get("name", f"verify-{task_id}")),
            command=command,
            timeout_seconds=int(entry.get("timeout_seconds", 600)),
        )
    return specs


# --------------------------------------------------------------------------- #
# (a) Each fixture resolves
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("task_id", FIXTURE_TASK_IDS)
def test_each_fixture_resolves(task_id: str) -> None:
    fixture = load_fixture(task_id, root=_REPO_ROOT)
    assert fixture is not None
    assert fixture.seed_dir.is_dir()
    assert fixture.has_verify


# --------------------------------------------------------------------------- #
# (b) Frozen suite loads, verifies its hash, and is wired to fixtures + run plan
# --------------------------------------------------------------------------- #


def test_frozen_mini_suite_loads_and_verifies_hash() -> None:
    suite = load_task_suite(_SUITE, root=_REPO_ROOT)
    assert suite.frozen
    assert suite.version == "phase2-mini-v1"
    assert sorted(suite.task_ids()) == sorted(FIXTURE_TASK_IDS)
    # The on-disk artifact re-hashes to its source builder: no drift, and the
    # loader's frozen-integrity check passed (it would have raised otherwise).
    assert suite.content_hash == build_phase2_mini_suite().freeze().content_hash


def test_every_suite_task_has_a_fixture_and_run_plan_entry() -> None:
    suite = load_task_suite(_SUITE, root=_REPO_ROOT)
    specs = _specs_from_run_plan()
    assert set(specs) == set(FIXTURE_TASK_IDS)
    for task in suite.tasks:
        assert load_fixture(task.task_id, root=_REPO_ROOT) is not None
        assert task.task_id in specs


# --------------------------------------------------------------------------- #
# (c) End-to-end through run_phase2_comparison over the whole mini-suite
# --------------------------------------------------------------------------- #


def _run(adapter: FixtureSolvingAdapter):
    return run_phase2_comparison(
        suite=load_task_suite(_SUITE, root=_REPO_ROOT),
        adapter=adapter,
        verify_specs=_specs_from_run_plan(),
        fixtures_root=_REPO_ROOT,
        generated_at=_STAMP,
    )


def test_mini_suite_promotes_when_candidate_solves_cheaper() -> None:
    # Both arms apply the correct fix (verify passes for each); the candidate is
    # cheaper -> every task PROMOTEs and the realized savings sum over them.
    artifact = _run(
        FixtureSolvingAdapter(
            baseline=ArmPlan(solve=True, tokens=100_000),
            candidate=ArmPlan(solve=True, tokens=60_000),
        )
    )
    assert artifact.n_tasks == 5
    assert artifact.n_promote == 5
    assert artifact.n_block == 0
    assert all(o.recommendation is Recommendation.PROMOTE for o in artifact.outcomes)
    assert all(o.l1_outcome is L1Outcome.PASSED for o in artifact.outcomes)
    assert all(o.baseline_succeeded and o.candidate_succeeded for o in artifact.outcomes)
    assert artifact.realized_token_savings_absolute == 200_000.0  # 5 * (100k - 60k)


def test_mini_suite_blocks_when_candidate_regresses() -> None:
    # Baseline solves (verify passes), candidate does nothing (verify fails) even
    # though it is cheaper -> a correctness regression -> BLOCK, no realized win.
    artifact = _run(
        FixtureSolvingAdapter(
            baseline=ArmPlan(solve=True, tokens=100_000),
            candidate=ArmPlan(solve=False, tokens=60_000),
        )
    )
    assert artifact.n_promote == 0
    assert artifact.n_block == 5
    assert all(o.recommendation is Recommendation.BLOCK for o in artifact.outcomes)
    assert all(o.l1_outcome is L1Outcome.REGRESSED for o in artifact.outcomes)
    assert all(o.token_improved for o in artifact.outcomes)  # cheaper, but blocked
    assert artifact.realized_token_savings_absolute == 0.0


def test_mini_suite_blocks_when_nothing_changes() -> None:
    # Neither arm edits anything: the baseline never passes verify, so it is not a
    # valid anchor -> the comparison fails closed (BLOCK) for every task.
    artifact = _run(
        FixtureSolvingAdapter(
            baseline=ArmPlan(solve=False, tokens=100_000),
            candidate=ArmPlan(solve=False, tokens=60_000),
        )
    )
    assert artifact.n_promote == 0
    assert artifact.n_block == 5
    assert all(o.recommendation is Recommendation.BLOCK for o in artifact.outcomes)
    assert all(o.l1_outcome is L1Outcome.NO_VERDICT for o in artifact.outcomes)
    assert artifact.realized_token_savings_absolute == 0.0
