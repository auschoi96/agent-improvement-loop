"""The ``phase2-mini`` runnable Task Suite for the live token-efficiency lever.

Unlike :mod:`ail.task_suite.seed` (the ``v1-seed`` benchmark abstracted from the
real L0 diagnosis), this is a small, **runnable** suite of five medium-difficulty
coding tasks, each backed by a live Phase-2 fixture under
``eval/phase2_fixtures/<task_id>/`` (``seed/`` + ``verify/``; see
``docs/PHASE2_LIVE_HARNESS.md``). It exists so the baseline-vs-candidate
comparison can run end-to-end on **real, file-mutating, deterministically
verifiable** tasks: the agent edits the seed, and a pristine ``pytest`` check
(``python -m pytest -q verify/``) decides PASS/FAIL per arm.

These tasks are **not** L0 trace reconstructions — they are authored, self-
contained (stdlib + pytest only, no network) coding problems whose ``seed`` ships
the gap so ``verify`` fails as-is and a correct change makes it pass. The
``category`` is a coarse label for the task's interaction profile, and
``source_trace_id`` records synthetic provenance (``phase2-fixture:<task_id>``),
not a real session.

:func:`build_phase2_mini_suite` is the single source of the suite content; the
committed ``eval/task_suite/phase2-mini/tasks.yaml`` is its frozen serialization,
and a test pins that the artifact and this builder agree (so the artifact cannot
drift from its source). The task ids match the fixture directory names so
:func:`ail.optimize.fixtures.load_fixture` resolves each one.
"""

from __future__ import annotations

from ail.task_suite.schema import Difficulty, Task, TaskCategory, TaskSuite

#: Content label of this runnable suite (distinct from the ``phase2-mini``
#: artifact *directory*; mirrors how ``v1-seed`` labels the ``v1`` artifact).
PHASE2_MINI_VERSION = "phase2-mini-v1"

#: Fixed so the frozen artifact is byte-deterministic (re-running the builder
#: yields an identical content hash).
PHASE2_MINI_CREATED_AT = "2026-06-29T00:00:00+00:00"

C = TaskCategory
D = Difficulty

_FIXTURE_TASKS: tuple[Task, ...] = (
    Task(
        task_id="ts-fix-01",
        category=C.REPEATED_TARGET_BOILERPLATE,
        difficulty=D.MEDIUM,
        source_trace_id="phase2-fixture:ts-fix-01",
        prompt=(
            "The tests in verify/ are failing. Fix the bug(s) in the shapes/ package so all "
            "tests pass. Do not modify the tests."
        ),
        notes=(
            "Synthetic live fixture eval/phase2_fixtures/ts-fix-01 (stdlib + pytest). Two "
            "bugs in two files: shapes/area.py triangle_area uses base*height instead of "
            "0.5*base*height, and shapes/registry.py mis-wires the 'triangle' entry. Seed "
            "fails verify; fixing both passes it."
        ),
    ),
    Task(
        task_id="ts-impl-02",
        category=C.TYPICAL_SHORT_SESSION,
        difficulty=D.MEDIUM,
        source_trace_id="phase2-fixture:ts-impl-02",
        prompt=(
            "Implement evaluate() in calc/evaluate.py so verify/ passes, using the existing "
            "helpers in calc/ops.py and calc/parser.py. Do not modify the tests."
        ),
        notes=(
            "Synthetic live fixture eval/phase2_fixtures/ts-impl-02 (stdlib + pytest). "
            "evaluate() is a NotImplementedError stub; a correct implementation over "
            "tokenize/precedence/apply (with * binding tighter than +/- and left-associative "
            "equal precedence) gives evaluate('2 + 3 * 4')==14 and evaluate('10 - 2 - 3')==5."
        ),
    ),
    Task(
        task_id="ts-refactor-03",
        category=C.REPEATED_TARGET_BOILERPLATE,
        difficulty=D.MEDIUM,
        source_trace_id="phase2-fixture:ts-refactor-03",
        prompt=(
            "Extract the duplicated currency-formatting logic into a shared helper common.py "
            "and use it in all three report modules. verify/ must still pass unchanged. Do not "
            "modify the tests."
        ),
        notes=(
            "Synthetic live fixture eval/phase2_fixtures/ts-refactor-03 (stdlib + pytest). The "
            "seed's behavior is already correct (render output preserved); the gap is "
            "structural — verify also asserts common.py exists and each report module imports "
            "format_currency from it, which fails until the duplication is extracted."
        ),
    ),
    Task(
        task_id="ts-config-04",
        category=C.TYPICAL_SHORT_SESSION,
        difficulty=D.MEDIUM,
        source_trace_id="phase2-fixture:ts-config-04",
        prompt=(
            "Add a max_retries setting (int, default 3) to app/config.yaml and enforce it in "
            "app/settings.py so verify/ passes. Do not modify the tests."
        ),
        notes=(
            "Synthetic live fixture eval/phase2_fixtures/ts-config-04 (stdlib + pytest; the "
            "config is parsed without PyYAML). The seed has no max_retries; a correct change "
            "adds it (default 3) and enforces it (positive int, ValueError otherwise) so "
            "load().max_retries==3 and invalid/missing inputs raise."
        ),
    ),
    Task(
        task_id="ts-route-05",
        category=C.HIGH_TOOL_CALL_VOLUME,
        difficulty=D.MEDIUM,
        source_trace_id="phase2-fixture:ts-route-05",
        prompt=(
            "Implement and register a get_user handler in api/ so verify/ passes, routing GET "
            "/users/<id> to it using the existing User model + store; return the user or a "
            "not-found result. Do not modify the tests."
        ),
        notes=(
            "Synthetic live fixture eval/phase2_fixtures/ts-route-05 (stdlib + pytest). The "
            "seed registers only GET /health; a correct change adds a get_user handler and "
            "registers GET /users/<id> so dispatch returns the User for /users/1 and a "
            "NotFound for /users/999."
        ),
    ),
)


def build_phase2_mini_suite() -> TaskSuite:
    """Construct the unfrozen ``phase2-mini`` suite from the five fixture tasks.

    Call :meth:`~ail.task_suite.schema.TaskSuite.freeze` to seal it; that frozen
    form is serialized to ``eval/task_suite/phase2-mini/tasks.yaml``.
    """
    return TaskSuite(
        version=PHASE2_MINI_VERSION,
        created_at=PHASE2_MINI_CREATED_AT,
        tasks=_FIXTURE_TASKS,
    )
