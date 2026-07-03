"""``ail-suite-freeze`` — seal a drafted Task Suite, fail-closed on unauthored checks.

The freeze half of the suite-builder on-ramp (:mod:`ail.jobs.suite_scaffold` is the
draft half). It takes the path to a draft ``tasks.yaml``, and — this is the whole
point — **refuses to freeze unless every task has a real, human-authored success
check.** A check that is still the scaffold placeholder (``PLACEHOLDER_CHECK``) or
empty means a human has not done the authoring yet; freezing then would seal a
benchmark that verifies nothing. So this fails closed, names the offending task ids,
freezes nothing, and exits non-zero. That refusal is the guard that forces authoring
— the same discipline the ground-truth path uses to refuse a fabricated label.

When every check is authored it seals the suite by reusing the existing machinery —
:meth:`ail.task_suite.schema.TaskSuite.freeze` (sets ``frozen=True`` and the
``content_hash``), :func:`ail.task_suite.save_task_suite`, then
:func:`ail.task_suite.load_task_suite` to confirm the sealed artifact reloads and
passes its integrity check. Checks live out-of-band in the companion ``checks.yaml``
(keyed by ``task_id``, as ``phase2-mini`` keys its ``verify/`` checks), because the
:class:`~ail.task_suite.schema.Task` schema carries no success-check field.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from ail.jobs.suite_scaffold import (
    CHECKS_FILENAME,
    PLACEHOLDER_CHECK,
    load_checks,
    root_version_from_tasks_path,
)
from ail.task_suite import TaskSuite, load_task_suite, save_task_suite

__all__ = [
    "SuiteFreezeError",
    "is_unauthored",
    "find_unauthored",
    "read_draft",
    "freeze_suite",
    "main",
]


class SuiteFreezeError(RuntimeError):
    """Freeze was refused: the draft is empty or a task still lacks an authored check.

    Fail-closed. Carries a message naming the offending task ids; :func:`main` turns
    it into a non-zero exit and freezes nothing.
    """


def is_unauthored(check: str | None) -> bool:
    """Whether a success check is still unauthored (empty or the scaffold placeholder)."""
    if check is None:
        return True
    stripped = check.strip()
    return not stripped or stripped == PLACEHOLDER_CHECK


def find_unauthored(suite: TaskSuite, checks: dict[str, str]) -> list[str]:
    """Task ids (in suite order) whose companion check is missing/empty/placeholder."""
    return [t.task_id for t in suite.tasks if is_unauthored(checks.get(t.task_id))]


def read_draft(tasks_path: str | Path) -> TaskSuite:
    """Validate and return the draft suite from a ``tasks.yaml``.

    Reads the raw YAML and validates it through the schema directly — **not** via
    :func:`ail.task_suite.load_task_suite`, which by design refuses an unfrozen
    artifact (the whole point of a draft). Schema invariants (unique ids, correct
    pool) still apply.
    """
    path = Path(tasks_path)
    if not path.is_file():
        raise FileNotFoundError(f"no draft Task Suite at {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return TaskSuite.model_validate(data)


def _refuse_message(tasks_path: Path, unauthored: list[str]) -> str:
    ids = "\n".join(f"    - {task_id}" for task_id in unauthored)
    return (
        f"REFUSING TO FREEZE {tasks_path}: {len(unauthored)} task(s) still have an unauthored "
        f"success check (empty or the scaffold placeholder). Author a real deterministic check "
        f"for each of these task ids in {CHECKS_FILENAME}, then re-run:\n"
        f"{ids}\n"
        "Nothing was frozen — a frozen suite must verify something."
    )


def freeze_suite(tasks_path: str | Path) -> TaskSuite:
    """Freeze a drafted suite once every check is authored; return the reloaded frozen suite.

    Fail-closed: raises :class:`SuiteFreezeError` (naming the task ids, freezing
    nothing) if the draft is empty or any check is still unauthored. Otherwise seals
    via :meth:`TaskSuite.freeze`, saves over the draft, and reloads through
    :func:`ail.task_suite.load_task_suite` to confirm the integrity check passes.
    """
    path = Path(tasks_path)
    suite = read_draft(path)
    if not suite.tasks:
        raise SuiteFreezeError(f"REFUSING TO FREEZE {path}: the draft has no tasks.")

    unauthored = find_unauthored(suite, load_checks(path))
    if unauthored:
        raise SuiteFreezeError(_refuse_message(path, unauthored))

    root, artifact_version = root_version_from_tasks_path(path)
    save_task_suite(suite.freeze(), artifact_version, root=root, overwrite=True)
    # Reload through the sealed loader: proves the on-disk artifact is frozen AND
    # that its stored content_hash matches its tasks (else TaskSuiteIntegrityError).
    return load_task_suite(artifact_version, root=root)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ail-suite-freeze",
        description=(
            "Freeze a drafted Task Suite. Refuses (non-zero, names the tasks, freezes nothing) "
            "if any task's success check in checks.yaml is still empty or the scaffold "
            "placeholder. Otherwise seals it (frozen=True + content_hash) and confirms it "
            "reloads with a passing integrity check."
        ),
    )
    parser.add_argument("path", help="Path to the draft tasks.yaml to freeze.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        frozen = freeze_suite(args.path)
    except SuiteFreezeError as exc:
        print(f"[ail-suite-freeze] {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"[ail-suite-freeze] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - schema/integrity failure: surface, don't crash
        print(f"[ail-suite-freeze] could not freeze: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(
        f"[ail-suite-freeze] FROZEN {len(frozen)} task(s): {args.path}\n"
        f"  version={frozen.version!r}  content_hash={frozen.content_hash}\n"
        "  reloaded and passed the integrity check."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
