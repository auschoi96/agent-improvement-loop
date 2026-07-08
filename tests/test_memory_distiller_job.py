"""Databricks job launcher behavior for the advisory-memory distiller."""

from __future__ import annotations

import runpy

import pytest


def test_memory_distiller_job_success_returns_without_system_exit(monkeypatch) -> None:
    import ail.memory.distiller as distiller_mod

    calls: list[str] = []

    def succeed() -> int:
        calls.append("main")
        return 0

    monkeypatch.setattr(distiller_mod, "main", succeed)

    runpy.run_module("ail.jobs.memory_distiller", run_name="__main__")

    assert calls == ["main"]


def test_memory_distiller_job_nonzero_exit_still_fails(monkeypatch) -> None:
    import ail.memory.distiller as distiller_mod

    monkeypatch.setattr(distiller_mod, "main", lambda: 17)

    with pytest.raises(SystemExit) as exc:
        runpy.run_module("ail.jobs.memory_distiller", run_name="__main__")

    assert exc.value.code == 17
