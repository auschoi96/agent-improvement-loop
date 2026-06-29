"""Shared test fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def synthetic_trace() -> Any:
    """A synthetic MLflow ``Trace`` reconstructed from a recorded schema.

    The JSON matches the real ``Trace.to_dict()`` shape observed against
    experiment 660599403165942 (AGENT + LLM + two TOOL spans, one of which
    errored), but carries no real session content. Reconstructing via
    ``Trace.from_dict`` exercises the exact normalization path used for live
    traces.
    """
    from mlflow.entities import Trace

    data = json.loads((FIXTURE_DIR / "synthetic_trace.json").read_text())
    return Trace.from_dict(data)
