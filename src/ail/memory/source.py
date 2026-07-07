"""Load and format the advisory-memory source (the RLM cohort roll-up).

The memory source for the Phase A-0 spike is the RLM batch report
(``artifacts/rlm_batch_report.json``): a cohort roll-up whose ``ranked_assets``
list holds :class:`~ail.l3.contract.RankedAsset` entries — each a recommendation
(type, title) that recurred across many organic traces, carrying its
``occurrences`` count and a sample of ``expected_benefits`` / ``rationales``.

This module reads those entries, selects the highest-value few (top-k by
``occurrences``), and renders each as one concise advisory line the
:class:`~ail.memory.intervention.MemoryInjectionIntervention` injects. The
:class:`~ail.l3.contract.RankedAsset` schema is **reused** verbatim — never
redefined here.

**Fail-closed.** An **absent** report (the file does not exist) yields an empty
learnings list, which makes the intervention a no-op equal to the baseline — the
honest behaviour when there is no memory to inject. A report that *exists* but is
corrupt JSON or violates the schema raises, because that is a real error to
surface, not a silent absence.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ail.l3.contract import RankedAsset

__all__ = [
    "DEFAULT_REPORT_PATH",
    "DEFAULT_TOP_K",
    "load_ranked_assets",
    "select_top_k",
    "format_advisory_line",
    "build_memory_learnings",
    "load_memory_learnings",
]

#: Default location of the RLM batch report (relative to the working directory).
DEFAULT_REPORT_PATH = Path("artifacts/rlm_batch_report.json")

#: Default number of learnings to inject — the highest-value few, not the tail.
DEFAULT_TOP_K = 8

#: The key under which the report holds its ranked-asset roll-up.
_RANKED_ASSETS_KEY = "ranked_assets"

#: Cap on an advisory line's detail so the injected block stays concise even when
#: a source rationale runs long.
_MAX_DETAIL_CHARS = 240


def load_ranked_assets(
    path: str | Path | None = None,
    *,
    parsed: Sequence[dict[str, Any]] | None = None,
) -> list[RankedAsset]:
    """Read :class:`~ail.l3.contract.RankedAsset` entries from the RLM report.

    Args:
        path: Path to the RLM batch report JSON. Defaults to
            :data:`DEFAULT_REPORT_PATH`. An **absent** file yields ``[]``
            (fail-closed: no memory source ⇒ no learnings ⇒ no-op == baseline).
        parsed: A pre-parsed list of ranked-asset mappings, bypassing the file
            read — the seam unit tests drive so no fixture file is needed.

    Returns:
        The parsed ranked assets (empty when the source is absent).

    Raises:
        json.JSONDecodeError: if an existing report is not valid JSON.
        pydantic.ValidationError: if an entry does not match the schema.
    """
    if parsed is not None:
        return [RankedAsset.model_validate(entry) for entry in parsed]
    report_path = Path(path) if path is not None else DEFAULT_REPORT_PATH
    if not report_path.is_file():
        return []
    data = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return []
    raw = data.get(_RANKED_ASSETS_KEY)
    if not isinstance(raw, list):
        return []
    return [RankedAsset.model_validate(entry) for entry in raw]


def select_top_k(assets: Sequence[RankedAsset], k: int = DEFAULT_TOP_K) -> list[RankedAsset]:
    """The ``k`` highest-value assets, ranked by ``occurrences`` (desc).

    An asset that recurs across the most traces is the highest-value learning to
    carry, so ``occurrences`` is the primary key; ties break on the report's own
    ``rank`` then ``title`` for a deterministic, reproducible selection. ``k <= 0``
    selects none.
    """
    if k <= 0:
        return []
    ordered = sorted(assets, key=lambda a: (-a.occurrences, a.rank, a.title))
    return ordered[:k]


def format_advisory_line(asset: RankedAsset) -> str:
    """Render one asset as a concise advisory line: ``- <title> — <benefit>``.

    Prefers the first ``expected_benefit`` (a concise, outcome-framed sentence)
    and falls back to the first ``rationale``; whitespace is collapsed and a long
    detail is truncated so the injected block stays compact.
    """
    detail = ""
    if asset.expected_benefits:
        detail = asset.expected_benefits[0]
    elif asset.rationales:
        detail = asset.rationales[0]
    detail = " ".join(detail.split())
    if len(detail) > _MAX_DETAIL_CHARS:
        detail = detail[: _MAX_DETAIL_CHARS - 1].rstrip() + "…"
    line = f"- {asset.title.strip()}"
    if detail:
        line += f" — {detail}"
    return line


def build_memory_learnings(
    assets: Sequence[RankedAsset], *, k: int = DEFAULT_TOP_K
) -> tuple[str, ...]:
    """Select the top-``k`` assets and render each as an advisory line."""
    return tuple(format_advisory_line(a) for a in select_top_k(assets, k))


def load_memory_learnings(
    path: str | Path | None = None,
    *,
    parsed: Sequence[dict[str, Any]] | None = None,
    k: int = DEFAULT_TOP_K,
) -> tuple[str, ...]:
    """Load the RLM report and build the top-``k`` advisory learnings from it.

    A one-call convenience over :func:`load_ranked_assets` +
    :func:`build_memory_learnings`. An absent source yields an empty tuple (the
    fail-closed no-op == baseline).
    """
    return build_memory_learnings(load_ranked_assets(path, parsed=parsed), k=k)
