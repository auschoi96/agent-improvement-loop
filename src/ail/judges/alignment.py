"""MemAlign alignment: align a judge from labeled Alignment-Set traces.

A thin wrapper over the public MLflow GenAI API ``judge.align(traces=...,
optimizer=...)`` with the OSS **MemAlign** optimizer
(:class:`mlflow.genai.judges.optimizers.MemAlignOptimizer`). MemAlign learns
from the human assessments carried on a set of traces — distilling guidelines
(semantic memory) and retrieving similar past examples (episodic memory) — and
returns a better-aligned judge.

Two guarantees this wrapper enforces, both from ``docs/ARCHITECTURE.md`` §2/§4:

* **Alignment input is strictly the Alignment Set.** :func:`align_judge` takes
  an :class:`~ail.judges.pools.AlignmentSet` and nothing else, so the pool that
  trains the judge can never be the Task Suite (which compares agents) or the
  Human Anchor (which audits the judge). Mixing them is what lets the agent and
  judge co-adapt.
* **Alignment runs on its own cadence, decoupled from agent optimization.**
  This module has no dependency on, and no call into, the optimizer that tunes
  the agent (GEPA / the loop controller). Aligning a judge and optimizing an
  agent are separate operations on separate pools; keeping them in separate
  modules with separate inputs is the structural half of "decoupled cadence",
  and the prose above is the documented half.

Dependency note: MemAlign requires the ``dspy`` extra and a live judge/embedding
model. To keep this package importable (and CI green) without those, the
MemAlign optimizer is imported **lazily** inside
:func:`build_memalign_optimizer`; :func:`align_judge` itself touches neither
``dspy`` nor a model at import time. Offline tests mock ``judge.align``; a
genuine alignment is gated behind ``@pytest.mark.live``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from ail.judges.contract import AlignmentReport
from ail.judges.pools import AlignmentSet

if TYPE_CHECKING:
    from mlflow.genai.judges import Judge
    from mlflow.genai.judges.base import AlignmentOptimizer

__all__ = [
    "MemAlignConfig",
    "AlignmentOutcome",
    "build_memalign_optimizer",
    "align_judge",
]


@dataclass(frozen=True, slots=True)
class MemAlignConfig:
    """Configuration for a MemAlign optimizer.

    Mirrors the public ``MemAlignOptimizer`` constructor. ``None`` values defer
    to MLflow's defaults (the default judge model for the active tracking
    backend, and the default embedding model).

    Args:
        reflection_lm: Model URI used to distill guidelines from feedback (e.g.
            ``"databricks:/..."`` or ``"openai:/gpt-4o-mini"``).
        retrieval_k: Number of similar past examples retrieved from episodic
            memory during evaluation.
        embedding_model: Model URI used to embed feedback examples.
        embedding_dim: Embedding dimension.
    """

    reflection_lm: str | None = None
    retrieval_k: int = 5
    embedding_model: str | None = None
    embedding_dim: int = 512


@dataclass(frozen=True, slots=True)
class AlignmentOutcome:
    """The result of an alignment cadence: the aligned judge plus its record.

    ``judge`` is the live, better-aligned MLflow ``Judge`` (ready to score or to
    re-align); ``report`` is the serializable :class:`AlignmentReport` for
    logging/audit.
    """

    judge: Judge
    report: AlignmentReport


def build_memalign_optimizer(config: MemAlignConfig | None = None) -> AlignmentOptimizer:
    """Construct a configured MemAlign optimizer.

    Imported lazily because ``MemAlignOptimizer`` pulls in ``dspy`` (an optional
    heavy dependency) at construction time. Pass the result as ``optimizer`` to
    :func:`align_judge` to override MLflow's default MemAlign settings.

    Raises:
        ImportError: If the optional ``dspy`` dependency is not installed —
            re-raised with guidance (preserving the original cause) rather than
            swallowed, since the caller explicitly asked for a configured
            optimizer.
    """
    cfg = config or MemAlignConfig()
    # MemAlign pulls in dspy; with it absent MLflow raises either ImportError or
    # an MlflowException ("DSPy library is required but not installed"). Either
    # way the cause is the missing optional dependency, so surface a single,
    # actionable ImportError while preserving the original via ``from exc``.
    try:
        from mlflow.genai.judges.optimizers import MemAlignOptimizer
    except Exception as exc:  # noqa: BLE001 - dspy-missing is ImportError or MlflowException
        raise ImportError(
            "MemAlign requires the optional 'dspy' optimizer dependency and could not be "
            f"constructed: {exc}. Install it (e.g. pip install dspy) to align judges."
        ) from exc
    return MemAlignOptimizer(
        reflection_lm=cfg.reflection_lm,
        retrieval_k=cfg.retrieval_k,
        embedding_model=cfg.embedding_model,
        embedding_dim=cfg.embedding_dim,
    )


def align_judge(
    judge: Judge,
    alignment_set: AlignmentSet,
    *,
    optimizer: AlignmentOptimizer | None = None,
    generated_at: str | None = None,
) -> AlignmentOutcome:
    """Align ``judge`` against the Alignment Set with MemAlign.

    Delegates to the public ``judge.align(traces=..., optimizer=...)``. With
    ``optimizer=None``, MLflow uses its default MemAlign optimizer; pass a
    :func:`build_memalign_optimizer` result to configure it.

    The signature only accepts an :class:`~ail.judges.pools.AlignmentSet`, so the
    pool that trains the judge is structurally fixed — the Human Anchor and Task
    Suite cannot be passed here.

    Args:
        judge: The MLflow ``Judge`` to align (e.g. from
            :mod:`ail.judges.scorers`).
        alignment_set: Labeled traces from the Alignment Set pool. Must be
            non-empty (MemAlign rejects an empty trace list).
        optimizer: Optional pre-built optimizer. ``None`` → MLflow's default
            MemAlign.
        generated_at: ISO-8601 timestamp for the report (defaults to now).

    Returns:
        An :class:`AlignmentOutcome` carrying the aligned judge and its report.

    Raises:
        ValueError: If ``alignment_set`` is empty.
    """
    traces = list(alignment_set.traces)
    if not traces:
        raise ValueError(
            "align_judge requires a non-empty AlignmentSet; MemAlign has nothing to learn from "
            "an empty trace list."
        )

    base_name = getattr(judge, "name", "judge")
    aligned = judge.align(traces=traces, optimizer=optimizer)

    report = AlignmentReport(
        base_judge_name=base_name,
        n_alignment_traces=len(traces),
        aligned=True,
        generated_at=generated_at or datetime.now(UTC).isoformat(),
        notes=[
            "aligned on the Alignment Set only (disjoint from Task Suite and Human Anchor); "
            "alignment cadence is decoupled from agent optimization."
        ],
    )
    return AlignmentOutcome(judge=aligned, report=report)
