"""Stage 5: the GEPA agent-optimization loop — evolve an agent artifact, gated.

This is the engine that turns **evaluation feedback into an automatically-evolved
agent artifact** (``docs/ARCHITECTURE.md`` §5). It wraps `GEPA
<https://github.com/gepa-ai/gepa>`_ (the reflective prompt-evolution optimizer,
installed via ``dspy``) around the *existing* frozen-suite comparison machinery so
that the thing GEPA climbs is **our objective, not a vibe**: a candidate artifact's
fitness is the harness's own ``PROMOTE``/``BLOCK`` decision plus the realized L0
token reduction, computed by running the candidate through
:func:`ail.optimize.phase2.run_phase2_comparison` on the **train split** of the
frozen suite.

The artifact under evolution is the **token-efficiency skill body** — the same
markdown the Phase-2 lever (:mod:`ail.optimize.lever`) injects into a candidate's
system prompt. GEPA proposes a new body, we re-score it through the harness, the
reflection LM reads the L0/L1 feedback and proposes again. The seed body is
:func:`ail.optimize.lever.token_efficiency_skill`'s body; any text component works
(an agent instructions string is the natural second component).

**Why ``gepa.optimize`` and not ``mlflow.genai.optimize_prompts``.** We use the
general-purpose :func:`gepa.optimize` (a.k.a. "optimize anything") with a custom
:class:`FrozenSuiteGepaAdapter`, rather than MLflow's
:func:`mlflow.genai.optimize_prompts`, because:

* **Our fitness is a two-arm, fail-closed comparison, not a scorer over a single
  output.** ``optimize_prompts`` scores ``predict_fn(inputs)`` against a reference
  ``outputs`` column via :class:`~mlflow.genai.scorers.Scorer` objects. Our
  objective runs **both** a baseline arm and a candidate arm, gates on execution
  success + L1 programmatic correctness *non-regression*, and only then rewards a
  strict token reduction — and the frozen suite carries **no human-authored
  expectations** to populate an ``outputs`` column (that is exactly why the harness
  runs under :data:`ail.compare.NO_LLM_JUDGE`). The :class:`gepa.core.adapter.GEPAAdapter`
  ``evaluate`` seam lets us call the harness directly and return one fail-closed
  fitness float per task — no contortion to fit a scorer/expectations shape.
* **The artifact is an injected skill body, not a registry prompt template.**
  ``optimize_prompts`` requires the prompts to live in the MLflow Prompt Registry
  and a ``predict_fn`` that calls ``PromptVersion.format`` at inference. Our
  artifact is a free-text blob wired through
  :class:`~ail.optimize.lever.SkillInjectionIntervention`;
  :func:`gepa.optimize`'s ``seed_candidate: dict[str, str]`` maps onto "evolve this
  text" one-to-one, with no registry round-trip.
* **No capability is lost.** ``optimize_prompts`` itself delegates to GEPA under the
  hood (``GepaPromptOptimizer``); calling :func:`gepa.optimize` directly is strictly
  more direct for a non-template artifact and lets us pass ``reflection_lm`` and the
  reflective-dataset shape ourselves.

**The anti-overfit wall (load-bearing).** :func:`split_suite` partitions the frozen
suite into a **train split** (GEPA optimizes against) and a disjoint **held-out
split** (final validation only). Two guarantees keep GEPA off the held-out tasks:

#. **At the call boundary** — only ``split.train_tasks`` is handed to
   :func:`gepa.optimize` as *both* ``trainset`` and ``valset``; the held-out tasks
   are never passed in, so GEPA has nothing held-out to evaluate.
#. **Structurally, inside the fitness function** — :meth:`FrozenSuiteGepaAdapter.evaluate`
   raises :class:`HeldOutLeakError` if it is *ever* asked to score a held-out (or
   non-train) task id, and records every task id it does score
   (:attr:`FrozenSuiteGepaAdapter.evaluated_task_ids`). A test asserts both: that a
   normal run only ever evaluates train ids, and that a held-out id raises.

**The human gate.** :func:`run_gepa_optimization` returns a
:class:`GepaOptimizationResult` carrying the evolved artifact **and** its live
held-out result versus the seed artifact's held-out result. It does **not** write
the skill to disk, register it, or promote it — promotion is a separate human step.

**Cost / fidelity.** Every fitness evaluation runs the agent live (two arms per
task), so iterations (``max_metric_calls``) and train-task count
(``max_train_tasks`` / the split) are bounded and configurable. A cheaper proxy
agent may drive the inner loop (``train_adapter``); the **final** selected candidate
is always validated on the live harness held-out split. See
``docs/GEPA_OPTIMIZATION.md`` for the full tradeoff.
"""

from __future__ import annotations

import math
import random
import re
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ail.compare import ComparisonConfig, Recommendation
from ail.ingest.base import AgentAdapter
from ail.optimize.assets import SkillAsset
from ail.optimize.lever import LeverConfig, SkillInjectionIntervention, token_efficiency_skill
from ail.optimize.phase2 import (
    Phase2Artifact,
    TaskOutcome,
    VerifySpec,
    run_phase2_comparison,
)
from ail.task_suite.schema import Task, TaskSuite

__all__ = [
    "GEPA_SCHEMA_VERSION",
    "DEFAULT_REFLECTION_LM",
    "DEFAULT_COMPONENT",
    "HeldOutLeakError",
    "GepaConfig",
    "SuiteSplit",
    "split_suite",
    "candidate_lever_config",
    "fitness_from_outcome",
    "GepaEvalBatch",
    "FrozenSuiteGepaAdapter",
    "GepaOptimizationResult",
    "run_gepa_optimization",
    "GepaOptimizeFn",
]

#: Version of the GEPA optimization-result contract.
GEPA_SCHEMA_VERSION = "ail.optimize.gepa/v1"

#: The reflection (teacher) LM that drives GEPA's reflective mutation. Recorded
#: as the MLflow model-URI form the orchestrator passes; normalized to the
#: litellm provider form (``databricks/<model>``) at the GEPA boundary by
#: :func:`_normalize_reflection_lm`.
DEFAULT_REFLECTION_LM = "databricks:/databricks-claude-sonnet-4-6"

#: The default named component GEPA evolves (the token-efficiency skill body).
DEFAULT_COMPONENT = "skill_body"

#: The signature of :func:`gepa.optimize`, narrowed to what we call (and so
#: injectable for tests without importing gepa). The real function takes many more
#: keyword args; we only pass the ones below and rely on its defaults otherwise.
GepaOptimizeFn = Callable[..., Any]


class HeldOutLeakError(RuntimeError):
    """Raised when the GEPA fitness function is asked to score a non-train task.

    The load-bearing anti-overfit wall (``docs/ARCHITECTURE.md`` §5): GEPA must
    **never** evaluate a held-out task during optimization, or the reported
    held-out gain would be measured against tasks the optimizer trained on and the
    number would lie. :meth:`FrozenSuiteGepaAdapter.evaluate` raises this if it ever
    receives a held-out (or otherwise non-train) task id, turning the wall from a
    convention into a hard failure a test can prove.
    """


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GepaConfig:
    """Bounded, documented knobs for one GEPA optimization run.

    Cost is the headline concern: every fitness evaluation runs the agent live
    (a baseline arm + a candidate arm per task), so the two cost dials are
    ``max_metric_calls`` (the optimizer's total evaluation budget) and the train
    size (``holdout_fraction`` / ``max_train_tasks``). The remaining knobs shape
    the objective and the reflective mutation.

    Args:
        component_name: The named text component GEPA evolves (default the
            token-efficiency :data:`DEFAULT_COMPONENT` skill body).
        objective_metric: The L0 metric whose reduction is the objective, passed
            straight to :class:`~ail.compare.ComparisonConfig` (default
            ``"total_tokens"``).
        min_token_reduction_pct: Minimum % reduction for the harness objective to
            be met (default ``0.0`` — any strict reduction counts).
        reflection_lm: The reflection/teacher LM URI (default
            :data:`DEFAULT_REFLECTION_LM`). Normalized to the litellm provider form
            at the GEPA boundary.
        max_metric_calls: GEPA's total evaluation budget — the dominant cost dial.
            Bounded; tune up for more search, down for cheaper runs.
        holdout_fraction: Fraction of the suite reserved for held-out validation
            when explicit ``holdout_task_ids`` are not supplied (default ``0.4``).
        max_train_tasks: Optional cap on how many train tasks GEPA actually
            optimizes against (cost bound). The capped-out train tasks are simply
            unused this run — they are **not** moved to held-out.
        seed: Deterministic seed for both the train/held-out split and GEPA.
        reflection_minibatch_size: Optional GEPA reflection minibatch size; ``None``
            uses GEPA's default.
    """

    component_name: str = DEFAULT_COMPONENT
    objective_metric: str = "total_tokens"
    min_token_reduction_pct: float = 0.0
    reflection_lm: str = DEFAULT_REFLECTION_LM
    max_metric_calls: int = 50
    holdout_fraction: float = 0.4
    max_train_tasks: int | None = None
    seed: int = 0
    reflection_minibatch_size: int | None = None

    def comparison_config(self) -> ComparisonConfig:
        """The :class:`~ail.compare.ComparisonConfig` this run gates fitness with."""
        return ComparisonConfig(
            objective_metric=self.objective_metric,
            min_token_reduction_pct=self.min_token_reduction_pct,
        )


# ---------------------------------------------------------------------------
# The train / held-out split (the anti-overfit wall)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SuiteSplit:
    """A disjoint partition of a frozen suite into train and held-out tasks.

    ``train_tasks`` is what GEPA optimizes against; ``holdout_tasks`` is reserved
    for final validation only and is **never** handed to the optimizer.
    ``unused_tasks`` holds train-pool tasks dropped by ``max_train_tasks`` (not
    held-out — just not exercised this run). The two id sets are guaranteed
    disjoint by construction.
    """

    train_tasks: tuple[Task, ...]
    holdout_tasks: tuple[Task, ...]
    unused_tasks: tuple[Task, ...] = ()
    seed: int = 0
    holdout_fraction: float | None = None

    @property
    def train_task_ids(self) -> frozenset[str]:
        """The set of train task ids (the side GEPA may evaluate)."""
        return frozenset(t.task_id for t in self.train_tasks)

    @property
    def holdout_task_ids(self) -> frozenset[str]:
        """The set of held-out task ids (off-limits to the optimizer)."""
        return frozenset(t.task_id for t in self.holdout_tasks)

    def assert_disjoint(self) -> None:
        """Raise if train and held-out overlap (the wall's structural invariant)."""
        overlap = self.train_task_ids & self.holdout_task_ids
        if overlap:
            raise ValueError(
                f"train/held-out split is not disjoint: overlapping task ids {sorted(overlap)}; "
                "GEPA would optimize against a held-out task"
            )


def split_suite(
    suite: TaskSuite,
    *,
    holdout_fraction: float = 0.4,
    holdout_task_ids: Collection[str] | None = None,
    max_train_tasks: int | None = None,
    seed: int = 0,
) -> SuiteSplit:
    """Partition ``suite`` into disjoint train and held-out splits (read-only).

    The suite is never mutated. With explicit ``holdout_task_ids`` the held-out
    side is exactly those tasks (they must all exist in the suite) and the train
    side is the rest; otherwise the held-out side is a deterministic, seeded sample
    of ``ceil(holdout_fraction * n)`` tasks (at least one). ``max_train_tasks``
    optionally caps the train side for cost; the dropped tasks land in
    :attr:`SuiteSplit.unused_tasks`, never in held-out.

    Raises:
        ValueError: if the suite is empty, ``holdout_task_ids`` names a task not in
            the suite, the fraction does not leave at least one task on each side,
            or the resulting split is not disjoint.
    """
    tasks = list(suite.tasks)
    if not tasks:
        raise ValueError("cannot split an empty Task Suite")
    by_id = {t.task_id: t for t in tasks}

    if holdout_task_ids is not None:
        requested = list(dict.fromkeys(holdout_task_ids))  # de-dup, keep order
        missing = [tid for tid in requested if tid not in by_id]
        if missing:
            raise ValueError(f"holdout_task_ids not in suite: {missing}")
        holdout_ids = set(requested)
    else:
        if not 0.0 < holdout_fraction < 1.0:
            raise ValueError(f"holdout_fraction must be in (0, 1), got {holdout_fraction}")
        ordered_ids = sorted(by_id)  # sort first so the seed, not suite order, decides
        rng = random.Random(seed)
        shuffled = ordered_ids[:]
        rng.shuffle(shuffled)
        n_holdout = max(1, math.ceil(holdout_fraction * len(shuffled)))
        n_holdout = min(n_holdout, len(shuffled) - 1)  # always leave >= 1 train task
        holdout_ids = set(shuffled[:n_holdout])

    # Preserve suite order within each side so the split is reproducible.
    holdout_tasks = tuple(t for t in tasks if t.task_id in holdout_ids)
    train_pool = tuple(t for t in tasks if t.task_id not in holdout_ids)
    if not holdout_tasks:
        raise ValueError("held-out split is empty; nothing to validate against")
    if not train_pool:
        raise ValueError("train split is empty; nothing for GEPA to optimize against")

    unused_tasks: tuple[Task, ...] = ()
    train_tasks = train_pool
    if max_train_tasks is not None and max_train_tasks < len(train_pool):
        if max_train_tasks < 1:
            raise ValueError(f"max_train_tasks must be >= 1, got {max_train_tasks}")
        train_tasks = train_pool[:max_train_tasks]
        unused_tasks = train_pool[max_train_tasks:]

    split = SuiteSplit(
        train_tasks=train_tasks,
        holdout_tasks=holdout_tasks,
        unused_tasks=unused_tasks,
        seed=seed,
        holdout_fraction=None if holdout_task_ids is not None else holdout_fraction,
    )
    split.assert_disjoint()
    return split


# ---------------------------------------------------------------------------
# Artifact <-> candidate config bridge
# ---------------------------------------------------------------------------


def _skill_asset_from_body(body: str, *, base: SkillAsset) -> SkillAsset:
    """A :class:`~ail.optimize.assets.SkillAsset` carrying an evolved ``body``.

    Keeps ``base``'s front-matter identity (``slug``/``name``/``description``) so
    the injection marker and provenance are stable, and rebuilds ``raw`` from the
    new body. Only ``name`` and ``body`` affect the injected system-prompt section
    (:meth:`~ail.optimize.assets.SkillAsset.as_system_prompt_section`).
    """
    raw = f"---\nname: {base.name}\ndescription: {base.description}\n---\n\n{body.strip()}\n"
    return SkillAsset(
        slug=base.slug,
        name=base.name,
        description=base.description,
        body=body,
        raw=raw,
        source_path=None,
    )


def candidate_lever_config(
    skill_body: str,
    *,
    name: str,
    base: SkillAsset | None = None,
) -> LeverConfig:
    """Build a candidate :class:`~ail.optimize.lever.LeverConfig` from a skill body.

    Wraps ``skill_body`` in the same :class:`~ail.optimize.lever.SkillInjectionIntervention`
    the Phase-2 lever uses, so an evolved body is evaluated through the *unchanged*
    comparison machinery. ``base`` supplies the front-matter identity (defaults to
    the token-efficiency skill).
    """
    base_asset = base or token_efficiency_skill()
    asset = _skill_asset_from_body(skill_body, base=base_asset)
    return LeverConfig(
        name=name,
        intervention=SkillInjectionIntervention(name=name, skill=asset),
        description=f"GEPA-evolved token-efficiency skill body ({name}).",
    )


# ---------------------------------------------------------------------------
# Fitness: the frozen-suite objective, fail-closed
# ---------------------------------------------------------------------------


def fitness_from_outcome(outcome: TaskOutcome | None) -> float:
    """Map one harness :class:`~ail.optimize.phase2.TaskOutcome` to a GEPA fitness.

    **Fitness is the frozen-suite objective, not a vibe.** A candidate scores above
    zero on a task **only** when the harness recommends ``PROMOTE`` — i.e. the
    token objective was met **and** every guardrail passed (execution success + L1
    correctness non-regression). The score is then the realized token-reduction
    fraction in ``(0, 1]``, so reducing more tokens scores higher. Everything else
    is fail-closed to ``0.0``: a missing outcome, a ``BLOCK``, a broken/regressed L1
    correctness guardrail, or a candidate that did not actually reduce tokens.
    Higher is better, as GEPA requires.
    """
    if outcome is None or outcome.recommendation is not Recommendation.PROMOTE:
        return 0.0
    pct = outcome.token_delta_pct
    if pct is None or pct >= 0:  # no reduction (or undefined) is not a win
        return 0.0
    return min(1.0, abs(pct) / 100.0)


# ---------------------------------------------------------------------------
# The GEPA adapter: our harness as GEPA's evaluate/reflect seam
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class GepaEvalBatch:
    """The result of scoring a batch — mirrors :class:`gepa.core.adapter.EvaluationBatch`.

    Defined locally (with the same ``outputs`` / ``scores`` / ``trajectories`` /
    ``objective_scores`` fields) so the fitness path imports **no** gepa: the GEPA
    engine duck-types this return (it only reads those attributes and hands the same
    object back to :meth:`FrozenSuiteGepaAdapter.make_reflective_dataset`), so the
    core install — and the offline tests — never need the optional optimizer
    backend. ``scores`` are per-example, higher-is-better, length-aligned with
    ``outputs`` (and ``trajectories`` when captured).
    """

    outputs: list[Any]
    scores: list[float]
    trajectories: list[Any] | None = None
    objective_scores: list[dict[str, float]] | None = None


class FrozenSuiteGepaAdapter:
    """Wire the frozen-suite comparison harness in as GEPA's fitness + reflection.

    Implements the :class:`gepa.core.adapter.GEPAAdapter` protocol structurally (no
    inheritance, so importing this module never imports gepa):

    * :meth:`evaluate` runs each train task's candidate body through
      :func:`ail.optimize.phase2.run_phase2_comparison` (the *unchanged* two-arm,
      fail-closed harness) and returns one :func:`fitness_from_outcome` score per
      task. It enforces the anti-overfit wall — a held-out or non-train task id
      raises :class:`HeldOutLeakError` — and records every scored id in
      :attr:`evaluated_task_ids`.
    * :meth:`make_reflective_dataset` turns each task's L0 token delta + L1
      correctness outcome + harness decision into the feedback text the reflection
      LM reads to propose a better body.

    The adapter holds the **inner** adapter (a cheaper proxy if supplied), the
    per-task L1 :class:`~ail.optimize.phase2.VerifySpec` map, and the split so the
    wall is self-contained.
    """

    #: Part of the :class:`gepa.core.adapter.GEPAAdapter` contract: gepa's
    #: ``ReflectiveMutationProposer`` reads ``adapter.propose_new_texts`` to decide
    #: whether to call a *custom* proposer (a callable) or fall back to its
    #: **built-in** reflection-LM proposer (when ``None``). We want the built-in path
    #: — it drives ``InstructionProposalSignature`` with the ``reflection_lm`` we pass
    #: at the GEPA boundary, reading the L0/L1/decision feedback in
    #: :meth:`make_reflective_dataset` to mutate the skill body. This attribute MUST
    #: exist (the proposer evaluates ``adapter.propose_new_texts is not None``
    #: directly, not via ``getattr``); omitting it makes every reflective-mutation
    #: iteration raise ``AttributeError`` and the loop evolve nothing. Typed as a
    #: bare ``Callable | None`` so this module imports no gepa.
    propose_new_texts: Callable[..., dict[str, str]] | None = None

    def __init__(
        self,
        *,
        suite: TaskSuite,
        adapter: AgentAdapter,
        split: SuiteSplit,
        verify_specs: Mapping[str, VerifySpec] | None = None,
        config: GepaConfig | None = None,
        fixtures_root: str | None = None,
        generated_at: str | None = None,
    ) -> None:
        self.suite = suite
        self.adapter = adapter
        self.split = split
        self.verify_specs = dict(verify_specs or {})
        self.config = config or GepaConfig()
        self.fixtures_root = fixtures_root
        self.generated_at = generated_at or datetime.now(UTC).isoformat()
        #: Every train task id this adapter has scored — the audit trail a test uses
        #: to prove GEPA only ever evaluated the train split.
        self.evaluated_task_ids: set[str] = set()

    # -- the wall ----------------------------------------------------------

    def _guard_train(self, task: Task) -> None:
        """Fail closed unless ``task`` is in the train split (the anti-overfit wall)."""
        if task.task_id in self.split.holdout_task_ids:
            raise HeldOutLeakError(
                f"GEPA fitness was asked to evaluate held-out task {task.task_id!r}; the "
                "anti-overfit wall forbids the optimizer ever seeing the held-out split"
            )
        if task.task_id not in self.split.train_task_ids:
            raise HeldOutLeakError(
                f"GEPA fitness was asked to evaluate task {task.task_id!r}, which is not in the "
                "train split (train ids: "
                f"{sorted(self.split.train_task_ids)})"
            )

    # -- fitness -----------------------------------------------------------

    def _score_task(self, task: Task, skill_body: str) -> TaskOutcome | None:
        """Run one train task's candidate body through the unchanged harness."""
        candidate = candidate_lever_config(skill_body, name="candidate-gepa")
        artifact = run_phase2_comparison(
            suite=self.suite,
            adapter=self.adapter,
            candidate=candidate,
            verify_specs=self.verify_specs,
            config=self.config.comparison_config(),
            task_ids={task.task_id},
            fixtures_root=self.fixtures_root,
            generated_at=self.generated_at,
        )
        return artifact.outcomes[0] if artifact.outcomes else None

    def evaluate(
        self,
        batch: list[Task],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> GepaEvalBatch:
        """Score a batch of train tasks for one candidate body (GEPA's fitness call).

        Returns a :class:`GepaEvalBatch` with one :func:`fitness_from_outcome` score
        per task (higher is better) and, when ``capture_traces`` is set, a per-task
        trajectory the reflective dataset reads. Per the GEPA adapter contract, an
        individual task's failure is a fail-closed ``0.0`` score (never a raise);
        only the wall (:class:`HeldOutLeakError`) and a misconfigured candidate
        raise.
        """
        if self.config.component_name not in candidate:
            raise KeyError(
                f"candidate is missing component {self.config.component_name!r}; "
                f"got components {sorted(candidate)}"
            )
        skill_body = candidate[self.config.component_name]

        outputs: list[dict[str, Any]] = []
        scores: list[float] = []
        trajectories: list[dict[str, Any]] | None = [] if capture_traces else None

        for task in batch:
            self._guard_train(task)
            self.evaluated_task_ids.add(task.task_id)
            outcome = self._score_task(task, skill_body)
            score = fitness_from_outcome(outcome)
            scores.append(score)
            record = _trajectory_record(task, outcome, score)
            outputs.append(
                {
                    "task_id": task.task_id,
                    "score": score,
                    "recommendation": record["recommendation"],
                }
            )
            if trajectories is not None:
                trajectories.append(record)

        return GepaEvalBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    # -- reflection --------------------------------------------------------

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: GepaEvalBatch,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """Build the reflective dataset: L0 token + L1 correctness feedback per task.

        Each record gives the reflection LM the task prompt, the candidate's score,
        and a feedback string that names the L0 token delta, the L1 correctness
        outcome, and the harness decision (with blocking reasons) — so the proposed
        body is shaped by *why* the candidate did or did not win, not by a bare
        number.
        """
        trajectories = eval_batch.trajectories or []
        records: list[dict[str, Any]] = [
            {
                "Inputs": {
                    "task_prompt": traj["prompt"],
                    "category": traj["category"],
                    "difficulty": traj["difficulty"],
                },
                "Generated Outputs": (
                    "The token-efficiency skill body under test produced this harness outcome."
                ),
                "Feedback": _feedback_text(traj),
            }
            for traj in trajectories
        ]
        # GEPA requests a subset of components to update; we evolve text components
        # by the same feedback, so return the records for each requested component.
        return dict.fromkeys(components_to_update, records)


def _trajectory_record(task: Task, outcome: TaskOutcome | None, score: float) -> dict[str, Any]:
    """A compact, JSON-serializable trajectory for one scored task."""
    if outcome is None:
        return {
            "task_id": task.task_id,
            "prompt": task.prompt,
            "category": task.category.value,
            "difficulty": task.difficulty.value,
            "score": score,
            "recommendation": Recommendation.BLOCK.value,
            "baseline_tokens": None,
            "candidate_tokens": None,
            "token_delta_pct": None,
            "l1_outcome": "no_verdict",
            "blocking_reasons": ["comparison produced no outcome for this task"],
        }
    return {
        "task_id": task.task_id,
        "prompt": task.prompt,
        "category": task.category.value,
        "difficulty": task.difficulty.value,
        "score": score,
        "recommendation": outcome.recommendation.value,
        "baseline_tokens": outcome.baseline_total_tokens,
        "candidate_tokens": outcome.candidate_total_tokens,
        "token_delta_pct": outcome.token_delta_pct,
        "l1_outcome": outcome.l1_outcome.value,
        "blocking_reasons": list(outcome.blocking_reasons),
    }


def _feedback_text(traj: Mapping[str, Any]) -> str:
    """Render the L0/L1/decision feedback string the reflection LM reads."""
    base = traj.get("baseline_tokens")
    cand = traj.get("candidate_tokens")
    pct = traj.get("token_delta_pct")
    pct_str = "n/a" if pct is None else f"{pct:+.2f}%"
    token_phrase = (
        "tokens not measured"
        if base is None or cand is None
        else f"L0 tokens {base:g} -> {cand:g} ({pct_str})"
    )
    decision = str(traj.get("recommendation", "block")).upper()
    l1 = traj.get("l1_outcome", "unknown")
    reasons = traj.get("blocking_reasons") or []
    lead = (
        "PROMOTED: the skill cut tokens with correctness held."
        if decision == Recommendation.PROMOTE.value.upper()
        else "BLOCKED: the candidate did not clear the fail-closed gate."
    )
    feedback = f"{lead} {token_phrase}; L1 correctness outcome: {l1}; harness decision: {decision}."
    if reasons:
        feedback += " Blocking reasons: " + " | ".join(str(r) for r in reasons) + "."
    feedback += (
        " To improve: reduce redundant Read/Bash calls and repeated setup boilerplate "
        "WITHOUT changing what the task produces — a correctness regression scores zero."
    )
    return feedback


# ---------------------------------------------------------------------------
# Result contract (human gate)
# ---------------------------------------------------------------------------


class GepaOptimizationResult(BaseModel):
    """The candidate evolved artifact + its held-out validation (NOT auto-promoted).

    This is what the loop returns to the human gate: the evolved skill body, the
    seed it started from, and the **live** held-out result of the evolved artifact
    versus the seed artifact, plus the split + run provenance. Promotion (writing
    the body to the asset, registering it, shipping it) is a separate human step —
    :attr:`human_gate_required` is always ``True`` and this object has no side
    effect that applies the artifact.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = GEPA_SCHEMA_VERSION
    generated_at: str | None = None
    human_gate_required: bool = True

    component_name: str = DEFAULT_COMPONENT
    seed_skill_body: str = ""
    evolved_skill_body: str = ""
    changed: bool = False

    reflection_lm: str = DEFAULT_REFLECTION_LM
    max_metric_calls: int = 0
    gepa_total_metric_calls: int | None = None
    gepa_num_candidates: int | None = None
    gepa_best_val_score: float | None = None

    suite_version: str = ""
    suite_content_hash: str = ""
    split_seed: int = 0
    holdout_fraction: float | None = None
    train_task_ids: list[str] = Field(default_factory=list)
    holdout_task_ids: list[str] = Field(default_factory=list)
    unused_train_task_ids: list[str] = Field(default_factory=list)

    #: The evolved artifact run on the held-out split, live (final validation).
    holdout_evolved: Phase2Artifact | None = None
    #: The seed artifact run on the same held-out split, live (the baseline to beat).
    holdout_seed_baseline: Phase2Artifact | None = None

    notes: list[str] = Field(default_factory=list)

    @property
    def holdout_savings_delta_pct(self) -> float | None:
        """Evolved minus seed realized held-out savings %, or ``None`` if either is unset.

        The honest anti-overfit headline: did evolution help on tasks GEPA never
        saw? Positive means the evolved body realized more token savings on the
        held-out split than the seed body did.
        """
        if self.holdout_evolved is None or self.holdout_seed_baseline is None:
            return None
        evolved = self.holdout_evolved.realized_token_savings_pct
        seed = self.holdout_seed_baseline.realized_token_savings_pct
        if evolved is None or seed is None:
            return None
        return round(evolved - seed, 4)


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def _normalize_reflection_lm(spec: str) -> str:
    """Normalize an MLflow model-URI (``provider:/model``) to litellm form (``provider/model``).

    GEPA resolves a string ``reflection_lm`` via ``litellm.completion``, which uses
    ``provider/model`` (single slash). The orchestrator passes the MLflow URI form
    ``databricks:/databricks-claude-sonnet-4-6``; this rewrites a leading
    ``provider:/`` to ``provider/`` while leaving a real URL scheme (``http://``)
    and an already-litellm string untouched.
    """
    return re.sub(r"^([a-z][a-z0-9_-]*):/(?!/)", r"\1/", spec)


def _holdout_validation(
    *,
    suite: TaskSuite,
    adapter: AgentAdapter,
    skill_body: str,
    config_name: str,
    holdout_ids: frozenset[str],
    verify_specs: Mapping[str, VerifySpec],
    cfg: GepaConfig,
    fixtures_root: str | None,
    generated_at: str,
) -> Phase2Artifact:
    """Run one skill body on the held-out split through the live harness."""
    candidate = candidate_lever_config(skill_body, name=config_name)
    return run_phase2_comparison(
        suite=suite,
        adapter=adapter,
        candidate=candidate,
        verify_specs=verify_specs,
        config=cfg.comparison_config(),
        task_ids=holdout_ids,
        fixtures_root=fixtures_root,
        generated_at=generated_at,
    )


def run_gepa_optimization(
    *,
    suite: TaskSuite,
    adapter: AgentAdapter,
    seed_skill_body: str | None = None,
    verify_specs: Mapping[str, VerifySpec] | None = None,
    config: GepaConfig | None = None,
    holdout_task_ids: Collection[str] | None = None,
    train_adapter: AgentAdapter | None = None,
    fixtures_root: str | None = None,
    gepa_optimize: GepaOptimizeFn | None = None,
    generated_at: str | None = None,
) -> GepaOptimizationResult:
    """Evolve the token-efficiency skill body with GEPA, then validate on held-out.

    Splits the frozen ``suite`` into disjoint train/held-out, runs
    :func:`gepa.optimize` with :class:`FrozenSuiteGepaAdapter` as the fitness +
    reflection seam against the **train split only**, then validates the selected
    candidate **and** the seed body on the **held-out split** through the live
    harness. Returns a human-gated :class:`GepaOptimizationResult` — it does **not**
    apply or promote the artifact.

    Args:
        suite: The frozen Task Suite (read only; split, never mutated).
        adapter: The **live** agent adapter used for held-out validation (and for
            the GEPA inner loop unless ``train_adapter`` is given).
        seed_skill_body: The body GEPA starts from; defaults to the token-efficiency
            skill's body (:func:`ail.optimize.lever.token_efficiency_skill`).
        verify_specs: Per-task L1 :class:`~ail.optimize.phase2.VerifySpec` map. A
            task with no spec has no correctness signal and fails closed (fitness
            ``0``), exactly as in Phase 2.
        config: :class:`GepaConfig` knobs (budget, split, reflection LM, objective).
        holdout_task_ids: Optional explicit held-out ids; otherwise a seeded
            ``config.holdout_fraction`` sample is held out.
        train_adapter: Optional cheaper **proxy** adapter for the GEPA inner loop.
            The final held-out validation still runs on the live ``adapter``.
        fixtures_root: Optional ``eval/phase2_fixtures`` root (per-arm isolation +
            tamper-proof verify); ``None`` uses repo discovery.
        gepa_optimize: The optimizer entry point. Defaults to :func:`gepa.optimize`
            (lazy-imported); injectable so tests drive the loop with **no** real
            optimizer, agent, or model.
        generated_at: ISO-8601 stamp recorded on the result and threaded through the
            comparisons (caller-supplied so a run is reproducible in tests).

    Returns:
        A :class:`GepaOptimizationResult` carrying the candidate evolved body and
        its live held-out result versus the seed body's — for the human gate.
    """
    cfg = config or GepaConfig()
    stamp = generated_at or datetime.now(UTC).isoformat()
    specs = dict(verify_specs or {})
    seed_body = seed_skill_body if seed_skill_body is not None else token_efficiency_skill().body

    split = split_suite(
        suite,
        holdout_fraction=cfg.holdout_fraction,
        holdout_task_ids=holdout_task_ids,
        max_train_tasks=cfg.max_train_tasks,
        seed=cfg.seed,
    )

    inner_adapter = train_adapter if train_adapter is not None else adapter
    gepa_adapter = FrozenSuiteGepaAdapter(
        suite=suite,
        adapter=inner_adapter,
        split=split,
        verify_specs=specs,
        config=cfg,
        fixtures_root=fixtures_root,
        generated_at=stamp,
    )

    optimize_fn = gepa_optimize if gepa_optimize is not None else _default_gepa_optimize()
    seed_candidate = {cfg.component_name: seed_body}

    # THE WALL, at the call boundary: only the train split is handed to GEPA, as
    # BOTH trainset and valset. The held-out tasks are never passed in.
    result = optimize_fn(
        seed_candidate=seed_candidate,
        trainset=list(split.train_tasks),
        valset=list(split.train_tasks),
        adapter=gepa_adapter,
        reflection_lm=_normalize_reflection_lm(cfg.reflection_lm),
        max_metric_calls=cfg.max_metric_calls,
        reflection_minibatch_size=cfg.reflection_minibatch_size,
        seed=cfg.seed,
        display_progress_bar=False,
    )

    evolved_body = _best_skill_body(result, cfg.component_name, seed_body)

    # Final validation on the LIVE harness held-out split: the evolved body and the
    # seed body, so the result reports the evolved artifact vs the original baseline.
    holdout_evolved = _holdout_validation(
        suite=suite,
        adapter=adapter,
        skill_body=evolved_body,
        config_name="candidate-gepa-evolved",
        holdout_ids=split.holdout_task_ids,
        verify_specs=specs,
        cfg=cfg,
        fixtures_root=fixtures_root,
        generated_at=stamp,
    )
    holdout_seed = _holdout_validation(
        suite=suite,
        adapter=adapter,
        skill_body=seed_body,
        config_name="candidate-gepa-seed",
        holdout_ids=split.holdout_task_ids,
        verify_specs=specs,
        cfg=cfg,
        fixtures_root=fixtures_root,
        generated_at=stamp,
    )

    return GepaOptimizationResult(
        generated_at=stamp,
        human_gate_required=True,
        component_name=cfg.component_name,
        seed_skill_body=seed_body,
        evolved_skill_body=evolved_body,
        changed=evolved_body != seed_body,
        reflection_lm=cfg.reflection_lm,
        max_metric_calls=cfg.max_metric_calls,
        gepa_total_metric_calls=_maybe_int(getattr(result, "total_metric_calls", None)),
        gepa_num_candidates=_maybe_int(getattr(result, "num_candidates", None)),
        gepa_best_val_score=_best_val_score(result),
        suite_version=suite.version,
        suite_content_hash=suite.content_hash,
        split_seed=cfg.seed,
        holdout_fraction=split.holdout_fraction,
        train_task_ids=sorted(split.train_task_ids),
        holdout_task_ids=sorted(split.holdout_task_ids),
        unused_train_task_ids=sorted(t.task_id for t in split.unused_tasks),
        holdout_evolved=holdout_evolved,
        holdout_seed_baseline=holdout_seed,
        notes=[
            "CANDIDATE artifact: this result is NOT auto-applied or promoted; promotion is a "
            "separate human step (the human gate).",
            "Fitness is the harness PROMOTE decision + realized L0 token reduction on the TRAIN "
            "split, fail-closed on execution / L1 correctness — never a vibe score.",
            "GEPA optimized against the train split only; the held-out split above was scored "
            "exclusively by the live harness after optimization (the anti-overfit wall).",
        ],
    )


def _default_gepa_optimize() -> GepaOptimizeFn:
    """Lazily resolve :func:`gepa.optimize` (so the core install stays importable)."""
    try:
        from gepa import optimize
    except ImportError as exc:  # pragma: no cover - exercised only without the optional dep
        raise ImportError(
            "gepa is required to run the GEPA optimization loop but is not installed. "
            "Install the optimizer backend (it ships with the 'align' extra via dspy): "
            "pip install 'ail[align]'"
        ) from exc
    return optimize


def _best_skill_body(result: Any, component_name: str, seed_body: str) -> str:
    """Extract the best candidate's evolved body from a GEPA result (defensively)."""
    candidate = getattr(result, "best_candidate", None)
    if isinstance(candidate, Mapping):
        body = candidate.get(component_name)
        if isinstance(body, str) and body.strip():
            return body
    return seed_body


def _best_val_score(result: Any) -> float | None:
    """The best candidate's aggregate validation score, if the result exposes it."""
    scores = getattr(result, "val_aggregate_scores", None)
    idx = getattr(result, "best_idx", None)
    if isinstance(scores, Sequence) and isinstance(idx, int) and 0 <= idx < len(scores):
        try:
            return float(scores[idx])
        except (TypeError, ValueError):
            return None
    return None


def _maybe_int(value: Any) -> int | None:
    """Coerce a metric to ``int`` for the result, or ``None`` if it is not numeric."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return None
