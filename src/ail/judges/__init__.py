"""L2 judged metrics: the evolving evaluation spine.

This package is the **L2 — Judged** tier of the layered metrics design
(``docs/ARCHITECTURE.md`` §3): LLM-as-judge scorers built on the public MLflow
GenAI API, aligned with MemAlign, and audited against the Human Anchor. It
exists to make subjective quality measurable *without* letting the judge and the
agent co-adapt — the failure mode §2 calls out.

Three capabilities, plus the disjoint-pool types that keep them honest:

* **Scorers** (:mod:`ail.judges.scorers`) — ``make_judge`` wrappers for
  ``correctness`` (the Phase-2 guardrail), ``modularity``, and ``groundedness``.
* **Alignment** (:mod:`ail.judges.alignment`) — ``judge.align`` over MemAlign,
  consuming the Alignment Set only, on a cadence decoupled from optimization.
* **Agreement** (:mod:`ail.judges.agreement`) — judge-vs-human agreement on the
  Human Anchor as a first-class metric with a configurable floor and a
  ``distrusted`` signal.
* **Pools** (:mod:`ail.judges.pools`) — typed handles (``AlignmentSet`` /
  ``HumanAnchor``) and :func:`assert_pools_disjoint` that make "the three pools
  are never mixed" a property of the types.

The loop controller and the app consume this surface; they do not reach into the
submodules' internals.
"""

from ail.judges.agreement import (
    AgreementConfig,
    ScorePair,
    coerce_score,
    compute_agreement,
    log_agreement,
    score_anchor,
)
from ail.judges.alignment import (
    AlignmentOutcome,
    MemAlignConfig,
    align_judge,
    build_memalign_optimizer,
)
from ail.judges.contract import (
    SCHEMA_VERSION,
    AgreementItem,
    AgreementReport,
    AlignmentReport,
)
from ail.judges.pools import (
    AlignmentSet,
    AnchorItem,
    HumanAnchor,
    Pool,
    PoolOverlapError,
    ScoreValue,
    UnresolvedTraceIdError,
    assert_pools_disjoint,
)
from ail.judges.registration import (
    DEFAULT_SAMPLING_RATE,
    list_registered_scorers,
    register_scorers,
    unregister_scorers,
)
from ail.judges.scorers import (
    CORRECTNESS,
    DEFAULT_SCORERS,
    GROUNDEDNESS,
    MODULARITY,
    ScorerSpec,
    make_correctness_judge,
    make_groundedness_judge,
    make_modularity_judge,
    make_scorer,
    with_rubric,
)

__all__ = [
    # contract
    "SCHEMA_VERSION",
    "AgreementItem",
    "AgreementReport",
    "AlignmentReport",
    # pools (frozen evaluation wall)
    "Pool",
    "PoolOverlapError",
    "UnresolvedTraceIdError",
    "ScoreValue",
    "AnchorItem",
    "HumanAnchor",
    "AlignmentSet",
    "assert_pools_disjoint",
    # scorers
    "ScorerSpec",
    "CORRECTNESS",
    "MODULARITY",
    "GROUNDEDNESS",
    "DEFAULT_SCORERS",
    "make_scorer",
    "make_correctness_judge",
    "make_modularity_judge",
    "make_groundedness_judge",
    "with_rubric",
    # alignment (MemAlign)
    "MemAlignConfig",
    "AlignmentOutcome",
    "build_memalign_optimizer",
    "align_judge",
    # agreement (anti-co-adaptation)
    "AgreementConfig",
    "ScorePair",
    "coerce_score",
    "compute_agreement",
    "score_anchor",
    "log_agreement",
    # registration (scheduled scorers)
    "DEFAULT_SAMPLING_RATE",
    "register_scorers",
    "list_registered_scorers",
    "unregister_scorers",
]
