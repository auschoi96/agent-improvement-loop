"""The proposed-action model — the controller's only output, fully human-gated.

``docs/LOOP_CONTROLLER.md`` (Option A) draws one hard line: the autonomous loop
**detects, decides, proves, gates, and proposes** — it never applies a live
change. :class:`ProposedAction` is what it emits: a typed, JSON-round-trippable
record carrying everything a human needs to approve the apply *later* (lane 3),
and nothing that could apply it now.

Every proposal carries the four things ``docs/LOOP_CONTROLLER.md`` §"The
proposed-action record" requires:

* **What** — the concrete :class:`ProposedChange` (a metric-view ``CREATE`` SQL, a
  skill/instruction diff, a GEPA-evolved-body reference, or a revert target).
* **Why** — the :class:`TriggerSignal`: which RLM recommendation / judge dimension
  / L0 waste pattern fired, with the trace references that justify it.
* **Proof** — the :class:`ProofSummary`: the frozen-suite objective delta with
  correctness held, sourced from the comparison harness's aggregate
  (:class:`~ail.optimize.phase2.Phase2Artifact`, itself built from
  :class:`~ail.compare.contract.ComparisonResult`). **Optional.** A
  *prove-before-propose* proposal (the ``ail.loop.controller.run_cycle`` path)
  carries it and only exists when it proved an improvement; an **evidence-first**
  proposal (the ``ail.loop.evidence_cycle.run_evidence_cycle`` path, per
  ``docs/PRODUCT_ARCHITECTURE.md`` §3/§7 — proving is opt-in Tier-2, not a pre-ship
  gate) carries ``proof=None`` and rests on its evidence + gate status alone. The
  human decides on the evidence, optionally running Tier-2 "verify on my suite"
  later to attach a proof.
* **Gate status** — the :class:`GateStatus`: the readiness tier, the certifying
  judge's agreement, and scored coverage, from
  :class:`~ail.readiness.contract.ReadinessStatus`.

**Risk class is informational, never an auto-apply switch.** Per Option A *every*
proposal is human-gated regardless of :class:`RiskClass`; the field tells the
reviewer the blast radius (additive asset vs. agent change), it does not let the
controller ship anything.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ail.optimize.phase2 import L1Outcome, Phase2Artifact
from ail.readiness.contract import ReadinessStatus

__all__ = [
    "SCHEMA_VERSION",
    "ActionKind",
    "RiskClass",
    "ProposalStatus",
    "TriggerKind",
    "ChangeKind",
    "LocalApplyTargetKind",
    "LocalApplySpec",
    "TriggerSignal",
    "ProposedChange",
    "ProofSummary",
    "GateStatus",
    "ProposedAction",
    "default_risk_class",
    "derive_proposal_id",
]

#: Version of the proposed-action contract. Bump the minor for additive,
#: backward-compatible fields; the major for breaking shape changes.
SCHEMA_VERSION = "ail.loop.proposals/v1.1"


class ActionKind(StrEnum):
    """The kind of change a proposal would make to an agent.

    The vocabulary of changes lane 3 knows how to apply. The decision rules in
    :mod:`ail.loop.decision_rules` emit the illustrative subset documented in
    ``docs/LOOP_CONTROLLER.md``; the enum itself is broader so a proposal from any
    source is typed.
    """

    METRIC_VIEW = "metric_view"
    SKILL_UPDATE = "skill_update"
    INSTRUCTION_UPDATE = "instruction_update"
    GEPA_PROMPT = "gepa_prompt"
    REVERT = "revert"
    #: An **open-ended** change an agent produces (a LATER executor lane, L7b-2) —
    #: not a pre-specified METRIC_VIEW/SKILL/INSTRUCTION/GEPA/REVERT, but "whatever the
    #: target agent needs" (new tool, new/edited/deleted skill, new table, metric view,
    #: cached examples, a multi-file refactor). L7b-1 defines only the representation;
    #: it does **not** run an agent or execute the change (apply fails closed on it).
    AGENT_TASK = "agent_task"


class RiskClass(StrEnum):
    """The blast radius of a change — **informational for the reviewer only**.

    Per Option A every proposal is human-gated regardless of risk class, so this
    is never an auto-apply switch (``docs/LOOP_CONTROLLER.md``). It tells the human
    *how much* a change touches:

    * :attr:`ADDITIVE_ASSET` — a new governed asset (e.g. a metric view) that adds
      a read-path without altering the agent; low blast radius, trivially
      reversible.
    * :attr:`AGENT_CHANGE` — a change to the agent's own prompt/skill/instructions;
      higher blast radius.
    """

    ADDITIVE_ASSET = "additive_asset"
    AGENT_CHANGE = "agent_change"


class ProposalStatus(StrEnum):
    """A proposal's lifecycle state.

    The controller only ever emits :attr:`PENDING`. The remaining states are set
    by the lane-3 approval queue (``docs/LOOP_CONTROLLER.md``): a human moves a
    proposal to :attr:`APPROVED`/:attr:`REJECTED`, the gated apply marks it
    :attr:`APPLIED`, and a newer proposal for the same target supersedes an older
    one (:attr:`SUPERSEDED`).
    """

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"
    SUPERSEDED = "superseded"


class TriggerKind(StrEnum):
    """Which feedback signal a proposal fired on — the head of the "why" payload.

    Mirrors the illustrative decision rules of ``docs/LOOP_CONTROLLER.md``.

    The deterministic **Lane A** rules (:mod:`ail.loop.decision_rules`) each set the
    specific signal kind they fired on. :attr:`AGENT_PLANNER` is the distinct source
    marker for a **Lane B** decision — one proposed by the LLM-agent planner
    (:mod:`ail.loop.planner`) rather than a deterministic rule — so a proposal's
    origin (A vs B) is attributable straight off ``trigger.kind``. A B decision
    still carries the target detail (``judge_name`` / ``metric`` / ``asset_type``)
    on its :class:`TriggerSignal`, so the controller's proof + certifying-judge
    gates apply to it identically; only the *source* differs.
    """

    RLM_RECOMMENDED_ASSET = "rlm_recommended_asset"
    REDUNDANT_READ_PATTERN = "redundant_read_pattern"
    JUDGE_DIMENSION_BELOW_THRESHOLD = "judge_dimension_below_threshold"
    POST_APPLY_REGRESSION = "post_apply_regression"
    AGENT_PLANNER = "agent_planner"


class ChangeKind(StrEnum):
    """The form the concrete change takes (and which payload field carries it)."""

    METRIC_VIEW_SQL = "metric_view_sql"
    SKILL_DIFF = "skill_diff"
    INSTRUCTION_DIFF = "instruction_diff"
    EVOLVED_BODY_REF = "evolved_body_ref"
    REVERT_REF = "revert_ref"
    #: The change form for :attr:`ActionKind.AGENT_TASK`: an NL ``plan`` (the intended
    #: change + why, from the evidence — required) plus, filled by the executor (L7b-2)
    #: in a sandbox *before* approval, a concrete ``preview_diff`` the human reviews and
    #: a ``produced_change_ref`` (an L6 snapshot / UC Volume ref) to commit on approval.
    AGENT_TASK_PLAN = "agent_task_plan"


class LocalApplyTargetKind(StrEnum):
    """How the local companion rewrites an explicitly registered target file."""

    CLAUDE_SKILL = "claude_skill"
    PROMPT_FILE = "prompt_file"
    AGENTS_MD = "agents_md"


#: Default risk class per action kind. A metric view is additive; every change to
#: the agent's own prompt/skill/instructions (and a revert of one) is an agent
#: change. A caller may override per proposal (e.g. reverting an *additive* asset).
_DEFAULT_RISK_CLASS: dict[ActionKind, RiskClass] = {
    ActionKind.METRIC_VIEW: RiskClass.ADDITIVE_ASSET,
    ActionKind.SKILL_UPDATE: RiskClass.AGENT_CHANGE,
    ActionKind.INSTRUCTION_UPDATE: RiskClass.AGENT_CHANGE,
    ActionKind.GEPA_PROMPT: RiskClass.AGENT_CHANGE,
    ActionKind.REVERT: RiskClass.AGENT_CHANGE,
    # An open-ended agent-produced change is, by definition, a change to the agent —
    # the highest-blast-radius kind; always AGENT_CHANGE (never an additive asset).
    ActionKind.AGENT_TASK: RiskClass.AGENT_CHANGE,
}


def default_risk_class(action_kind: ActionKind) -> RiskClass:
    """The default :class:`RiskClass` for ``action_kind`` (informational only)."""
    return _DEFAULT_RISK_CLASS[action_kind]


#: The change form each action kind must carry — enforced on :class:`ProposedAction`
#: so a proposal can never claim one action while carrying another's change.
_ACTION_CHANGE_KIND: dict[ActionKind, ChangeKind] = {
    ActionKind.METRIC_VIEW: ChangeKind.METRIC_VIEW_SQL,
    ActionKind.SKILL_UPDATE: ChangeKind.SKILL_DIFF,
    ActionKind.INSTRUCTION_UPDATE: ChangeKind.INSTRUCTION_DIFF,
    ActionKind.GEPA_PROMPT: ChangeKind.EVOLVED_BODY_REF,
    ActionKind.REVERT: ChangeKind.REVERT_REF,
    ActionKind.AGENT_TASK: ChangeKind.AGENT_TASK_PLAN,
}

#: Which payload field on :class:`ProposedChange` each change kind must populate.
#: For :attr:`ChangeKind.AGENT_TASK_PLAN` the required field is the NL ``plan`` — the
#: concrete ``preview_diff`` / ``produced_change_ref`` are filled later by the executor
#: (L7b-2), so they are NOT required at proposal time (a plan-only proposal is valid).
_CHANGE_PAYLOAD_FIELD: dict[ChangeKind, str] = {
    ChangeKind.METRIC_VIEW_SQL: "sql",
    ChangeKind.SKILL_DIFF: "diff",
    ChangeKind.INSTRUCTION_DIFF: "diff",
    ChangeKind.EVOLVED_BODY_REF: "evolved_body_ref",
    ChangeKind.REVERT_REF: "revert_target",
    ChangeKind.AGENT_TASK_PLAN: "plan",
}


class _Model(BaseModel):
    """Base for the proposal models: forbid unknown fields so drift is loud."""

    model_config = ConfigDict(extra="forbid")


class LocalApplySpec(_Model):
    """Immutable instructions for applying a reviewed artifact on the user's machine."""

    schema_version: str = "ail.local_apply/v1"
    target_kind: LocalApplyTargetKind
    target_path: str
    artifact_uri: str
    artifact_path: str
    artifact_field: str = "evolved_skill_body"
    baseline_sha256: str
    candidate_sha256: str
    validation_command: list[str] = Field(min_length=1)
    validation_timeout_seconds: int = Field(default=600, ge=1, le=3600)
    mlflow_run_id: str
    reviewer_experiment_id: str
    holdout_evolved_savings_pct: float | None = None
    holdout_seed_savings_pct: float | None = None
    holdout_savings_delta_pct: float | None = None
    holdout_task_ids: list[str] = Field(default_factory=list)

    @field_validator("target_path")
    @classmethod
    def _relative_target_path(cls, value: str) -> str:
        text = value.strip().replace("\\", "/")
        path = PurePosixPath(text)
        if not text or path.is_absolute() or text in {".", ".."} or ".." in path.parts:
            raise ValueError("target_path must be a non-empty project-relative path without '..'")
        return str(path)

    @field_validator(
        "artifact_uri",
        "artifact_path",
        "artifact_field",
        "mlflow_run_id",
        "reviewer_experiment_id",
    )
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("local-apply artifact fields must be non-empty")
        return value.strip()

    @field_validator("baseline_sha256", "candidate_sha256")
    @classmethod
    def _sha256(cls, value: str) -> str:
        digest = value.strip().lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("local-apply hashes must be lowercase SHA-256 hex digests")
        return digest

    @field_validator("validation_command")
    @classmethod
    def _command(cls, value: list[str]) -> list[str]:
        command = [str(part).strip() for part in value]
        if not command or any(not part for part in command):
            raise ValueError("validation_command must contain non-empty argv entries")
        return command


class TriggerSignal(_Model):
    """The **why**: the feedback that fired, with trace references.

    Names the *evidence* the proposal rests on (``docs/LOOP_CONTROLLER.md`` —
    "each rule names the evidence it fired on, so the proposal carries a defensible
    why"). ``trace_refs`` point back into the cohort so a reviewer verifies the
    trigger rather than trusting it.

    Args:
        kind: Which signal fired.
        summary: Human-readable statement of the trigger.
        metric: The objective metric or judged dimension the signal concerns, when
            applicable.
        observed_value: The measured value that fired the rule (e.g. a judge score).
        threshold: The bar it was compared against (e.g. the goal's guardrail
            threshold), when applicable.
        n_traces: Recurrence — how many distinct traces exhibited the signal.
        trace_refs: The trace ids that justify the trigger.
        judge_name: The judge whose dimension fired (for a judge-dimension signal);
            this is the judge whose trust must certify the proposal.
        asset_type: The recommended asset type (for an RLM-recommended-asset signal).
        source_rank: The RLM cohort-roll-up rank of the recommendation, when applicable.
    """

    kind: TriggerKind
    summary: str
    metric: str | None = None
    observed_value: float | None = None
    threshold: float | None = None
    n_traces: int = 0
    trace_refs: list[str] = Field(default_factory=list)
    judge_name: str | None = None
    asset_type: str | None = None
    source_rank: int | None = None


class ProposedChange(_Model):
    """The **what**: the concrete change, in exactly one form per :class:`ChangeKind`.

    A change must carry its payload — the matching field is required and non-empty
    (a proposal never carries an empty/fabricated change). The controller does not
    introspect the agent itself; the change body is produced by the (injectable)
    candidate builder that wraps :mod:`ail.optimize.assets` / GEPA.

    Args:
        kind: The form of the change.
        summary: Human-readable description for the reviewer.
        sql: The metric-view ``CREATE`` DDL (required for ``METRIC_VIEW_SQL``).
        diff: The skill/instruction diff (required for ``SKILL_DIFF`` /
            ``INSTRUCTION_DIFF``).
        evolved_body_ref: A reference (URI / artifact path) to the GEPA-evolved
            body (required for ``EVOLVED_BODY_REF``) — the body itself lives in the
            prompt registry / candidate artifact, not inlined here.
        revert_target: The version/asset to revert to (required for ``REVERT_REF``).
        plan: For an ``AGENT_TASK_PLAN`` change — the **NL intended change + why**,
            drawn from the evidence (**required** for that kind; the reviewer reads it
            to understand what the executor intends before any concrete change exists).
        preview_diff: For an ``AGENT_TASK_PLAN`` change — the **concrete produced
            change preview** (a diff / change-set) the human reviews before approving.
            ``None`` until the executor (L7b-2) produces it in a sandbox pre-approval;
            L7b-1 only defines the slot. The user decision (``docs/PRODUCT_ARCHITECTURE.md``
            §7): the human previews the *real* change, not just the NL plan.
        produced_change_ref: For an ``AGENT_TASK_PLAN`` change — an L6 snapshot / UC
            Volume ref to the produced change-set, used to **commit on approval**.
            ``None`` until the executor (L7b-2) fills it; L7b-1 only defines the slot.
    """

    kind: ChangeKind
    summary: str
    sql: str | None = None
    diff: str | None = None
    evolved_body_ref: str | None = None
    revert_target: str | None = None
    plan: str | None = None
    preview_diff: str | None = None
    produced_change_ref: str | None = None
    local_apply_spec: LocalApplySpec | None = None

    @model_validator(mode="after")
    def _require_payload(self) -> ProposedChange:
        field = _CHANGE_PAYLOAD_FIELD[self.kind]
        value = getattr(self, field)
        # A payload must carry actual content: reject missing, empty, AND whitespace-only.
        # Every payload field is an optional string, so the stripped check applies
        # uniformly — a whitespace-only diff / SQL / ref / target is as meaningless as an
        # empty one, and a whitespace-only ``plan`` would make an AGENT_TASK carry no real
        # intended-change text (and key ``derive_proposal_id`` on whitespace).
        if value is None or not value.strip():
            raise ValueError(
                f"ProposedChange of kind {self.kind.value!r} must set a non-empty "
                f"{field!r}; refusing an empty change (fail-closed)."
            )
        return self


class ProofSummary(_Model):
    """The **proof**: the frozen-suite objective delta with correctness held.

    Sourced from the comparison harness aggregate
    (:class:`~ail.optimize.phase2.Phase2Artifact`, built per task from
    :class:`~ail.compare.contract.ComparisonResult`). Realized savings are summed
    over PROMOTE (objective-met + correctness-held) tasks only — a blocked or
    crashed task's token delta is never counted as a win. Positive savings = the
    candidate improved the objective.

    :attr:`proved_improvement` is ``True`` iff at least one task PROMOTEd;
    :attr:`correctness_held` adds that no task regressed correctness. The
    controller emits a proposal only when **both** hold (fail-closed) — this object
    is the record of that proof, not the gate itself.
    """

    objective_metric: str
    proved_improvement: bool = False
    correctness_held: bool = False
    realized_savings_absolute: float = 0.0
    realized_savings_pct: float | None = None
    n_promote: int = 0
    n_block: int = 0
    n_errored: int = 0
    suite_content_hash: str = ""
    suite_version: str = ""

    @classmethod
    def from_phase2_artifact(cls, artifact: Phase2Artifact) -> ProofSummary:
        """Extract the proof headline from a frozen-suite comparison artifact.

        ``proved_improvement`` requires at least one PROMOTE task; ``correctness_held``
        additionally requires no task to have **regressed** correctness
        (``L1Outcome.REGRESSED``) — the same correctness-held rule
        :mod:`ail.publish_versions` applies to the version comparison.
        """
        any_regressed = any(o.l1_outcome is L1Outcome.REGRESSED for o in artifact.outcomes)
        return cls(
            objective_metric=artifact.objective_metric,
            proved_improvement=artifact.n_promote > 0,
            correctness_held=artifact.n_promote > 0 and not any_regressed,
            realized_savings_absolute=artifact.realized_token_savings_absolute,
            realized_savings_pct=artifact.realized_token_savings_pct,
            n_promote=artifact.n_promote,
            n_block=artifact.n_block,
            n_errored=artifact.n_errored,
            suite_content_hash=artifact.suite_content_hash,
            suite_version=artifact.suite_version,
        )


class GateStatus(_Model):
    """The **gate status**: readiness tier + certifying-judge agreement + coverage.

    Sourced from :class:`~ail.readiness.contract.ReadinessStatus`. ``gated`` is the
    controller's combined verdict (readiness wall cleared *and* — for a
    judge-dependent trigger — the certifying judge is trusted); a proposal is only
    emitted when ``gated`` is ``True``. The component figures travel with it so the
    reviewer sees *why* the state is (or is not) gated, never an opaque flag.
    """

    readiness_tier: str
    can_prove_improvement: bool = False
    judge_agreement: float | None = None
    scored_coverage: float = 0.0
    n_distrusted_judges: int = 0
    gated: bool = False
    reasons: list[str] = Field(default_factory=list)

    @classmethod
    def from_readiness(
        cls,
        readiness: ReadinessStatus,
        *,
        gated: bool,
        reasons: list[str],
        judge_name: str | None = None,
    ) -> GateStatus:
        """Project a :class:`ReadinessStatus` onto the proposal's gate payload.

        ``judge_name`` (the trigger's certifying judge, when any) selects which
        judge's agreement rate to surface; ``gated``/``reasons`` are the
        controller's combined verdict (readiness + judge trust).
        """
        agreement: float | None = None
        if judge_name is not None:
            jh = next((j for j in readiness.eval_health.judges if j.judge_name == judge_name), None)
            agreement = jh.agreement_rate if jh is not None else None
        return cls(
            readiness_tier=readiness.tier.value,
            can_prove_improvement=readiness.can_prove_improvement,
            judge_agreement=agreement,
            scored_coverage=readiness.eval_health.scored_coverage,
            n_distrusted_judges=readiness.eval_health.n_distrusted_judges,
            gated=gated,
            reasons=list(reasons),
        )


class ProposedAction(_Model):
    """One human-gated proposed change — the controller's sole output artifact.

    Construction is fully validated: the :attr:`change` form must match the
    :attr:`action_kind` (so a proposal cannot claim one action while carrying
    another's change). The controller only ever builds these with
    :attr:`ProposalStatus.PENDING`; lane 3 advances the status on approval/apply.

    The record is deliberately *inert*: it carries the change body but no apply
    capability. Nothing here registers a prompt, sets a champion alias, or runs a
    ``CREATE`` — those happen only on human approval in lane 3.
    """

    schema_version: str = SCHEMA_VERSION
    proposal_id: str
    agent_name: str
    experiment_id: str
    action_kind: ActionKind
    risk_class: RiskClass
    status: ProposalStatus = ProposalStatus.PENDING
    objective_metric: str
    goal_cohort: str
    trigger: TriggerSignal
    change: ProposedChange
    proof: ProofSummary | None = None
    gate_status: GateStatus
    created_at: str | None = None  # ISO-8601
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_change_matches_action(self) -> ProposedAction:
        expected = _ACTION_CHANGE_KIND[self.action_kind]
        if self.change.kind is not expected:
            raise ValueError(
                f"action_kind {self.action_kind.value!r} requires a change of kind "
                f"{expected.value!r}, got {self.change.kind.value!r}."
            )
        return self


def derive_proposal_id(*, agent_name: str, action_kind: ActionKind, change: ProposedChange) -> str:
    """A stable id for a proposal, derived from its agent + action + change content.

    Deterministic so re-running a cycle that decides the *same* change yields the
    *same* id — the publish step keys on ``(agent_name, proposal_id)``, so an
    idempotent id means a re-publish replaces the same row rather than duplicating
    it. Two materially different changes (different SQL/diff/ref/target) hash to
    different ids.
    """
    # ``plan`` is the AGENT_TASK's content-identifying field (its ``sql``/``diff``/refs
    # are all None), so include it — else two different agent-task plans would collide on
    # the same id. ``preview_diff`` / ``produced_change_ref`` are deliberately excluded:
    # the executor (L7b-2) fills them *after* the id is minted, so keying on them would
    # move a proposal's identity out from under the row it already published.
    payload = " ".join(
        [
            agent_name,
            action_kind.value,
            change.kind.value,
            change.sql or "",
            change.diff or "",
            change.evolved_body_ref or "",
            change.revert_target or "",
            change.plan or "",
            change.local_apply_spec.model_dump_json() if change.local_apply_spec else "",
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
