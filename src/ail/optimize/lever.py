"""The Phase-2 token-efficiency lever: skill-injection intervention + configs.

This wires the generated :mod:`token-efficient-execution <ail.optimize.assets>`
skill into the candidate-vs-baseline comparison machinery
(:func:`ail.compare.compare_candidate`) as a pair of named configs the runner and
the orchestrator drive:

* :data:`BASELINE` — **no asset**. The agent runs the task as-is.
* :data:`CANDIDATE` — **asset enabled**. A :class:`SkillInjectionIntervention`
  adds the token-efficiency skill so the agent stops re-reading files it has
  already read, batches related shell commands, and drops repeated ``cd``/setup
  boilerplate.

**Injection mechanism — skill, not tool.** The intervention augments only the
candidate task's ``system_prompt`` with the skill body; it changes **nothing**
else (same prompt, model, tools), so the comparison isolates the skill's effect.
This matches how the Claude Code adapter runs: it sets ``setting_sources=[]`` (no
ambient skill discovery) and injects context explicitly via the system prompt, so
this is the seam through which a skill actually reaches that agent. An MCP
"read-cache" tool was considered and deliberately **not** shipped: Claude Code's
built-in ``Read`` cannot be intercepted by an MCP server, so a separate cache tool
would need the agent to abandon ``Read`` for a differently-named tool it is not
biased to call — it would not reliably be used, and faking one would violate the
"honesty over a tool the agent won't invoke" rule. The behavioural skill drives
the same outcome (fewer redundant ``Read``/``Bash`` calls) through the channel the
agent already uses, and the L0 redundancy metric measures exactly that.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cache

from ail.compare.harness import Intervention
from ail.ingest.base import AgentTask
from ail.optimize.assets import TOKEN_EFFICIENCY_SKILL, SkillAsset, load_skill_asset

__all__ = [
    "SkillInjectionIntervention",
    "LeverConfig",
    "token_efficiency_skill",
    "token_efficiency_intervention",
    "BASELINE",
    "CANDIDATE",
]


@dataclass(frozen=True, kw_only=True)
class SkillInjectionIntervention(Intervention):
    """Inject a :class:`~ail.optimize.assets.SkillAsset` into a task's system prompt.

    Pure (per the :class:`~ail.compare.harness.Intervention` contract): returns a
    new :class:`~ail.ingest.base.AgentTask` with the skill appended to
    ``system_prompt`` and every other field untouched, never mutating its
    argument. Appending (rather than replacing) preserves any task-supplied system
    prompt, so the skill is additive context — the single controlled difference
    between baseline and candidate.
    """

    name: str = "skill-injection"
    skill: SkillAsset

    def apply(self, task: AgentTask) -> AgentTask:
        section = self.skill.as_system_prompt_section()
        existing = task.system_prompt or ""
        system_prompt = f"{existing}\n\n{section}" if existing.strip() else section
        return replace(task, system_prompt=system_prompt)


@dataclass(frozen=True, slots=True)
class LeverConfig:
    """A named comparison config: an optional intervention applied to the candidate.

    ``intervention is None`` is the **baseline** config (the agent runs the task
    with no asset); a non-``None`` intervention is the **candidate** config (asset
    enabled). The harness runs both arms of a single
    :func:`~ail.compare.compare_candidate` call from one candidate config — the
    baseline arm is the un-intervened task — so :data:`BASELINE` exists to name and
    record "no asset" explicitly for provenance, while :data:`CANDIDATE` carries
    the intervention the harness actually applies.
    """

    name: str
    intervention: SkillInjectionIntervention | None = None
    description: str = ""

    @property
    def asset_enabled(self) -> bool:
        """Whether this config applies an asset (the candidate) or not (baseline)."""
        return self.intervention is not None


@cache
def token_efficiency_skill() -> SkillAsset:
    """Load the token-efficiency :class:`~ail.optimize.assets.SkillAsset` (cached)."""
    return load_skill_asset(TOKEN_EFFICIENCY_SKILL)


def token_efficiency_intervention() -> SkillInjectionIntervention:
    """The CANDIDATE intervention: inject the token-efficiency skill."""
    return SkillInjectionIntervention(
        name="token-efficiency-skill",
        skill=token_efficiency_skill(),
    )


#: The BASELINE config — no asset; the agent runs the task unchanged.
BASELINE = LeverConfig(
    name="baseline-no-asset",
    intervention=None,
    description="No asset: the agent runs each Task-Suite task as-is.",
)

#: The CANDIDATE config — the token-efficiency skill injected into the system prompt.
CANDIDATE = LeverConfig(
    name="candidate-token-efficiency-skill",
    intervention=token_efficiency_intervention(),
    description=(
        "Token-efficiency skill enabled: instructs the agent to avoid re-reading "
        "files already read in-session, batch related shell commands, and drop "
        "repeated cd/setup boilerplate."
    ),
)
