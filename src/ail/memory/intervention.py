"""The advisory-memory intervention: inject accumulated learnings as context.

This is the Phase A-0 sibling of
:class:`ail.optimize.lever.SkillInjectionIntervention`. Where the skill lever
injects a single authored ``SKILL.md`` body, this injects a short **advisory**
block of *learnings* distilled from prior agent sessions (the RLM cohort roll-up;
see :mod:`ail.memory.source`) into the candidate task's ``system_prompt`` — and
**nothing else** — so a baseline-vs-candidate comparison isolates the effect of
the memory alone (the single controlled difference).

**Advisory, not mandatory.** The block is framed as suggestions the agent may
apply when relevant, not standing orders — this is a "prove value before
complexity" spike, and the learnings are aggregate hints, not verified rules.

**Fail-closed no-op.** With **no** learnings the intervention returns the task
**unchanged** (the same object), so an empty/absent memory source makes the
candidate byte-identical to the baseline. It can never invent a benefit: no
memory means no injected context, which the harness reads as no intervention.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ail.compare.harness import Intervention
from ail.ingest.base import AgentTask

__all__ = ["MemoryInjectionIntervention"]

#: The advisory preamble that heads the injected block. Framed as suggestions,
#: not orders (the learnings are aggregate hints from prior sessions, not
#: verified rules), so the agent applies them only when relevant.
_ADVISORY_PREAMBLE = (
    "The following are advisory learnings distilled from prior agent sessions. "
    "They are suggestions that may help you work more efficiently; apply them only "
    "when relevant to this task."
)


@dataclass(frozen=True, kw_only=True)
class MemoryInjectionIntervention(Intervention):
    """Inject advisory learnings into a task's system prompt.

    Pure (per the :class:`~ail.compare.harness.Intervention` contract): returns a
    new :class:`~ail.ingest.base.AgentTask` with the learnings appended to
    ``system_prompt`` and every other field untouched, never mutating its
    argument. Mirrors :class:`ail.optimize.lever.SkillInjectionIntervention` — the
    injection is **additive** context, the single controlled difference between
    baseline and candidate.

    ``learnings`` is an ordered, immutable tuple of pre-formatted advisory lines
    (built by :func:`ail.memory.source.build_memory_learnings`). An **empty**
    tuple makes :meth:`apply` a no-op that returns the task unchanged — the
    fail-closed contract: no memory ⇒ no intervention ⇒ identical to baseline.
    """

    name: str = "advisory-memory"
    learnings: tuple[str, ...] = ()

    def as_system_prompt_section(self) -> str:
        """Render the learnings as a system-prompt section, in a stable marker.

        The block is wrapped in a clear ``<learnings>`` marker (mirroring the
        skill lever's ``<skill>`` marker) so the injection is unambiguous and
        auditable in a captured trace.
        """
        bullets = "\n".join(self.learnings)
        return (
            f'<learnings source="prior-sessions">\n{_ADVISORY_PREAMBLE}\n\n{bullets}\n</learnings>'
        )

    def apply(self, task: AgentTask) -> AgentTask:
        # Fail-closed no-op: with no learnings there is nothing to inject, so the
        # candidate is byte-identical to the baseline (same object, unchanged).
        if not self.learnings:
            return task
        section = self.as_system_prompt_section()
        existing = task.system_prompt or ""
        system_prompt = f"{existing}\n\n{section}" if existing.strip() else section
        return replace(task, system_prompt=system_prompt)
