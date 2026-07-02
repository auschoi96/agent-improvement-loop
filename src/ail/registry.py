"""The **agent registry** — the observability layer's primary key onto agents.

The multi-agent observability app (``docs/OBSERVABILITY_APP.md``) is built on one
decision: **one MLflow experiment per agent**. The registry is the typed,
config-driven map that makes that decision usable — it maps a friendly, unique
``agent_name`` to that agent's dedicated ``experiment_id`` (plus an optional judge
config and an optional within-experiment tag filter). "Specify the agent you're
tracking" in the app = a registry entry; "distinguish between agents" = the
registry lists them, each carrying its own experiment.

Design rules (mirroring the rest of the framework):

* **Typed + JSON/YAML round-trippable.** :class:`Agent` / :class:`AgentRegistry`
  are pydantic v2 models that forbid unknown fields, so a config typo is loud and
  the registry can be persisted to (and read back from) a UC table or a YAML file
  without custom (de)serialization.
* **Dependency-light.** Only pydantic + :mod:`ail.cohorts` (for the within-
  experiment :class:`~ail.cohorts.TagFilter`/:class:`~ail.cohorts.Cohort`). No
  MLflow, no SQL — the *publish* tier (:mod:`ail.publish_versions`) is what writes
  the registry to a UC table the app reads; this module just defines and loads it.
* **A seed, in code.** :data:`DEFAULT_REGISTRY` is the committed seed (the current
  Claude Code agent). ``config/agents.yaml`` is its YAML mirror for operators; a
  test keeps the two in sync.

The agent boundary is the **experiment**; *within* an experiment, tag-defined
cohorts (notably :attr:`Agent.tag_filter` and the per-version
``ail.agent_version`` slices the live comparison hangs on) drop to sub-selection —
see :mod:`ail.cohorts`.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ail.cohorts import TAG_AGENT, Cohort

__all__ = [
    "TAG_AGENT_VERSION",
    "Agent",
    "AgentRegistry",
    "DEFAULT_REGISTRY",
    "CLAUDE_CODE_EXPERIMENT_ID",
    "load_registry",
]

#: The current Claude Code agent's dedicated MLflow experiment (the reference
#: experiment this framework was built against).
CLAUDE_CODE_EXPERIMENT_ID = "660599403165942"

#: Conventional tag key naming which *version* of an agent emitted a trace
#: (e.g. ``ail.agent_version = v1-token-efficiency-skill``). Versions are the unit
#: the live baseline-vs-new comparison and the lineage timeline hang on, and they
#: map 1:1 to prompt-registry versions (``docs/OBSERVABILITY_APP.md``). A
#: convention, not a requirement — arbitrary user keys remain first-class via
#: :class:`ail.cohorts.TagFilter`.
TAG_AGENT_VERSION = "ail.agent_version"


class _Config(BaseModel):
    """Base for registry models: forbid unknown fields so a config typo is loud."""

    model_config = ConfigDict(extra="forbid")


class Agent(_Config):
    """One registered agent: the app's primary key onto a dedicated experiment.

    Args:
        agent_name: Friendly, unique name — the app's primary key and the value a
            trace carries under the ``ail.agent`` convention tag.
        experiment_id: This agent's dedicated MLflow experiment id (the agent
            boundary; one experiment per agent).
        description: Optional human context shown in the agent list / switcher.
        judge_config: Opaque, optional per-agent L2 judge/scorer configuration.
            Kept as a free-form mapping here (Phase A only needs to *carry* it);
            the judges lane interprets it. ``None`` means "no per-agent judges
            configured yet".
        tag_filter: Optional within-experiment sub-selection as ``{tag_key:
            tag_value}`` equality clauses (AND'd). ``None`` means the whole
            experiment is the agent's cohort. Built into a :class:`ail.cohorts.Cohort`
            by :meth:`cohort`.
        target_workspace: The path / repo the **open-ended executor** (a LATER lane,
            L7b-2) will edit and snapshot — the target agent's own source. User-provided
            and **optional at the model level** (``None`` cleanly represents "not
            configured yet", so a registry entry is valid before the executor is wired),
            but **REQUIRED for the executor**: an ``AGENT_TASK`` cannot be executed
            against an agent with no ``target_workspace``. L7b-1 only *carries* this
            field; it neither runs the executor nor validates the path exists.
    """

    agent_name: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    description: str = ""
    judge_config: dict[str, Any] | None = None
    tag_filter: dict[str, str] | None = None
    target_workspace: str | None = None

    def cohort(self) -> Cohort:
        """The :class:`~ail.cohorts.Cohort` selecting this agent's traces.

        Uses :attr:`tag_filter` when set (AND'd equality clauses), otherwise the
        ``ail.agent = <agent_name>`` convention. The cohort is identity only — a
        downstream readiness/metrics step applies it to actual traces; an empty
        match is the legitimate *collecting* state, not an error.
        """
        if self.tag_filter:
            return Cohort.from_tags(self.agent_name, dict(self.tag_filter))
        return Cohort.from_tag(self.agent_name, TAG_AGENT, self.agent_name)


class AgentRegistry(_Config):
    """The set of registered agents, keyed by unique :attr:`Agent.agent_name`.

    Validates uniqueness of names at construction so the app's primary key can
    never be ambiguous. Iterable and indexable by name.
    """

    agents: list[Agent] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        seen: set[str] = set()
        dupes: set[str] = set()
        for agent in self.agents:
            if agent.agent_name in seen:
                dupes.add(agent.agent_name)
            seen.add(agent.agent_name)
        if dupes:
            raise ValueError(f"duplicate agent_name(s) in registry: {', '.join(sorted(dupes))}")

    def get(self, agent_name: str) -> Agent:
        """Return the :class:`Agent` named ``agent_name`` or raise ``KeyError``."""
        for a in self.agents:
            if a.agent_name == agent_name:
                return a
        have = ", ".join(self.names()) or "<empty>"
        raise KeyError(f"no agent named {agent_name!r} in registry (have: {have})")

    def names(self) -> list[str]:
        """Registered agent names, in registry order."""
        return [a.agent_name for a in self.agents]

    def __iter__(self) -> Iterator[Agent]:  # type: ignore[override]
        return iter(self.agents)

    def __len__(self) -> int:
        return len(self.agents)


#: The committed seed registry: the current Claude Code agent. ``config/agents.yaml``
#: is the YAML mirror (kept in sync by a test).
DEFAULT_REGISTRY = AgentRegistry(
    agents=[
        Agent(
            agent_name="claude_code",
            experiment_id=CLAUDE_CODE_EXPERIMENT_ID,
            description="Claude Code CLI sessions (the reference agent).",
        )
    ]
)


def load_registry(path: str | Path | None = None) -> AgentRegistry:
    """Load an :class:`AgentRegistry` from a YAML/JSON file, or the committed seed.

    ``path is None`` returns :data:`DEFAULT_REGISTRY` (the in-code seed). When a
    ``path`` is given it is parsed as YAML (a superset of JSON) and validated
    against the typed contract — an unknown field or a duplicate name fails loud.
    The expected shape is ``{"agents": [{"agent_name": ..., "experiment_id": ...},
    ...]}``.
    """
    if path is None:
        return DEFAULT_REGISTRY
    import yaml

    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text) or {}
    return AgentRegistry.model_validate(data)
