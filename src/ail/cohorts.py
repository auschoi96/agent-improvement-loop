"""First-class **cohorts**: named, tag-defined slices of one experiment's traces.

One MLflow experiment routinely holds traces from several agents or deployments.
A :class:`Cohort` is a *named selection* of those traces, defined entirely by a
**tag filter** over :attr:`~ail.ingest.base.NormalizedTrace.tags`. Tagging a
trace ``ail.agent = claude_code`` (or any user-chosen key/value) and pointing a
cohort at that tag is how the loop segments one experiment into per-agent /
per-deployment lanes that can be measured apart.

**Tags are the user's, not ours.** The primary path is the user tagging traces
in the MLflow UI with whatever keys they like; cohorts *respect* those arbitrary
keys. The :data:`ail.* <TAG_NAMESPACE>` constants below are a documented
*convention* — a tidy default for callers who want one — never a requirement.
:meth:`Cohort.by_agent` / :meth:`Cohort.by_cohort_tag` are conveniences over
that convention; :meth:`Cohort.from_tag` / :meth:`Cohort.from_tags` take any keys.

**Empty is a valid state.** A cohort that currently matches zero traces is the
*collecting / not-ready* state — the deployment exists but hasn't produced
enough traces yet. This module does not judge readiness; it just makes a cohort
a clean input (``name`` + a pure :meth:`Cohort.select`) that a future per-cohort
readiness module can consume.

This module is deliberately dependency-light (stdlib + the
:class:`~ail.ingest.base.NormalizedTrace` contract): it carries no MLflow or
agent-SDK import, so it is trivially usable anywhere. The MLflow read/write
integration (cohort-aware ingestion and the tag-write helper) lives in
:mod:`ail.ingest.mlflow_source`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from ail.ingest.base import NormalizedTrace

__all__ = [
    "TAG_NAMESPACE",
    "TAG_AGENT",
    "TAG_COHORT",
    "TagClause",
    "TagFilter",
    "Cohort",
]

#: The reserved convention namespace for cohort-related tag keys. A *convention*,
#: not a requirement — arbitrary user-defined keys are first-class everywhere.
TAG_NAMESPACE = "ail"
#: Conventional tag key naming which agent/producer emitted a trace
#: (e.g. ``ail.agent = claude_code``). Powers :meth:`Cohort.by_agent`.
TAG_AGENT = "ail.agent"
#: Conventional tag key naming the cohort a trace belongs to
#: (e.g. ``ail.cohort = nightly-regression``). Powers :meth:`Cohort.by_cohort_tag`.
TAG_COHORT = "ail.cohort"


# Tag values safe to embed directly in an MLflow filter string literal. The
# pushdown emits ``tags.`key` = 'value'`` with the value inside single quotes,
# so a value carrying a single quote or backslash could break (or be abused to
# inject into) the filter. Such values are not pushed down — they fall through
# to the in-memory post-filter, which has no such limitation.
def _pushdown_safe_value(value: str) -> bool:
    return "'" not in value and "\\" not in value


def _pushdown_safe_key(key: str) -> bool:
    # Keys are backtick-quoted in the filter; a backtick would break the quoting.
    return bool(key) and "`" not in key


def _clause(key: str, value: str | Iterable[str] | None) -> TagClause:
    """Build a :class:`TagClause` from the flexible mapping-value forms.

    * ``str`` -> single accepted value (equality).
    * iterable of ``str`` -> value-in-set (must be non-empty).
    * ``None`` -> presence-only (key must exist with any value).
    """
    if value is None:
        return TagClause(key)
    if isinstance(value, str):
        return TagClause(key, frozenset({value}))
    values = frozenset(str(v) for v in value)
    if not values:
        raise ValueError(
            f"tag filter clause for {key!r} has no values; pass None for a presence-only match"
        )
    return TagClause(key, values)


@dataclass(frozen=True, slots=True)
class TagClause:
    """A single tag predicate: a key plus the values that satisfy it.

    Membership in :attr:`values` is an OR (any one accepted value matches). An
    empty :attr:`values` means **presence-only**: the key must exist on the
    trace with any value at all.
    """

    key: str
    values: frozenset[str] = frozenset()

    def matches(self, tags: Mapping[str, str]) -> bool:
        """Whether ``tags`` satisfies this clause."""
        if self.key not in tags:
            return False
        if not self.values:
            return True
        return tags[self.key] in self.values


@dataclass(frozen=True, slots=True)
class TagFilter:
    """A conjunctive (AND) filter over trace tags.

    A trace matches iff **every** clause matches. An empty filter (no clauses)
    matches every trace — i.e. "the whole experiment".
    """

    clauses: tuple[TagClause, ...] = ()

    @classmethod
    def from_mapping(cls, spec: Mapping[str, str | Iterable[str] | None]) -> TagFilter:
        """Build a filter from a ``{key: value}`` mapping.

        Each value may be a single string (equality), an iterable of strings
        (value-in-set), or ``None`` (presence-only). All entries are AND'd.
        """
        return cls(tuple(_clause(key, value) for key, value in spec.items()))

    def matches(self, tags: Mapping[str, str]) -> bool:
        """Whether ``tags`` satisfies all clauses (vacuously true when empty)."""
        return all(clause.matches(tags) for clause in self.clauses)

    def to_mlflow_filter(self) -> str | None:
        """Express the equality-pushable subset as an MLflow trace filter string.

        Returns an ``AND``-joined filter (e.g.
        ``tags.`ail.agent` = 'claude_code'``) covering only the clauses that are
        single-valued *and* whose key/value are safe to embed literally. Multi-
        value, presence-only, and unsafe-literal clauses are **omitted** here on
        purpose: they are satisfied by the in-memory post-filter instead. The
        result is therefore a correct *prefilter* — it never excludes a trace the
        full filter would keep — usable to narrow ``mlflow.search_traces`` before
        :meth:`Cohort.select` enforces the complete filter. Returns ``None`` when
        nothing is pushable.
        """
        predicates = []
        for clause in self.clauses:
            if len(clause.values) != 1:
                continue
            (value,) = tuple(clause.values)
            if not _pushdown_safe_key(clause.key) or not _pushdown_safe_value(value):
                continue
            predicates.append(f"tags.`{clause.key}` = '{value}'")
        return " AND ".join(predicates) if predicates else None


@dataclass(frozen=True, slots=True)
class Cohort:
    """A named selection of traces defined by a :class:`TagFilter`.

    ``name`` identifies the cohort (e.g. an agent or deployment name) and is the
    key under which per-cohort metrics are reported. ``description`` is optional
    human context. The filter is applied to a trace's
    :attr:`~ail.ingest.base.NormalizedTrace.tags`.
    """

    name: str
    tag_filter: TagFilter
    description: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Cohort.name must be a non-empty string")

    # -- constructors ------------------------------------------------------

    @classmethod
    def from_tag(
        cls,
        name: str,
        key: str,
        value: str | Iterable[str] | None,
        *,
        description: str | None = None,
    ) -> Cohort:
        """Cohort matching a single tag ``key`` (equality, value-in-set, or presence).

        ``value`` follows the same forms as :meth:`TagFilter.from_mapping`: a
        string (equality), an iterable of strings (value-in-set), or ``None``
        (presence-only).
        """
        return cls(name, TagFilter((_clause(key, value),)), description)

    @classmethod
    def from_tags(
        cls,
        name: str,
        tags: Mapping[str, str | Iterable[str] | None],
        *,
        description: str | None = None,
    ) -> Cohort:
        """Cohort matching several AND'd tag clauses (arbitrary user keys welcome)."""
        return cls(name, TagFilter.from_mapping(tags), description)

    @classmethod
    def by_agent(
        cls,
        agent: str,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> Cohort:
        """Cohort over the ``ail.agent`` convention key (one agent per cohort).

        Defaults ``name`` to the agent value. A convenience over
        :meth:`from_tag` with :data:`TAG_AGENT`; nothing requires producers to
        emit this key.
        """
        return cls.from_tag(name or agent, TAG_AGENT, agent, description=description)

    @classmethod
    def by_cohort_tag(
        cls,
        value: str,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> Cohort:
        """Cohort over the ``ail.cohort`` convention key.

        Defaults ``name`` to the tag value. A convenience over :meth:`from_tag`
        with :data:`TAG_COHORT`.
        """
        return cls.from_tag(name or value, TAG_COHORT, value, description=description)

    # -- selection ---------------------------------------------------------

    def matches(self, trace: NormalizedTrace) -> bool:
        """Whether ``trace`` belongs to this cohort, per its tags."""
        return self.tag_filter.matches(trace.tags)

    def select(self, traces: Iterable[NormalizedTrace]) -> list[NormalizedTrace]:
        """Return the subset of ``traces`` belonging to this cohort (order preserved).

        An empty result is the legitimate *collecting / not-ready* state, not an
        error.
        """
        return [trace for trace in traces if self.matches(trace)]

    def to_mlflow_filter(self) -> str | None:
        """The cohort's equality-pushdown filter string (see :meth:`TagFilter.to_mlflow_filter`)."""
        return self.tag_filter.to_mlflow_filter()
