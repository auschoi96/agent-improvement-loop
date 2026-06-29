# Cohorts

**Status:** stable · **Module:** `ail.cohorts` · **MLflow integration:** `ail.ingest.mlflow_source`

A **cohort** is a *named selection of traces* defined by a **tag filter**. One
MLflow experiment routinely holds traces from several agents or deployments;
cohorts segment that one experiment into per-agent / per-deployment lanes that
can be measured apart.

Cohorts are **orthogonal** to the three evaluation pools in `ail.pools`
(task-suite / alignment / human-anchor). Pools are the disjoint *evaluation
wall*; cohorts are an overlapping *slice-by-tag* view. A trace can be in many
cohorts at once; it is in exactly one pool.

## Tags are the user's, not ours

The **primary path is the user tagging traces in the MLflow UI** with whatever
keys they like. Cohorts *respect* those arbitrary keys — `squad`, `ticket`,
`experiment-arm`, anything. We never require our own keys.

For callers who want a tidy default, there is a documented **convention
namespace**, `ail.*`:

| Tag key      | Meaning                                  | Convenience constructor       |
|--------------|------------------------------------------|-------------------------------|
| `ail.agent`  | which agent/producer emitted the trace   | `Cohort.by_agent("claude_code")` |
| `ail.cohort` | the cohort a trace belongs to            | `Cohort.by_cohort_tag("nightly")` |

These are conventions, not requirements. `ail.agent`/`ail.cohort` get
one-line constructors; any other key uses `Cohort.from_tag` / `Cohort.from_tags`.

## Tag filter semantics

A `TagFilter` is a list of `TagClause`s that are **AND'd** — a trace matches iff
*every* clause matches. Within one clause, the accepted values are **OR'd**:

- **equality** — `"prod"` → tag must equal `prod`.
- **value-in-set** — `{"prod", "staging"}` → tag must be one of these.
- **presence-only** — `None` → tag must exist with *any* value.

An **empty filter** (no clauses) matches every trace.

```python
from ail.cohorts import Cohort

# one agent (uses the ail.agent convention)
Cohort.by_agent("claude_code")

# arbitrary user keys, ANDed; env is a value-in-set
Cohort.from_tags("prod-claude", {"ail.agent": "claude_code", "env": {"prod", "staging"}})

# a user's own key only — no ail.* anywhere
Cohort.from_tag("alpha-squad", "squad", "alpha")

# presence-only: any trace that has been reviewed at all
Cohort.from_tag("reviewed", "reviewed", None)
```

`cohort.select(traces)` returns the matching subset (order preserved);
`cohort.matches(trace)` tests one trace.

## Tag-aware ingestion (additive)

`MLflowTraceSource` gains two **additive** methods; the existing
`iter_traces` / `get_trace` / `fetch_traces` are untouched and behave exactly as
before.

```python
from ail.ingest.mlflow_source import MLflowTraceSource
from ail.cohorts import Cohort

source = MLflowTraceSource(profile="dais-demo")
traces = source.fetch_cohort_traces(
    Cohort.by_agent("claude_code"),
    experiment_id="660599403165942",
)
```

**Pushdown + post-filter.** The cohort's *equality* clauses are pushed into
`mlflow.search_traces` as a filter string (e.g.
``tags.`ail.agent` = 'claude_code'``), AND'd onto any caller-supplied
`filter_string`, so the backend narrows the scan. That pushdown is a correct
*prefilter* but not always complete — value-in-set and presence-only clauses
(and any value unsafe to embed in a filter literal) are **not** pushed. So every
returned trace is additionally checked with `cohort.matches(...)` in memory: the
**post-filter is the source of truth**, and the yielded set is always exactly
the cohort.

> **`max_results` caveat.** It bounds the traces *scanned* by the backend. When
> the filter is only partially pushed down, the post-filter may drop some of
> those, so the number returned can be fewer than `max_results`. Callers needing
> a guaranteed count should over-fetch or page.

## Per-cohort L0 metrics (additive wrapper)

`ail.metrics.cohort` wraps the existing `compute_l0` — it does **not** rewrite
the metrics engine. It selects the cohort's traces and feeds them to
`compute_l0`, so a per-cohort report *is* an ordinary
[`L0MetricsReport`](./L0_METRICS_CONTRACT.md) over a filtered subset.

```python
from ail.metrics.cohort import compute_cohort_l0, compute_l0_by_cohort
from ail.cohorts import Cohort

report = compute_cohort_l0(traces, Cohort.by_agent("claude_code"), generated_at=now)

reports = compute_l0_by_cohort(
    traces,
    [Cohort.by_agent("claude_code"), Cohort.by_agent("codex")],
    generated_at=now,
)  # -> {"claude_code": L0MetricsReport, "codex": L0MetricsReport}
```

## A cohort with 0 traces is the *collecting / not-ready* state

An empty selection is **not an error**. It is the legitimate state of a
deployment that exists but hasn't produced enough traces yet:
`compute_cohort_l0` returns a valid report with `n_traces == 0`. **Readiness**
(how many traces, of what quality, before a cohort is "ready") is computed
**per cohort by a future module** — this module deliberately stops at giving
that module a clean input: a `name` plus a pure `select`.

## Applying tags programmatically (`apply_trace_tags`)

The **primary** way traces get cohort tags is the user, in the MLflow UI.
`apply_trace_tags(trace_ids, tags)` is a convenience for programmatic backfill:

```python
from ail.ingest.mlflow_source import apply_trace_tags

apply_trace_tags(
    ["tr-abc", "tr-def"],
    {"ail.agent": "claude_code", "ail.cohort": "nightly"},
    profile="dais-demo",
)
```

> **Write access.** This is the more privileged path. Writing trace tags on
> Databricks-managed MLflow needs workspace/warehouse **write** access (the
> trace store is UC-backed) that the orchestration identity running the loop may
> not hold — expect a `PermissionDenied` here even when the read path works.
> Treat a failure as "tag it in the UI instead", not as a loop-blocking error.
> A `client` is injectable for tests/custom auth; when omitted, one is built
> against the configured Databricks workspace.
