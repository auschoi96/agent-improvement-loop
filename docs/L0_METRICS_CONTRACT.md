# L0 Metrics Output Contract

**Status:** stable · **Schema version:** `l0.metrics/v1`

This is the contract a downstream consumer (the L0 leaderboard UI, Wave 1.5)
reads. It is produced by `ail.metrics.l0_deterministic.compute_l0(...)` and
serialized verbatim by pydantic (`L0MetricsReport.model_dump_json()`), so the
JSON shape below **is** the source of truth — the pydantic models in
`src/ail/metrics/contract.py` define it and round-trip it.

## What "L0" guarantees

Every value is **deterministic** — derived mechanically from a
`NormalizedTrace` (token counts, timestamps, tool spans). No model is in the
loop, so nothing here can be inflated by the agent under test. The single place
judgement enters is **cost**, which needs a model→price mapping; cost is always
accompanied by a `priced` flag and `flags`/`pricing_flags` so an unknown or
approximate price is surfaced, never silently fabricated.

## Versioning

`schema_version` is `l0.metrics/v1`. Additive, backward-compatible fields bump
the minor; any breaking shape change bumps the major. Consumers should read
`schema_version` and refuse a major they don't understand. Models set
`extra="forbid"`, so an unknown field is a hard error (drift is loud, not
silent).

## Top-level shape — `L0MetricsReport`

```jsonc
{
  "schema_version": "l0.metrics/v1",
  "experiment_id": "660599403165942",
  "generated_at": "2026-06-29T08:26:32.208703+00:00",  // ISO-8601, caller-supplied
  "n_traces": 90,
  "aggregate": { /* AggregateMetrics */ },
  "by_model":    [ /* GroupMetrics, sorted by total_tokens desc */ ],
  "by_producer": [ /* GroupMetrics */ ],
  "by_status":   [ /* GroupMetrics */ ],
  "traces":      [ /* TraceMetrics, sorted by total_tokens desc */ ],
  "pricebook":   [ /* PriceBookEntry */ ],
  "pricing_flags": [ "...", "..." ]   // corpus-level cost caveats
}
```

### `TraceMetrics` (one per trace, in `traces`)

```jsonc
{
  "trace_id": "trace:/austin_choi_omni_agent_catalog.mlflow_traces.cc/a9b23c23...",
  "producer": "claude_code",          // or null when undetected
  "model": "claude-opus-4-8",         // or null when not recorded
  "session_id": null,
  "status": "OK",                     // OK | ERROR | IN_PROGRESS | UNKNOWN
  "request_time": "2026-06-...T...Z", // ISO-8601 or null
  "duration_seconds": 33095.088,      // or null
  "tokens":  { /* TokenBreakdown */ },
  "cost":    { /* CostBreakdown */ },
  "total_tool_calls": 60,
  "tool_counts": { "Bash": 35, "Read": 5, "Write": 6, ... },
  "redundancy": { /* ToolRedundancy */ }
}
```

### `TokenBreakdown`

```jsonc
{
  "input_tokens": 868255,
  "output_tokens": 74674,
  "total_tokens": 942929,             // producer-preferred total (may fold cache in differently)
  "cache_creation_input_tokens": 0,
  "cache_read_input_tokens": 0,
  "cache_total_tokens": 0
}
```

In an aggregate / group, each field is the **sum** across the member traces.

### `CostBreakdown` (per trace)

```jsonc
{
  "input_usd": 4.341275,
  "output_usd": 1.86685,
  "cache_write_usd": 0.0,
  "cache_read_usd": 0.0,
  "total_usd": 6.208125,
  "priced": true,                     // false => model missing/uncovered; all dollars 0.0
  "flags": []                         // e.g. "model 'X' is not in the price book; cost not estimated"
}
```

Cost is computed from the **component** token counts priced separately, never
from a blended total. `priced: false` means the dollars are deliberately `0.0`
and `flags` says why — tokens are still counted.

### `ToolRedundancy` (per trace; also summed in `aggregate`)

```jsonc
{
  "total_tool_calls": 60,
  "distinct_tool_calls": 60,
  "redundant_tool_calls": 0,
  "redundancy_rate": 0.0,             // STRICT: redundant/total over byte-identical calls
  "repeated_calls": [ /* RepeatedCall, count>=2, sorted desc, capped */ ]
}
```

Two distinct views — this matters for how you read the numbers:

- **`redundancy_rate`** is the **strict, un-gameable** rate: it only counts
  calls whose tool name *and* full arguments are byte-identical.
- **`repeated_calls`** is the **diagnostic**: identities that recurred,
  where "identity" is looser than exact arguments:

  | `signature_kind` | how identity is keyed | catches |
  |---|---|---|
  | `path` | file tools (`Read`/`Edit`/`Write`/`MultiEdit`/`NotebookEdit`) keyed on `file_path` | "the same file was read/edited N times" |
  | `shell` | shell tools keyed on the normalized command prologue (first line, per-session scratch UUIDs collapsed to `<id>`) | "the same `cd`/setup boilerplate was re-run N times" |
  | `args`  | every other tool keyed on exact arguments | identical repeated calls |

  ```jsonc
  // RepeatedCall
  { "tool": "Bash", "identity": "cd /Users/.../agent-framework", "count": 27, "signature_kind": "shell" }
  ```

### `AggregateMetrics`

```jsonc
{
  "n_traces": 90,
  "tokens": { /* TokenBreakdown, summed */ },
  "token_stats": { "count": 90, "min": ..., "median": 18828.5, "mean": 84884.3, "p90": 242673.9, "max": 942929 },
  "cost": { /* CostAggregate */ },
  "total_tool_calls": 1149,
  "redundancy": { /* ToolRedundancy: corpus totals + top repeated_calls across traces */ },
  "status_counts": { "OK": 90 }
}
```

`token_stats` is the distribution of per-trace `total_tokens` — the L0 headline
for this corpus is that it is **bimodal** (low median, heavy tail), so the
distribution is first-class, not just the sum.

### `CostAggregate`

```jsonc
{
  "input_usd": ..., "output_usd": ..., "cache_write_usd": ..., "cache_read_usd": ...,
  "total_usd": 61.356799,             // sums ONLY priced traces
  "priced_traces": 88,
  "unpriced_traces": 2,               // counted but excluded from total_usd
  "flags": [ "2 trace(s) unpriced (model missing or not in price book); ..." ]
}
```

### `GroupMetrics` (each entry of `by_model` / `by_producer` / `by_status`)

```jsonc
{ "key": "claude-opus-4-8", "n_traces": 84, "tokens": {...}, "cost": {...}, "total_tool_calls": ... }
```

### `PriceBookEntry` (each entry of `pricebook`)

```jsonc
{
  "model": "claude-opus-4-8",
  "input_usd_per_mtok": 5.0,
  "output_usd_per_mtok": 25.0,
  "cache_write_usd_per_mtok": 6.25,   // 1.25x input (documented 5-min-TTL multiplier)
  "cache_read_usd_per_mtok": 0.5,     // 0.1x input
  "source": "claude-api skill model pricing table (cached 2026-06-04)",
  "confidence": "high",               // high | medium | low
  "notes": "cache rates derived from documented 5-min-TTL multipliers (write 1.25x, read 0.1x)"
}
```

## Pricing: honesty rules

- **Base input/output rates** come from the Claude API pricing reference
  (sourced, not invented); see each `PriceBookEntry.source`.
- **Cache rates** are derived from Anthropic's documented prompt-caching
  multipliers (read ≈ 0.1× input, write ≈ 1.25× input at the 5-minute TTL).
  MLflow traces do not record which cache TTL was used, so the 5-minute rate is
  assumed and **flagged** whenever cache tokens are present (a 1-hour TTL would
  be 2.0× input).
- **Unknown models are never guessed.** A model absent from the price book is
  left unpriced (`priced: false`), its tokens still counted, and it is named in
  `pricing_flags`. Override or extend pricing by passing a custom `pricebook`
  to `compute_l0(...)`.
- Treat dollar figures as **estimates**; verify against live pricing before any
  billing decision (also stated in `pricing_flags`).

## Producing the report

```python
from ail.ingest.mlflow_source import MLflowTraceSource
from ail.metrics.l0_deterministic import compute_l0

traces = MLflowTraceSource(...).fetch_traces(experiment_id="660599403165942")
report = compute_l0(traces, experiment_id="660599403165942", generated_at="...")
report.model_dump_json(indent=2)   # <- the contract bytes a UI consumes
```

Or the single entrypoint that also reproduces Example 1 and writes artifacts:

```bash
python -m ail.metrics.report --experiment 660599403165942 --out-dir artifacts
```

> **Auth note (reference workspace).** The reference experiment is backed by a
> UC table read through MLflow 3's v4 trace REST store, which on this workspace
> rejects OAuth-profile credentials for the span `batchGet`. The reliable path
> is explicit token auth:
> `export DATABRICKS_HOST=https://<workspace-host>` and
> `export DATABRICKS_TOKEN=$(databricks auth token -p dais-demo | jq -r .access_token)`,
> then run the command above. `_build_source` detects those env vars and uses
> them; otherwise it falls back to the `--profile` CLI profile.
