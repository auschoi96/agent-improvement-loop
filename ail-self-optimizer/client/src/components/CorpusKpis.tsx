import { useAnalyticsQuery, Card, CardContent, Badge, Skeleton } from '@databricks/appkit-ui/react';
import { sql } from '@databricks/appkit-ui/js';
import { useMemo, type ReactNode } from 'react';
import { fmtInt, fmtUsd, fmtPct } from '../lib/formatters';
import { spansTableFromAnnotations, traceFreshness } from '../lib/traces';

function Kpi({ label, value, sub }: { label: string; value: string; sub?: ReactNode }) {
  return (
    <Card className="shadow-sm">
      <CardContent className="p-4">
        <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{label}</div>
        <div className="mt-1 text-2xl font-bold text-foreground tabular-nums">{value}</div>
        {sub && <div className="mt-1 text-xs text-muted-foreground">{sub}</div>}
      </CardContent>
    </Card>
  );
}

function TraceKpi({ otelSpansTable, snapshotCount }: { otelSpansTable: string; snapshotCount: number }) {
  const params = useMemo(() => ({ otel_spans_table: sql.string(otelSpansTable) }), [otelSpansTable]);
  const { data, loading, error } = useAnalyticsQuery('live_trace_summary', params);
  const liveCount = Number(data?.[0]?.live_trace_count);

  if (loading) {
    return <Kpi label="Traces" value={fmtInt(snapshotCount)} sub="Checking live OTEL…" />;
  }
  if (error || !Number.isFinite(liveCount)) {
    return (
      <Kpi
        label="Traces"
        value={fmtInt(snapshotCount)}
        sub={
          <span className="flex items-center gap-1">
            <Badge variant="outline">SNAPSHOT</Badge>
            <span>Live OTEL count unavailable</span>
          </span>
        }
      />
    );
  }

  const freshness = traceFreshness(liveCount, snapshotCount);
  const detail =
    freshness.state === 'pending'
      ? `${fmtInt(freshness.pending)} awaiting L0 refresh`
      : freshness.state === 'current'
        ? 'L0 metrics current'
        : 'OTEL export is behind the L0 snapshot';

  return (
    <Kpi
      label="Traces"
      value={fmtInt(liveCount)}
      sub={
        <span className="flex items-center gap-1">
          <Badge variant="outline">LIVE</Badge>
          <span>{detail}</span>
        </span>
      }
    />
  );
}

export function CorpusKpis({ experimentId, annotationsTable }: { experimentId: string; annotationsTable?: string }) {
  // Memoize so a re-render doesn't retrigger the query (AppKit parameter guidance).
  const params = useMemo(() => ({ experiment_id: sql.string(experimentId) }), [experimentId]);
  const { data, loading, error } = useAnalyticsQuery('corpus_summary', params);

  if (loading) {
    return (
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {Array.from({ length: 8 }, (_, i) => (
          <Skeleton key={`kpi-skeleton-${i}`} className="h-24 w-full" />
        ))}
      </div>
    );
  }
  if (error) {
    return <div className="text-destructive bg-destructive/10 p-3 rounded-md">Error: {error}</div>;
  }
  const row = data?.[0];
  if (!row) return <div className="text-muted-foreground">No corpus data.</div>;
  const otelSpansTable = spansTableFromAnnotations(annotationsTable);

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      {otelSpansTable ? (
        <TraceKpi otelSpansTable={otelSpansTable} snapshotCount={Number(row.trace_count)} />
      ) : (
        <Kpi label="Traces" value={fmtInt(row.trace_count)} sub="Snapshot · live table not registered" />
      )}
      <Kpi label="Total Tokens" value={fmtInt(row.total_tokens)} sub="across all sessions" />
      <Kpi label="Median Tokens" value={fmtInt(row.median_tokens)} sub="bimodal: low median, heavy tail" />
      <Kpi label="p90 Tokens" value={fmtInt(row.p90_tokens)} />
      <Kpi label="Max Tokens" value={fmtInt(row.max_tokens)} sub="single largest session" />
      <Kpi label="Tool Calls" value={fmtInt(row.total_tool_calls)} />
      <Kpi label="Redundancy Rate" value={fmtPct(row.redundancy_rate)} sub="strict, byte-identical repeats" />
      <Kpi
        label="Est. Cost"
        value={fmtUsd(row.total_cost_usd)}
        sub={
          <span className="flex items-center gap-1">
            <Badge variant="outline">ESTIMATE</Badge>
            <span>
              {fmtInt(row.priced_traces)} priced · {fmtInt(row.unpriced_traces)} unpriced
            </span>
          </span>
        }
      />
    </div>
  );
}
