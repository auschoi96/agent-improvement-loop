import { useMemo } from 'react';
import {
  useAnalyticsQuery,
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
  Badge,
  Skeleton,
} from '@databricks/appkit-ui/react';
import { sql } from '@databricks/appkit-ui/js';
import { fmtInt, fmtMetricValue, fmtSignedPct, toNum } from '../lib/formatters';
import { presentStatus, toneBadgeVariant, toneBannerClasses, deltaDirectionClass } from '../lib/versionStatus';

const METRIC_LABELS: Record<string, string> = {
  total_tokens: 'Total tokens',
  tokens_per_trace: 'Tokens / trace',
  total_tool_calls: 'Tool calls',
  redundancy_rate: 'Redundancy rate',
  total_usd: 'Est. cost',
};

const metricLabel = (metric: string): string => METRIC_LABELS[metric] ?? metric;

const parseConfig = (raw: string | null | undefined): Record<string, unknown> => {
  if (!raw) return {};
  try {
    const parsed: unknown = JSON.parse(raw);
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? (parsed as Record<string, unknown>) : {};
  } catch {
    return {};
  }
};

const configValue = (value: unknown): string => {
  if (value === null || value === undefined || value === '') return 'not captured';
  if (typeof value === 'string') return value;
  return JSON.stringify(value);
};

function ConfigurationDiff({
  agentName,
  experimentId,
  baselineVersion,
  candidateVersion,
}: {
  agentName: string;
  experimentId: string;
  baselineVersion: string;
  candidateVersion: string;
}) {
  const params = useMemo(
    () => ({ agent_name: sql.string(agentName), experiment_id: sql.string(experimentId) }),
    [agentName, experimentId]
  );
  const { data, loading, error } = useAnalyticsQuery('agent_versions', params);
  if (loading) return <Skeleton className="h-44 w-full" />;
  if (error) return <div className="text-destructive bg-destructive/10 p-3 rounded-md">Error: {error}</div>;

  const baseline = data?.find((version) => version.agent_version === baselineVersion);
  const candidate = data?.find((version) => version.agent_version === candidateVersion);
  if (!baseline || !candidate) return null;

  const baselineConfig = parseConfig(baseline.config_json);
  const candidateConfig = parseConfig(candidate.config_json);
  const keys = Array.from(new Set([...Object.keys(baselineConfig), ...Object.keys(candidateConfig)])).sort();

  return (
    <Card className="shadow-sm">
      <CardHeader>
        <CardTitle>Coding-agent configuration</CardTitle>
        <CardDescription>
          External MLflow agent versions capture launcher configuration and provenance; an active model ID links future
          traces to the exact version that produced them.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-[minmax(8rem,1fr)_minmax(0,2fr)_minmax(0,2fr)] gap-x-3 gap-y-2 text-xs">
          <div className="font-medium text-muted-foreground">Field</div>
          <div className="font-medium">{baselineVersion}</div>
          <div className="font-medium">{candidateVersion}</div>
          {keys.map((key) => {
            const before = configValue(baselineConfig[key]);
            const after = configValue(candidateConfig[key]);
            const changed = before !== after;
            return (
              <div key={key} className="contents">
                <div className="text-muted-foreground break-words">{key.replaceAll('_', ' ')}</div>
                <div className="font-mono break-all">{before}</div>
                <div className={`font-mono break-all ${changed ? 'text-blue-600 dark:text-blue-400' : ''}`}>
                  {after}
                </div>
              </div>
            );
          })}
        </div>
        <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
          <Badge variant="outline">baseline MLflow ID: {baseline.logged_model_id || 'not registered'}</Badge>
          <Badge variant="outline">candidate MLflow ID: {candidate.logged_model_id || 'not registered'}</Badge>
          <Badge variant="outline">
            fingerprints: {baseline.config_fingerprint} → {candidate.config_fingerprint}
          </Badge>
        </div>
      </CardContent>
    </Card>
  );
}

// The per-metric delta cards. A separate component so its query parameters (which
// depend on the chosen baseline/candidate versions) can be memoized in isolation.
function MetricDeltas({
  agentName,
  experimentId,
  baselineVersion,
  candidateVersion,
}: {
  agentName: string;
  experimentId: string;
  baselineVersion: string;
  candidateVersion: string;
}) {
  const params = useMemo(
    () => ({
      agent_name: sql.string(agentName),
      experiment_id: sql.string(experimentId),
      baseline_version: sql.string(baselineVersion),
      candidate_version: sql.string(candidateVersion),
    }),
    [agentName, experimentId, baselineVersion, candidateVersion]
  );
  const { data, loading, error } = useAnalyticsQuery('version_metric_deltas', params);

  if (loading) {
    return (
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {Array.from({ length: 5 }, (_, i) => (
          <Skeleton key={`delta-skeleton-${i}`} className="h-28 w-full" />
        ))}
      </div>
    );
  }
  if (error) {
    return <div className="text-destructive bg-destructive/10 p-3 rounded-md">Error: {error}</div>;
  }
  if (!data?.length) {
    return <div className="text-muted-foreground">No per-metric deltas published.</div>;
  }

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
      {data.map((d) => {
        // SQL DOUBLEs can arrive as strings at runtime, so coerce before any
        // numeric comparison (a lexical '830' < '1285' would flip the arrow).
        const deltaAbs = toNum(d.delta_absolute);
        const changed = deltaAbs !== 0;
        const dirClass = deltaDirectionClass(Boolean(d.improved), changed);
        const arrow = !changed ? '' : deltaAbs < 0 ? '↓' : '↑';
        return (
          <Card key={d.metric} className="shadow-sm">
            <CardContent className="p-4">
              <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                {metricLabel(d.metric)}
              </div>
              <div className={`mt-1 text-2xl font-bold tabular-nums ${dirClass}`}>
                {arrow} {fmtSignedPct(d.delta_pct)}
              </div>
              <div className="mt-1 text-xs text-muted-foreground tabular-nums">
                {fmtMetricValue(d.metric, d.baseline_value)} → {fmtMetricValue(d.metric, d.candidate_value)}
                {d.metric === 'total_usd' && ' (unpriced)'}
              </div>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}

export function VersionComparison({ agentName, experimentId }: { agentName: string; experimentId: string }) {
  const params = useMemo(
    () => ({ agent_name: sql.string(agentName), experiment_id: sql.string(experimentId) }),
    [agentName, experimentId]
  );
  const { data, loading, error } = useAnalyticsQuery('version_comparisons', params);

  if (loading) return <Skeleton className="h-48 w-full" />;
  if (error) {
    return <div className="text-destructive bg-destructive/10 p-3 rounded-md">Error: {error}</div>;
  }
  const cmp = data?.[0];
  if (!cmp) {
    return (
      <div className="text-muted-foreground border rounded-md p-4">
        No version comparison published for this agent yet — register a baseline and a candidate version and publish a
        controlled comparison to populate this view.
      </div>
    );
  }

  const status = presentStatus(cmp.status);
  const reasons = (cmp.reasons || '').split(' | ').filter(Boolean);
  const headlineChanged = Number(cmp.headline_baseline) !== Number(cmp.headline_candidate);
  const headlineClass = deltaDirectionClass(Boolean(cmp.headline_improved), headlineChanged);

  return (
    <div className="space-y-4">
      {/* Trust verdict banner — tone is gated by the Python-decided status. The
          headline delta is real and measured; this banner is the only place the
          "improvement cleared" verdict lives, and it is amber (not green) until
          the readiness wall clears. */}
      <div className={`rounded-lg border p-4 ${toneBannerClasses(status.tone)}`}>
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant={toneBadgeVariant(status.tone)}>{status.label}</Badge>
          <span className="text-sm font-medium text-foreground">
            {cmp.baseline_version} → {cmp.candidate_version}
          </span>
          <Badge variant="outline">readiness: {cmp.readiness_tier}</Badge>
        </div>
        <p className="mt-2 text-sm">{status.description}</p>
        <p className="mt-2 text-xs text-muted-foreground">
          Controlled comparison · {cmp.proof_source.replaceAll('_', ' ')} ·{' '}
          <strong>{fmtInt(cmp.n_promote)} PROMOTE</strong> / {fmtInt(cmp.n_block)} BLOCK
          {cmp.n_errored > 0 ? ` / ${fmtInt(cmp.n_errored)} errored` : ''} ·{' '}
          {cmp.correctness_held ? 'correctness held' : 'correctness NOT held'} · frozen suite{' '}
          {cmp.frozen_suite_present ? 'present' : 'absent'} · {fmtInt(cmp.trace_count)} organic trace(s)
        </p>
        {reasons.length > 0 && (
          <ul className="mt-2 text-xs text-muted-foreground list-disc list-inside space-y-0.5">
            {reasons.map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ul>
        )}
      </div>

      <ConfigurationDiff
        agentName={agentName}
        experimentId={experimentId}
        baselineVersion={cmp.baseline_version}
        candidateVersion={cmp.candidate_version}
      />

      {/* Headline objective card. */}
      <Card className="shadow-sm">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            Headline · {metricLabel(cmp.headline_metric)}
            <Badge variant="outline">objective</Badge>
          </CardTitle>
          <CardDescription>The proven, correctness-held objective delta over the PROMOTE task set.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className={`text-4xl font-bold tabular-nums ${headlineClass}`}>
            {fmtSignedPct(cmp.headline_delta_pct)}
          </div>
          <div className="mt-1 text-sm text-muted-foreground tabular-nums">
            {fmtMetricValue(cmp.headline_metric, cmp.headline_baseline)} →{' '}
            {fmtMetricValue(cmp.headline_metric, cmp.headline_candidate)} tokens
          </div>
        </CardContent>
      </Card>

      <MetricDeltas
        agentName={agentName}
        experimentId={experimentId}
        baselineVersion={cmp.baseline_version}
        candidateVersion={cmp.candidate_version}
      />
    </div>
  );
}
