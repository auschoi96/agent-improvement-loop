import { useMemo } from 'react';
import { useAnalyticsQuery, Card, CardContent, Badge, Skeleton } from '@databricks/appkit-ui/react';
import { sql } from '@databricks/appkit-ui/js';
import { fmtInt, fmtSignedPct, toNum } from '../lib/formatters';
import {
  sourceLabel,
  deltaTone,
  deltaToneTextClass,
  forcedCardClasses,
  fmtPctMagnitude,
  toBool,
  FORCED_BADGE_LABEL,
  SOURCE_GEPA_EVOLVED,
  SOURCE_SEED,
} from '../lib/lineage';

// The per-agent prompt-registry lineage / audit timeline (Phase C). Each registered
// prompt version, newest first: its source (seed vs GEPA-evolved), the proven
// held-out delta (evolved-vs-seed savings), GEPA scores, the candidate artifact, and
// which version is the CHAMPION. A force-registered non-improving version is flagged
// with a warning badge + the recorded reason and is NEVER styled as an improvement —
// that honesty is the whole point of the audit trail.
export function LineageTimeline({ agentName, experimentId }: { agentName: string; experimentId: string }) {
  const params = useMemo(
    () => ({ agent_name: sql.string(agentName), experiment_id: sql.string(experimentId) }),
    [agentName, experimentId]
  );
  const { data, loading, error } = useAnalyticsQuery('prompt_lineage', params);

  if (loading) {
    return (
      <div className="space-y-3">
        {Array.from({ length: 3 }, (_, i) => (
          <Skeleton key={`lineage-skeleton-${i}`} className="h-28 w-full" />
        ))}
      </div>
    );
  }
  if (error) {
    return <div className="text-destructive bg-destructive/10 p-3 rounded-md">Error: {error}</div>;
  }
  if (!data?.length) {
    return (
      <div className="text-muted-foreground border rounded-md p-4">
        No registered prompt versions yet — nothing has been promoted for this agent.
      </div>
    );
  }

  return (
    <ol className="space-y-4">
      {data.map((row) => {
        const isChampion = toBool(row.is_champion);
        const isForced = toBool(row.is_forced_non_improving);
        const isGepa = row.source === SOURCE_GEPA_EVOLVED;
        const hasDelta = row.holdout_savings_delta_pct !== null && row.holdout_savings_delta_pct !== undefined;
        const tone = deltaTone(row);
        const toneClass = deltaToneTextClass(tone);
        // Forced = amber caution; champion (and genuine) = emerald accent; never let a
        // forced version borrow the champion/improvement styling.
        const cardClass = isForced ? forcedCardClasses() : isChampion ? 'border-emerald-500/40' : '';

        return (
          <li key={row.version}>
            <Card className={`shadow-sm ${cardClass}`}>
              <CardContent className="p-4 space-y-2">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-base font-semibold tabular-nums">v{row.version}</span>
                  <Badge variant="outline">{sourceLabel(row.source)}</Badge>
                  {isChampion && <Badge variant="default">CHAMPION</Badge>}
                  {isForced && <Badge variant="destructive">⚠ {FORCED_BADGE_LABEL}</Badge>}
                </div>

                {isGepa && hasDelta ? (
                  <div>
                    <div className={`text-2xl font-bold tabular-nums ${toneClass}`}>
                      {fmtSignedPct(row.holdout_savings_delta_pct)} held-out savings vs seed
                    </div>
                    <div className="mt-0.5 text-xs text-muted-foreground tabular-nums">
                      evolved {fmtPctMagnitude(row.holdout_evolved_savings_pct)} vs seed{' '}
                      {fmtPctMagnitude(row.holdout_seed_savings_pct)} (held-out PROMOTE-task token savings)
                    </div>
                  </div>
                ) : (
                  <div className="text-sm text-muted-foreground">
                    {row.source === SOURCE_SEED
                      ? 'Baseline seed — the version every candidate is measured against; no held-out delta.'
                      : 'No held-out comparison recorded for this version.'}
                  </div>
                )}

                {/* Audit honesty: a forced version surfaces WHY it is not a proven win. */}
                {isForced && (
                  <p className="text-xs text-amber-700 dark:text-amber-300">
                    <strong>Forced registration — not a proven improvement.</strong>{' '}
                    {row.registration_reason || 'Registered despite not beating its seed on the held-out split.'}
                  </p>
                )}

                <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
                  {isGepa && row.gepa_best_val_score !== null && row.gepa_best_val_score !== undefined && (
                    <span>
                      GEPA best val {toNum(row.gepa_best_val_score).toFixed(2)}
                      {row.gepa_num_candidates ? ` · ${fmtInt(row.gepa_num_candidates)} candidates` : ''}
                    </span>
                  )}
                  {row.suite_version && <span>suite {row.suite_version}</span>}
                  {row.candidate_artifact && (
                    <span>
                      artifact: <code className="text-foreground">{row.candidate_artifact}</code>
                    </span>
                  )}
                  {row.registered_at && <span>registered {row.registered_at}</span>}
                </div>
              </CardContent>
            </Card>
          </li>
        );
      })}
    </ol>
  );
}
