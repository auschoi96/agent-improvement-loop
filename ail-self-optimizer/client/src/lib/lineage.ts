// Presentation + audit-honesty rules for the prompt-lineage timeline
// (LineageTimeline.tsx). The provenance is decided in Tier A Python and published to
// agent_prompt_lineage; this module only maps it to labels/tones.
//
// CRITICAL audit honesty: a version that was force-registered despite NOT beating its
// seed on the held-out split (`is_forced_non_improving`) must NEVER be styled as a
// genuine improvement. `deltaTone` returns 'warning' for any forced version
// regardless of the recorded delta, and the green 'positive' tone is reserved for a
// real gepa-evolved version whose held-out delta actually beat its seed. This is the
// whole point of the audit trail — catching a change that only looked like a win.

export type LineageDeltaTone = 'positive' | 'warning' | 'muted';

export const SOURCE_GEPA_EVOLVED = 'gepa-evolved';
export const SOURCE_SEED = 'seed';

// The label shown on the forced / non-improving warning badge.
export const FORCED_BADGE_LABEL = 'forced / not a proven improvement';

// SQL BOOLEAN can arrive as a real boolean or as a string ('true'/'false') at
// runtime; the audit-honesty flags must not be fooled by a truthy "false" string
// (a non-empty string is truthy in JS, so a naive cast would flag every version).
export function toBool(v: boolean | string | number | null | undefined): boolean {
  if (typeof v === 'boolean') return v;
  if (typeof v === 'number') return v !== 0;
  if (typeof v === 'string') return v.toLowerCase() === 'true' || v === '1';
  return false;
}

export function sourceLabel(source: string): string {
  switch (source) {
    case SOURCE_SEED:
      return 'Seed (baseline)';
    case SOURCE_GEPA_EVOLVED:
      return 'GEPA-evolved';
    default:
      return source;
  }
}

export interface LineageRowLike {
  source: string;
  is_forced_non_improving: boolean | string | number | null | undefined;
  holdout_savings_delta_pct: number | string | null | undefined;
}

// The tone for a version's held-out delta. Forced => 'warning' (never green), no
// matter what number was recorded; a genuine gepa-evolved version that beat its seed
// (delta > 0) => 'positive'; anything else (seed baseline, a no-op, an unproven
// candidate) => 'muted'.
export function deltaTone(row: LineageRowLike): LineageDeltaTone {
  if (toBool(row.is_forced_non_improving)) return 'warning';
  const delta = row.holdout_savings_delta_pct;
  const beatSeed = delta !== null && delta !== undefined && Number(delta) > 0;
  if (row.source === SOURCE_GEPA_EVOLVED && beatSeed) return 'positive';
  return 'muted';
}

export function deltaToneTextClass(tone: LineageDeltaTone): string {
  switch (tone) {
    case 'positive':
      return 'text-emerald-600 dark:text-emerald-400';
    case 'warning':
      return 'text-destructive';
    default:
      return 'text-muted-foreground';
  }
}

// Card classes for a forced (non-improving) version — amber caution fill + border,
// deliberately NOT the emerald/green a real improvement gets.
export function forcedCardClasses(): string {
  return 'border-amber-500/50 bg-amber-500/10';
}

// A magnitude percentage already in percent units (e.g. 45.0 -> "45.0%"); null/
// undefined renders as an em dash, never a fabricated 0%.
export function fmtPctMagnitude(v: number | string | null | undefined): string {
  if (v === null || v === undefined) return '—';
  return `${Number(v).toFixed(1)}%`;
}
