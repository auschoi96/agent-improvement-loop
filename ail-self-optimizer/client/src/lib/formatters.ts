// SQL numerics can arrive as strings at runtime (large BIGINT, DECIMAL, ROUND/
// SUM results), so always coerce before formatting.
export const toNum = (v: number | string | null | undefined): number => Number(v ?? 0);

export const fmtInt = (v: number | string | null | undefined): string =>
  toNum(v).toLocaleString('en-US', { maximumFractionDigits: 0 });

export const fmtUsd = (v: number | string | null | undefined): string =>
  `$${toNum(v).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

export const fmtPct = (rate: number | string | null | undefined): string => `${(toNum(rate) * 100).toFixed(1)}%`;

// A signed percentage that is ALREADY in percent units (e.g. delta_pct = -35.41).
// null/undefined (an undefined %, e.g. a zero baseline) renders as an em dash —
// never a fabricated 0%.
export const fmtSignedPct = (pct: number | string | null | undefined): string => {
  if (pct === null || pct === undefined) return '—';
  const n = toNum(pct);
  return `${n > 0 ? '+' : ''}${n.toFixed(1)}%`;
};

// A metric value formatted by its kind. Rates are fractions (0..1) shown as %;
// dollars as USD; everything else as a plain integer count/token total.
export const fmtMetricValue = (metric: string, value: number | string | null | undefined): string => {
  if (metric === 'total_usd') return fmtUsd(value);
  if (metric === 'redundancy_rate') return fmtPct(value);
  return fmtInt(value);
};
