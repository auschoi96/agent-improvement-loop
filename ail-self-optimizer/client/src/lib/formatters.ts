// SQL numerics can arrive as strings at runtime (large BIGINT, DECIMAL, ROUND/
// SUM results), so always coerce before formatting.
export const toNum = (v: number | string | null | undefined): number => Number(v ?? 0);

export const fmtInt = (v: number | string | null | undefined): string =>
  toNum(v).toLocaleString('en-US', { maximumFractionDigits: 0 });

export const fmtUsd = (v: number | string | null | undefined): string =>
  `$${toNum(v).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

export const fmtPct = (rate: number | string | null | undefined): string => `${(toNum(rate) * 100).toFixed(1)}%`;
