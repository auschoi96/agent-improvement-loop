// The product brand accent. Mirrors the `--brand` CSS token defined in index.css.
// Charts (AppKit BarChart → Recharts) need a concrete color string rather than a
// CSS variable, so the value lives here in TS and is re-declared as var(--brand)
// for CSS-land usage. Keep the two in sync.
export const BRAND_ACCENT = '#40d1f5';
