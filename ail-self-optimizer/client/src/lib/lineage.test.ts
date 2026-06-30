import { describe, it, expect } from 'vitest';
import {
  toBool,
  sourceLabel,
  deltaTone,
  deltaToneTextClass,
  fmtPctMagnitude,
  SOURCE_GEPA_EVOLVED,
  SOURCE_SEED,
} from './lineage';

describe('toBool', () => {
  it('treats the string "false" as false (not truthy)', () => {
    // A non-empty string is truthy in JS — a naive cast would flag every version as
    // forced/champion. The honesty flags must survive a SQL boolean arriving as text.
    expect(toBool('false')).toBe(false);
    expect(toBool('true')).toBe(true);
    expect(toBool(false)).toBe(false);
    expect(toBool(true)).toBe(true);
    expect(toBool(0)).toBe(false);
    expect(toBool(1)).toBe(true);
    expect(toBool(null)).toBe(false);
    expect(toBool(undefined)).toBe(false);
  });
});

describe('sourceLabel', () => {
  it('labels seed and gepa-evolved, passing through anything else', () => {
    expect(sourceLabel(SOURCE_SEED)).toMatch(/seed/i);
    expect(sourceLabel(SOURCE_GEPA_EVOLVED)).toMatch(/gepa/i);
    expect(sourceLabel('mystery')).toBe('mystery');
  });
});

describe('deltaTone — audit honesty', () => {
  it('a forced version is WARNING even when its recorded delta is positive (never green)', () => {
    const forced = { source: SOURCE_GEPA_EVOLVED, is_forced_non_improving: true, holdout_savings_delta_pct: 15 };
    expect(deltaTone(forced)).toBe('warning');
    expect(deltaTone(forced)).not.toBe('positive');
    // ...and even when the flag arrives as the string "true".
    expect(deltaTone({ ...forced, is_forced_non_improving: 'true' })).toBe('warning');
  });

  it('a genuine gepa-evolved version that beat its seed is POSITIVE (green)', () => {
    expect(
      deltaTone({ source: SOURCE_GEPA_EVOLVED, is_forced_non_improving: false, holdout_savings_delta_pct: 15 })
    ).toBe('positive');
  });

  it('a gepa version that did not beat seed (delta <= 0) is muted, not positive', () => {
    expect(
      deltaTone({ source: SOURCE_GEPA_EVOLVED, is_forced_non_improving: false, holdout_savings_delta_pct: 0 })
    ).toBe('muted');
    expect(
      deltaTone({ source: SOURCE_GEPA_EVOLVED, is_forced_non_improving: false, holdout_savings_delta_pct: -5 })
    ).toBe('muted');
  });

  it('a seed baseline is muted (no held-out delta)', () => {
    expect(deltaTone({ source: SOURCE_SEED, is_forced_non_improving: false, holdout_savings_delta_pct: null })).toBe(
      'muted'
    );
  });
});

describe('deltaToneTextClass', () => {
  it('positive is emerald, warning is destructive (never green for forced), muted is muted', () => {
    expect(deltaToneTextClass('positive')).toMatch(/emerald/);
    expect(deltaToneTextClass('warning')).toMatch(/destructive/);
    expect(deltaToneTextClass('warning')).not.toMatch(/emerald/);
    expect(deltaToneTextClass('muted')).toMatch(/muted/);
  });
});

describe('fmtPctMagnitude', () => {
  it('renders a percent magnitude and an em dash for null (never fabricated 0%)', () => {
    expect(fmtPctMagnitude(45)).toBe('45.0%');
    expect(fmtPctMagnitude('30')).toBe('30.0%');
    expect(fmtPctMagnitude(null)).toBe('—');
    expect(fmtPctMagnitude(undefined)).toBe('—');
  });
});
