import { describe, it, expect } from 'vitest';
import { presentStatus, toneBadgeVariant, deltaDirectionClass } from './versionStatus';
import { fmtSignedPct, fmtMetricValue } from './formatters';

describe('presentStatus', () => {
  it('only proven_improvement is the positive (green) tone', () => {
    expect(presentStatus('proven_improvement').tone).toBe('positive');
  });

  it('controlled_proof_collecting is caution, NOT positive (no false green)', () => {
    const p = presentStatus('controlled_proof_collecting');
    expect(p.tone).toBe('caution');
    expect(p.tone).not.toBe('positive');
    expect(p.label).toMatch(/collecting/i);
  });

  it('regressed is negative; collecting is neutral', () => {
    expect(presentStatus('regressed').tone).toBe('negative');
    expect(presentStatus('collecting').tone).toBe('neutral');
  });

  it('unknown status falls back to neutral with the raw label', () => {
    const p = presentStatus('something_new');
    expect(p.tone).toBe('neutral');
    expect(p.label).toBe('something_new');
  });
});

describe('toneBadgeVariant', () => {
  it('maps tones to AppKit Badge variants', () => {
    expect(toneBadgeVariant('positive')).toBe('default');
    expect(toneBadgeVariant('caution')).toBe('secondary');
    expect(toneBadgeVariant('negative')).toBe('destructive');
    expect(toneBadgeVariant('neutral')).toBe('outline');
  });
});

describe('deltaDirectionClass', () => {
  it('an unchanged metric is muted regardless of improved flag', () => {
    expect(deltaDirectionClass(true, false)).toMatch(/muted/);
  });
  it('an improved change is emerald, a worse change is destructive', () => {
    expect(deltaDirectionClass(true, true)).toMatch(/emerald/);
    expect(deltaDirectionClass(false, true)).toMatch(/destructive/);
  });
});

describe('formatters', () => {
  it('fmtSignedPct keeps sign and renders null as an em dash (never fabricated 0%)', () => {
    expect(fmtSignedPct(-35.4086)).toBe('-35.4%');
    expect(fmtSignedPct(12)).toBe('+12.0%');
    expect(fmtSignedPct(null)).toBe('—');
  });

  it('fmtMetricValue formats by metric kind', () => {
    expect(fmtMetricValue('total_tokens', 2570)).toBe('2,570');
    expect(fmtMetricValue('redundancy_rate', 0.046875)).toBe('4.7%');
    expect(fmtMetricValue('total_usd', 0)).toBe('$0.00');
  });
});
