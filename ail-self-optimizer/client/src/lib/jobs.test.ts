import { describe, it, expect } from 'vitest';
import { fmtDurationMs, fmtEpochMs, outcomeTone, runStateText, runTone, UNTRACKED_OPTIMIZERS } from './jobs';

describe('runStateText — verbatim, never invented', () => {
  it('joins a terminal life_cycle_state and result_state exactly as returned', () => {
    expect(runStateText({ life_cycle_state: 'TERMINATED', result_state: 'SUCCESS' })).toBe('TERMINATED · SUCCESS');
    expect(runStateText({ life_cycle_state: 'TERMINATED', result_state: 'FAILED' })).toBe('TERMINATED · FAILED');
  });

  it('shows just the lifecycle for a non-terminal run (no fabricated result)', () => {
    expect(runStateText({ life_cycle_state: 'RUNNING' })).toBe('RUNNING');
  });

  it('says UNKNOWN when the SDK reported no state — never guesses', () => {
    expect(runStateText({})).toBe('UNKNOWN');
  });
});

describe('runTone — faithful to the verbatim state, neutral by default', () => {
  it('colors only the two unambiguous terminal results', () => {
    expect(runTone({ life_cycle_state: 'TERMINATED', result_state: 'SUCCESS' })).toBe('success');
    expect(runTone({ life_cycle_state: 'TERMINATED', result_state: 'FAILED' })).toBe('error');
  });

  it('a running lifecycle is active', () => {
    expect(runTone({ life_cycle_state: 'RUNNING' })).toBe('active');
    expect(runTone({ life_cycle_state: 'PENDING' })).toBe('active');
  });

  it('an unrecognized / ambiguous terminal result is NEVER colored as success or error', () => {
    // CANCELED / TIMEDOUT / SKIPPED and any future state fall to neutral — we never
    // dress up a state we cannot be certain about.
    expect(runTone({ life_cycle_state: 'TERMINATED', result_state: 'CANCELED' })).toBe('neutral');
    expect(runTone({ life_cycle_state: 'INTERNAL_ERROR' })).toBe('neutral');
    expect(runTone({ life_cycle_state: 'TERMINATED', result_state: 'SOME_NEW_STATE' })).toBe('neutral');
    expect(runTone({})).toBe('neutral');
  });
});

describe('fmtEpochMs — honest timestamps', () => {
  it('formats epoch ms as a UTC timestamp', () => {
    expect(fmtEpochMs(1_000)).toBe('1970-01-01 00:00:01Z');
  });

  it('returns an em dash when the time is absent (a run that has not ended)', () => {
    expect(fmtEpochMs(undefined)).toBe('—');
    expect(fmtEpochMs(null)).toBe('—');
    expect(fmtEpochMs(0)).toBe('—');
    expect(fmtEpochMs(Number.NaN)).toBe('—');
  });
});

describe('fmtDurationMs — the SDK duration only, never derived', () => {
  it('formats ms / s / m+s', () => {
    expect(fmtDurationMs(400)).toBe('400 ms');
    expect(fmtDurationMs(1_500)).toBe('1.5 s');
    expect(fmtDurationMs(90_000)).toBe('1m 30s');
  });

  it('returns an em dash when the SDK reported no duration — never computes one', () => {
    expect(fmtDurationMs(undefined)).toBe('—');
    expect(fmtDurationMs(0)).toBe('—');
    expect(fmtDurationMs(null)).toBe('—');
  });
});

describe('outcomeTone — applied is a win, rejected is neutral (never an error)', () => {
  it('maps known proposal statuses faithfully', () => {
    expect(outcomeTone('applied')).toBe('success');
    expect(outcomeTone('pending')).toBe('active');
    expect(outcomeTone('approved')).toBe('active');
    expect(outcomeTone('rejected')).toBe('neutral');
    expect(outcomeTone('superseded')).toBe('neutral');
  });

  it('an unrecognized status is neutral, never success', () => {
    expect(outcomeTone('mystery')).toBe('neutral');
    expect(outcomeTone(undefined)).toBe('neutral');
  });
});

describe('UNTRACKED_OPTIMIZERS — the explicit not-tracked set', () => {
  it('only names optimizer work that still lacks its own tracked job', () => {
    const blob = UNTRACKED_OPTIMIZERS.map((o) => `${o.key} ${o.name}`.toLowerCase()).join(' ');
    expect(blob).toMatch(/gepa/);
    expect(blob).toMatch(/asset/);
    expect(blob).not.toMatch(/rlm/);
    expect(blob).not.toMatch(/memalign/);
  });
});
