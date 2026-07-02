import { describe, it, expect } from 'vitest';
import {
  MISSING,
  buildLabelRequest,
  httpErrorMessage,
  labelMessage,
  missingDimensions,
  progressLabel,
  progressRatio,
  type DimensionProgress,
  type TraceTarget,
} from './labeling';

describe('progressLabel — renders the engine floor VERBATIM (two-tier, no hardcoded number)', () => {
  it('shows the engine-sent floor exactly — a sentinel unequal to any real default', () => {
    // The real floor is ail.readiness.ReadinessThresholds.quality_min_labels. We use a
    // distinctive sentinel here on purpose: any hardcoded number in the renderer would
    // make this fail. We intentionally do not restate the real default anywhere.
    const dim: Pick<DimensionProgress, 'labels_so_far' | 'label_floor'> = {
      labels_so_far: 9,
      label_floor: 33,
    };
    expect(progressLabel(dim)).toBe('9 / 33');
  });

  it('falls back to a neutral placeholder when a number is missing — never a fabricated digit', () => {
    const missing = progressLabel({ labels_so_far: NaN, label_floor: NaN });
    expect(missing).toBe(`${MISSING} / ${MISSING}`);
    expect(missing).not.toMatch(/\d/);
  });
});

describe('progressRatio — a clamped fill, never scaling a magnitude the engine did not send', () => {
  it('is have/floor, clamped to [0,1]', () => {
    expect(progressRatio({ labels_so_far: 5, label_floor: 20 })).toBe(0.25);
    expect(progressRatio({ labels_so_far: 40, label_floor: 20 })).toBe(1);
    expect(progressRatio({ labels_so_far: 0, label_floor: 20 })).toBe(0);
  });

  it('is null (empty bar) when the numbers are missing or the floor is 0', () => {
    expect(progressRatio({ labels_so_far: 5, label_floor: 0 })).toBeNull();
    expect(progressRatio({ labels_so_far: NaN, label_floor: 20 })).toBeNull();
  });
});

describe('buildLabelRequest — omits the labeler (server resolves it), fail-closed on empty', () => {
  const target = { experiment_id: 'exp-1', trace_id: 't1' };

  it('builds a name-matched request without a labeler field', () => {
    const req = buildLabelRequest(target, 'correctness', 'pass', '  clear evidence  ');
    expect(req).toEqual({
      experiment_id: 'exp-1',
      trace_id: 't1',
      name: 'correctness',
      value: 'pass',
      rationale: 'clear evidence',
    });
    expect('labeler' in req).toBe(false);
    expect('approver' in req).toBe(false);
  });

  it('passes a numeric value through (including 0) rather than dropping it', () => {
    expect(buildLabelRequest(target, 'modularity', 5).value).toBe(5);
    expect(buildLabelRequest(target, 'modularity', 0).value).toBe(0);
  });

  it('throws when the dimension name is empty', () => {
    expect(() => buildLabelRequest(target, '  ', 'pass')).toThrow(/dimension/);
  });

  it('throws when the value is missing or blank (cannot submit an empty label)', () => {
    expect(() => buildLabelRequest(target, 'correctness', null)).toThrow(/value/);
    expect(() => buildLabelRequest(target, 'correctness', '   ')).toThrow(/value/);
    expect(() => buildLabelRequest(target, 'correctness', undefined)).toThrow(/value/);
  });
});

describe('labelMessage — honest outcome mapping', () => {
  it('labeled is a success with the verbatim engine progress', () => {
    const msg = labelMessage({ outcome: 'labeled', name: 'correctness', labels_so_far: 4, label_floor: 20 });
    expect(msg.tone).toBe('success');
    expect(msg.text).toContain('4 / 20');
  });

  it('refused surfaces WHY and states nothing was written', () => {
    const msg = labelMessage({ outcome: 'refused', refused_reason: "'x' is not a registered judge" });
    expect(msg.tone).toBe('error');
    expect(msg.text).toContain('not a registered judge');
    expect(msg.text).toContain('Nothing was written');
  });

  it('error surfaces the failure and is never dressed up as labeled', () => {
    const msg = labelMessage({ outcome: 'error', error: 'PermissionError: not authorized' });
    expect(msg.tone).toBe('error');
    expect(msg.text).toContain('not authorized');
    expect(msg.text).not.toContain('Labeled');
  });
});

describe('httpErrorMessage', () => {
  it('maps 401 to a sign-in message and other codes to a generic failure', () => {
    expect(httpErrorMessage(401)).toMatch(/sign in/i);
    expect(httpErrorMessage(502)).toContain('502');
  });
});

describe('missingDimensions — reads the labeled map, never recomputes coverage', () => {
  it('returns the registered dimensions this trace still lacks, in order', () => {
    const target: TraceTarget = {
      trace_id: 't1',
      labeled: { correctness: true, modularity: false },
    };
    expect(missingDimensions(target, ['correctness', 'modularity'])).toEqual(['modularity']);
  });

  it('is empty when the trace is fully labeled', () => {
    const target: TraceTarget = { trace_id: 't1', labeled: { correctness: true, modularity: true } };
    expect(missingDimensions(target, ['correctness', 'modularity'])).toEqual([]);
  });
});
