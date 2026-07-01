import { describe, it, expect } from 'vitest';
import {
  TUTORIAL_STEPS,
  LOOP_STAGES,
  MEASUREMENT_LAYERS,
  READINESS_GATES,
  THRESHOLD_PLACEHOLDER,
  clampTutorialStep,
  isFirstTutorialStep,
  isLastTutorialStep,
  tutorialProgressPct,
  readinessGateLines,
  thresholdsFromRequirements,
} from './tutorial';
import type { RequirementsResponse, Thresholds } from './onboarding';

// DISTINCTIVE SENTINEL thresholds — deliberately NONE of the real code-enforced
// defaults (baseline 10 / prove 50 / labels 20 / coverage 0.5). If any gate number
// were hardcoded in TypeScript instead of read from this input, these sentinels
// could not appear in the output (and the "no default leaked" assertion would fail).
const SENTINEL: Thresholds = {
  baseline_min_traces: 7,
  prove_min_traces: 33,
  quality_min_labels: 88,
  scored_coverage_floor: 0.42,
};

describe('TUTORIAL_STEPS — the content spine (mirrors docs/GETTING_STARTED.md)', () => {
  it('covers the four content beats with unique keys and real body prose', () => {
    const keys = TUTORIAL_STEPS.map((s) => s.key);
    expect(keys).toEqual(['loop', 'connect', 'readiness', 'approval']);
    expect(new Set(keys).size).toBe(keys.length);
    for (const step of TUTORIAL_STEPS) {
      expect(step.title.trim().length).toBeGreaterThan(0);
      expect(step.body.length).toBeGreaterThan(0);
      expect(step.body.every((p) => p.trim().length > 0)).toBe(true);
    }
  });

  it('states the honest "connect, don’t upload" and "0 traces → 0 optimization" foundation', () => {
    const connect = TUTORIAL_STEPS.find((s) => s.key === 'connect');
    const text = [connect?.body ?? [], connect?.points ?? []].flat().join(' ');
    expect(text).toMatch(/upload/i);
    expect(text).toMatch(/0 traces/);
  });

  it('models the loop with a human-approval stage and the three measurement layers', () => {
    expect(LOOP_STAGES.some((s) => s.role === 'human')).toBe(true);
    expect(MEASUREMENT_LAYERS.map((l) => l.key)).toEqual(['l0', 'l2', 'l3']);
  });
});

describe('tutorial navigation helpers — pure and bounded', () => {
  it('clampTutorialStep keeps the index in range', () => {
    expect(clampTutorialStep(-5)).toBe(0);
    expect(clampTutorialStep(999)).toBe(TUTORIAL_STEPS.length - 1);
    expect(clampTutorialStep(1)).toBe(1);
  });

  it('first/last predicates only fire at the ends', () => {
    expect(isFirstTutorialStep(0)).toBe(true);
    expect(isFirstTutorialStep(1)).toBe(false);
    expect(isLastTutorialStep(TUTORIAL_STEPS.length - 1)).toBe(true);
    expect(isLastTutorialStep(0)).toBe(false);
  });

  it('progress is monotonic and ends at 100%', () => {
    expect(tutorialProgressPct(0)).toBeCloseTo(100 / TUTORIAL_STEPS.length);
    expect(tutorialProgressPct(TUTORIAL_STEPS.length - 1)).toBe(100);
    expect(tutorialProgressPct(999)).toBe(100);
    expect(tutorialProgressPct(1)).toBeGreaterThan(tutorialProgressPct(0));
  });
});

describe('readinessGateLines — TWO-TIER: every threshold is the Python value, verbatim', () => {
  it('renders each gate against its OWN Python threshold field (sentinels pass through)', () => {
    const lines = readinessGateLines(SENTINEL);
    const by = Object.fromEntries(lines.map((l) => [l.key, l]));

    expect(by.baseline.requirement).toBe('≥ 7 traces');
    expect(by.prove.requirement).toBe('≥ 33 traces');
    expect(by.labels.requirement).toBe('≥ 88 human labels');
    // The coverage floor is a fraction the Python engine presents as a percentage —
    // a display-only transform of the SAME sentinel value.
    expect(by.coverage.requirement).toBe('≥ 42%');
    expect(lines.every((l) => l.loaded)).toBe(true);
  });

  it('never leaks a hardcoded default — only the supplied sentinels appear', () => {
    const rendered = readinessGateLines(SENTINEL)
      .map((l) => l.requirement)
      .join(' | ');
    // The real defaults (10 / 20 / 50) must NOT appear when sentinels are supplied;
    // their presence would prove a number was authored in TS rather than read from Python.
    expect(rendered).not.toMatch(/\b10\b/);
    expect(rendered).not.toMatch(/\b20\b/);
    expect(rendered).not.toMatch(/\b50\b/);
  });

  it('falls closed to a neutral placeholder when thresholds are not loaded — never a number', () => {
    const lines = readinessGateLines(null);
    expect(lines).toHaveLength(READINESS_GATES.length);
    for (const line of lines) {
      expect(line.requirement).toBe(THRESHOLD_PLACEHOLDER);
      expect(line.loaded).toBe(false);
      expect(line.requirement).not.toMatch(/\d/);
    }
  });

  it('carries the explanatory unlock prose for every gate regardless of load state', () => {
    for (const line of readinessGateLines(null)) {
      expect(line.title.trim().length).toBeGreaterThan(0);
      expect(line.unlocks.trim().length).toBeGreaterThan(0);
    }
  });
});

describe('thresholdsFromRequirements — fail-closed extraction', () => {
  const ok: RequirementsResponse = {
    outcome: 'requirements',
    thresholds: SENTINEL,
    catalog: [],
    selected: [],
    union_gates: [],
    requires_labels: false,
    summary: '',
  };

  it('returns the thresholds only for a genuine requirements outcome', () => {
    expect(thresholdsFromRequirements(ok)).toBe(SENTINEL);
  });

  it('returns null for a null response, an error outcome, or missing thresholds', () => {
    expect(thresholdsFromRequirements(null)).toBeNull();
    expect(thresholdsFromRequirements({ ...ok, outcome: 'error' })).toBeNull();
    // Missing thresholds object (defensive against a malformed response).
    expect(thresholdsFromRequirements({ ...ok, thresholds: undefined as unknown as Thresholds })).toBeNull();
  });

  it('composes with readinessGateLines so a failed fetch shows placeholders, not numbers', () => {
    const lines = readinessGateLines(thresholdsFromRequirements({ ...ok, outcome: 'error' }));
    expect(lines.every((l) => l.requirement === THRESHOLD_PLACEHOLDER)).toBe(true);
  });
});
