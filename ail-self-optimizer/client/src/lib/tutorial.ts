// Content model + pure view logic for the in-app "How it works" tutorial, kept as
// pure functions/data so the step model, navigation, and the readiness-gate view are
// unit-testable without a DOM (mirrors lib/onboarding.ts). The TutorialGuide
// component is a thin renderer over these. The content mirrors docs/GETTING_STARTED.md.
//
// LOAD-BEARING TWO-TIER DISCIPLINE (docs/OBSERVABILITY_APP.md; the exact thing a
// reviewer blocked in PR #46): NO readiness THRESHOLD NUMBER is authored here. The
// gate rows below carry only explanatory prose (what a gate means / what it unlocks)
// plus the NAME of the Python threshold field to read. The actual floor numbers are
// rendered VERBATIM from the Python engine's /api/onboarding/requirements
// `thresholds` object at display time (see `readinessGateLines`), and are never
// hardcoded, invented, or branched-on by their magnitude in TypeScript. A missing /
// failed fetch renders a neutral placeholder — never a fabricated number.

import type { RequirementsResponse, Thresholds } from './onboarding';

// ---------------------------------------------------------------------------
// The loop at a glance — the cycle stages, as data so the renderer can draw a
// lightweight inline flow (styled chips + arrows, no diagram dependency).
// ---------------------------------------------------------------------------

export interface LoopStage {
  key: string;
  label: string;
  detail: string;
  // Coarse role, so the renderer can tint the human-in-the-loop stage distinctly
  // without the content model knowing about colors.
  role: 'agent' | 'measure' | 'control' | 'human' | 'apply';
}

// agent → traces → L0/L2/L3 → controller (detect/prove/gate/propose) → YOUR approval
// → gated apply → auditable lineage → back to the agent (GETTING_STARTED §"60-second
// overview" + §6). The renderer closes the loop visually back to the first stage.
export const LOOP_STAGES: readonly LoopStage[] = [
  { key: 'agent', label: 'Your agent', detail: 'Claude Code · Codex · anything', role: 'agent' },
  { key: 'traces', label: 'Traces', detail: 'land in one MLflow experiment', role: 'measure' },
  { key: 'measure', label: 'L0 · L2 · L3', detail: 'metrics · judges · recursive review', role: 'measure' },
  { key: 'controller', label: 'Controller', detail: 'detect → prove → gate → propose', role: 'control' },
  { key: 'approval', label: 'Your approval', detail: 'review the why + proof, then approve or reject', role: 'human' },
  { key: 'apply', label: 'Gated apply', detail: 'fail-closed, auditable lineage', role: 'apply' },
] as const;

// ---------------------------------------------------------------------------
// The three measurement layers (GETTING_STARTED §5 stages 1/2/4). Explanatory
// prose only — no threshold numbers.
// ---------------------------------------------------------------------------

export interface MeasurementLayer {
  key: string;
  name: string;
  tagline: string;
  detail: string;
}

export const MEASUREMENT_LAYERS: readonly MeasurementLayer[] = [
  {
    key: 'l0',
    name: 'L0 — deterministic',
    tagline: 'un-gameable',
    detail:
      'Tokens, cost, latency, tool-call count, redundancy — mechanically derived from trace metadata, no model in the loop. Your irrefutable baseline; it works immediately with just trace access.',
  },
  {
    key: 'l2',
    name: 'L2 — judges',
    tagline: 'distrusted by default',
    detail:
      'LLM-judge scorers (correctness, efficiency, groundedness…) aligned to YOUR human labels via MemAlign. A judge is untrusted until its agreement with human labels clears the floor — then re-distrusted if it drifts.',
  },
  {
    key: 'l3',
    name: 'L3 — RLM / HALO',
    tagline: 'reads huge traces',
    detail:
      'A trace-specialized recursive reviewer reads 500K-token coding-agent traces that a single judge call or a human cannot, and emits a structured verdict plus ranked, deployable fix recommendations.',
  },
] as const;

// ---------------------------------------------------------------------------
// The tutorial steps — a guided walkthrough (mirrors the wizard's stepper). Each
// step's key selects which content block the renderer shows; the prose here is
// explanatory only.
// ---------------------------------------------------------------------------

export type TutorialStepKey = 'loop' | 'connect' | 'readiness' | 'approval';

export interface TutorialStep {
  key: TutorialStepKey;
  title: string;
  tagline: string;
  // Short body paragraphs (explanatory prose — never a threshold number).
  body: string[];
  // Optional supporting bullets.
  points?: string[];
}

export const TUTORIAL_STEPS: readonly TutorialStep[] = [
  {
    key: 'loop',
    title: 'The loop at a glance',
    tagline: 'What actually happens between your agent and a shipped improvement',
    body: [
      'Your agent emits traces; three measurement layers turn those traces into evidence; an autonomous controller detects a fix, proves it on a frozen test set, gates it on readiness, and proposes it — but nothing reaches your live agent until you approve it in this app.',
      'Everything above the approval step is autonomous. The apply is fail-closed and recorded as auditable lineage, so any change can be reverted.',
    ],
  },
  {
    key: 'connect',
    title: 'Connect, don’t upload',
    tagline: 'The system improves only what it can measure',
    body: [
      'There is no “upload your agent” button. You point the loop at the MLflow experiment your agent already logs to — via native autolog or OpenTelemetry — and it reads those traces read-only.',
      'The optimizer improves only what it can measure, and it can only measure what has been traced. Give each agent its own experiment: that is how each gets its own judges and its own scoring.',
    ],
    points: [
      'Option 1 — native autolog (Claude Code, OpenAI, LangChain, DSPy, Codex).',
      'Option 2 — OpenTelemetry: export live, or import a backlog.',
      '0 traces → 0 trustworthy optimization. Until enough data exists the loop says “collecting — not ready yet” instead of a green number that isn’t real.',
    ],
  },
  {
    key: 'readiness',
    title: 'Readiness gates',
    tagline: 'Amber, never green until proven',
    body: [
      'Different goals need different amounts of data. The loop refuses to claim improvement until the gate for your goal is met, and tells you exactly how far away you are.',
      'A real delta is shown as soon as it exists, but it stays amber — never styled as a win — until it has been proven on the frozen suite and cleared the trust gates.',
    ],
  },
  {
    key: 'approval',
    title: 'You approve every change',
    tagline: 'Autonomous up to approval — the human control plane',
    body: [
      'The framework detects, decides, proves, and proposes on its own — but a change reaches your live agent only when you approve it here, with the evidence in front of you: the WHY (what triggered it), the PROOF (a frozen-suite delta with correctness held), and the GATE status.',
      'Approve triggers the gated apply behind a fail-closed wall (the engine re-checks the proof and gate at apply time); reject records the reason. This is the app’s only write-path, and every applied change is recorded and revertible.',
    ],
  },
] as const;

// ---------------------------------------------------------------------------
// Pure navigation helpers (mirror lib/onboarding.ts's clampStep / isLastStep).
// ---------------------------------------------------------------------------

export const clampTutorialStep = (index: number): number => Math.max(0, Math.min(index, TUTORIAL_STEPS.length - 1));

export const isFirstTutorialStep = (index: number): boolean => index <= 0;

export const isLastTutorialStep = (index: number): boolean => index >= TUTORIAL_STEPS.length - 1;

// Progress through the walkthrough, 1-based over the total, as a percentage — for
// the same <Progress /> bar the wizard's stepper uses.
export const tutorialProgressPct = (index: number): number =>
  ((clampTutorialStep(index) + 1) / TUTORIAL_STEPS.length) * 100;

// ---------------------------------------------------------------------------
// Readiness-gate view (TWO-TIER: every threshold number comes from Python)
// ---------------------------------------------------------------------------

// Which field of the Python `thresholds` object each gate reads. This name — never a
// literal number — is the only threshold reference authored in TS.
type ThresholdField = keyof Thresholds;

export interface ReadinessGate {
  key: string;
  // What unlocks at this gate (prose — GETTING_STARTED §2).
  title: string;
  unlocks: string;
  // The Python threshold field whose value is the requirement, rendered verbatim.
  field: ThresholdField;
  // The unit noun for a count gate ('' for the coverage fraction).
  unit: string;
  // The coverage floor is a fraction the doc/Python present as a percentage; this
  // flags a display-only transform of the Python value (never a re-derivation).
  asPercent: boolean;
}

// The gates and what each unlocks (GETTING_STARTED §2). The `field` names map each to
// its Python-owned floor; NO magnitude is written here.
export const READINESS_GATES: readonly ReadinessGate[] = [
  {
    key: 'baseline',
    title: 'Baseline & diagnosis',
    unlocks: 'Unlocks the L0 baseline and RLM diagnosis — the deterministic picture of where the waste is.',
    field: 'baseline_min_traces',
    unit: 'traces',
    asPercent: false,
  },
  {
    key: 'prove',
    title: 'Prove a token / cost win',
    unlocks: 'Enough traces to run the controlled WITH-vs-WITHOUT proof; only a proven win goes green.',
    field: 'prove_min_traces',
    unit: 'traces',
    asPercent: false,
  },
  {
    key: 'labels',
    title: 'Trust a quality judge',
    unlocks: 'Human labels to align the MemAlign judge before any quality/accuracy claim is trusted.',
    field: 'quality_min_labels',
    unit: 'human labels',
    asPercent: false,
  },
  {
    key: 'coverage',
    title: 'Judge-agreement floor',
    unlocks: 'Judge-vs-human agreement must clear this floor, or the judge is re-distrusted and no quality is claimed.',
    field: 'scored_coverage_floor',
    unit: '',
    asPercent: true,
  },
] as const;

// The neutral placeholder shown when the Python thresholds have not loaded (or the
// fetch failed). Never a number — honesty over a fabricated floor.
export const THRESHOLD_PLACEHOLDER = '—';

export interface ReadinessGateLine {
  key: string;
  title: string;
  unlocks: string;
  // The requirement, with the threshold sourced verbatim from Python — or the
  // neutral placeholder when it is not loaded.
  requirement: string;
  loaded: boolean;
}

// Format a Python-sourced threshold value into the requirement phrase. The NUMBER is
// the Python value rendered verbatim; the surrounding words ("≥", the unit noun, the
// "%" for a fraction) are explanatory scaffold. The percentage is a display-only
// transform of the fraction (matching the Python `_gate_phrase` presentation) — the
// value is not branched-on by magnitude, only formatted.
function formatRequirement(gate: ReadinessGate, value: number): string {
  if (gate.asPercent) return `≥ ${Math.round(value * 100)}%`;
  return `≥ ${value}${gate.unit ? ` ${gate.unit}` : ''}`;
}

// Build the readiness-gate rows for the tutorial. When `thresholds` is null (not yet
// loaded, or the fetch failed) every row shows the neutral placeholder — never an
// invented number. This is the ONLY place the tutorial surfaces a threshold number,
// and it reads each one from the Python `thresholds` object by field name.
export function readinessGateLines(thresholds: Thresholds | null): ReadinessGateLine[] {
  return READINESS_GATES.map((gate) => {
    if (!thresholds) {
      return {
        key: gate.key,
        title: gate.title,
        unlocks: gate.unlocks,
        requirement: THRESHOLD_PLACEHOLDER,
        loaded: false,
      };
    }
    return {
      key: gate.key,
      title: gate.title,
      unlocks: gate.unlocks,
      requirement: formatRequirement(gate, thresholds[gate.field]),
      loaded: true,
    };
  });
}

// Fail-closed extraction of the Python thresholds from a requirements response: only
// a genuine `requirements` outcome with a thresholds object yields numbers. An error
// outcome, a null response, or a missing thresholds object yields null → the tutorial
// shows the neutral placeholder rather than a fabricated floor.
export function thresholdsFromRequirements(resp: RequirementsResponse | null): Thresholds | null {
  if (!resp || resp.outcome !== 'requirements' || !resp.thresholds) return null;
  return resp.thresholds;
}
