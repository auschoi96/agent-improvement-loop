// Onboarding-wizard view logic (slice 1), kept as pure functions so the stepper
// state machine, per-step validation, request builders, and response→message
// mapping are unit-testable without a DOM (mirrors lib/approvals.ts). The
// OnboardingWizard component is a thin renderer over these.
//
// Two-tier discipline (docs/OBSERVABILITY_APP.md): the goal→scorer mapping and the
// readiness floors/gates are NOT modeled here — they are fetched from the Python
// engine (/api/onboarding/requirements) and rendered verbatim. This file only holds
// the client-side stepper state and the honest message mapping.

// ---------------------------------------------------------------------------
// Server response shapes (the typed JSON ail.onboarding.service prints)
// ---------------------------------------------------------------------------

export interface GoalOption {
  key: string;
  label: string;
  objective_metric: string;
  scorer: string;
  scorer_kind: string;
  requires_quality: boolean;
  guardrail_judges: string[];
  optional_quality_judge: string | null;
  description: string;
}

export interface GateRequirement {
  name: string;
  label: string;
  needed: string;
  threshold: number | null;
}

export interface GoalRequirement {
  key: string;
  label: string;
  objective_metric: string;
  scorer: string;
  scorer_kind: string;
  requires_quality: boolean;
  requires_labels: boolean;
  guardrail_judges: string[];
  optional_quality_judge: string | null;
  gates: GateRequirement[];
  // Python-composed, human-readable gate description (with the real threshold
  // numbers). The client renders it VERBATIM — no thresholds/bundles authored in TS.
  summary: string;
}

export interface Thresholds {
  baseline_min_traces: number;
  prove_min_traces: number;
  quality_min_labels: number;
  scored_coverage_floor: number;
}

export interface RequirementsResponse {
  outcome: 'requirements' | 'error';
  thresholds: Thresholds;
  catalog: GoalOption[];
  selected: GoalRequirement[];
  union_gates: GateRequirement[];
  requires_labels: boolean;
  // Python-composed overall data-gate note (with real threshold numbers), rendered
  // verbatim by the client. Empty when no goals are selected.
  summary: string;
  error?: string;
}

export interface ValidationResponse {
  outcome: 'validated' | 'error' | 'refused';
  experiment_id: string;
  name?: string;
  exists?: boolean;
  fresh?: boolean;
  trace_count?: number;
  trace_count_capped?: boolean;
  already_registered?: boolean;
  registered_as?: string | null;
  reasons?: string[];
  error?: string | null;
  refused_reason?: string | null;
}

export interface CreationResponse {
  outcome: 'created' | 'error' | 'refused';
  experiment_id?: string;
  name?: string;
  error?: string | null;
  prerequisite?: string | null;
  refused_reason?: string | null;
}

export interface RegisterResponse {
  outcome: 'registered' | 'refused' | 'error';
  agent_name?: string;
  experiment_id?: string;
  goals?: string[];
  refused_reason?: string | null;
  error?: string | null;
}

// ---------------------------------------------------------------------------
// The stepper
// ---------------------------------------------------------------------------

export type WizardStepKey = 'experiment' | 'goals' | 'data_gate' | 'register';

export interface WizardStep {
  key: WizardStepKey;
  title: string;
  description: string;
}

// The wizard spine (docs/ONBOARDING_WIZARD.md §30). One agent = one experiment.
export const WIZARD_STEPS: readonly WizardStep[] = [
  {
    key: 'experiment',
    title: 'Experiment',
    description: 'Point at a fresh MLflow experiment, or create one — one agent per experiment.',
  },
  {
    key: 'goals',
    title: 'Goals',
    description: 'Choose what to improve. Each goal maps to a deterministic metric or a calibrated judge.',
  },
  {
    key: 'data_gate',
    title: 'Data gates',
    description: 'Accept that optimization will not act until the readiness gates are met.',
  },
  {
    key: 'register',
    title: 'Register',
    description: 'Name the agent and register it — it then appears in the agent switcher.',
  },
] as const;

export type ExperimentMode = 'validate' | 'create';

// A fresh experiment the wizard has CONFIRMED (validated fresh, or freshly created).
// The only thing that unlocks step 1 — never set on a non-fresh / errored result.
export interface ResolvedExperiment {
  experiment_id: string;
  name: string;
  fresh: boolean;
}

export interface WizardState {
  stepIndex: number;
  experimentMode: ExperimentMode;
  experimentIdInput: string;
  experimentNameInput: string;
  resolved: ResolvedExperiment | null;
  goals: string[];
  accepted: boolean;
  agentName: string;
}

export const initialWizardState: WizardState = {
  stepIndex: 0,
  experimentMode: 'validate',
  experimentIdInput: '',
  experimentNameInput: '',
  resolved: null,
  goals: [],
  accepted: false,
  agentName: '',
};

export function toggleGoal(goals: readonly string[], key: string): string[] {
  return goals.includes(key) ? goals.filter((g) => g !== key) : [...goals, key];
}

export interface StepValidation {
  ok: boolean;
  reason: string | null;
}

// Per-step gate: exactly what must be true before Next/Finish is allowed. Fail-closed
// — an unresolved / non-fresh experiment never lets the user past step 1, and the
// data-gate acceptance is a hard requirement (docs/ONBOARDING_WIZARD.md §60).
export function stepValidation(state: WizardState): StepValidation {
  const step = WIZARD_STEPS[state.stepIndex]?.key;
  switch (step) {
    case 'experiment':
      return state.resolved?.fresh
        ? { ok: true, reason: null }
        : { ok: false, reason: 'Validate or create a fresh experiment to continue.' };
    case 'goals':
      return state.goals.length > 0
        ? { ok: true, reason: null }
        : { ok: false, reason: 'Select at least one goal to improve.' };
    case 'data_gate':
      return state.accepted
        ? { ok: true, reason: null }
        : { ok: false, reason: 'Accept the data prerequisites to continue.' };
    case 'register':
      return state.agentName.trim() ? { ok: true, reason: null } : { ok: false, reason: 'Enter a unique agent name.' };
    default:
      return { ok: false, reason: 'Unknown step.' };
  }
}

export const canAdvance = (state: WizardState): boolean => stepValidation(state).ok;

export const isLastStep = (state: WizardState): boolean => state.stepIndex >= WIZARD_STEPS.length - 1;

export const clampStep = (index: number): number => Math.max(0, Math.min(index, WIZARD_STEPS.length - 1));

// ---------------------------------------------------------------------------
// Request bodies (the actor is NEVER sent — the server resolves it from the
// authenticated request; a body actor would be ignored by the route).
// ---------------------------------------------------------------------------

export const requirementsBody = (goals: readonly string[]): { goals: string[] } => ({
  goals: [...goals],
});

export const validateExperimentBody = (experimentId: string): { experiment_id: string } => ({
  experiment_id: experimentId.trim(),
});

export const createExperimentBody = (name: string): { name: string } => ({ name: name.trim() });

export const registerBody = (
  agentName: string,
  experimentId: string,
  goals: readonly string[]
): { agent_name: string; experiment_id: string; goals: string[] } => ({
  agent_name: agentName.trim(),
  experiment_id: experimentId.trim(),
  goals: [...goals],
});

// ---------------------------------------------------------------------------
// Honest message mapping
// ---------------------------------------------------------------------------

export type Tone = 'success' | 'warning' | 'error' | 'info';

export interface ToneMessage {
  tone: Tone;
  text: string;
}

// The freshness verdict, honestly: only a genuinely-fresh experiment is a success.
// A non-fresh one is a WARNING that names exactly why (prior traces / already
// claimed); an access failure is an ERROR — never dressed up as fresh.
export function freshnessMessage(resp: ValidationResponse): ToneMessage {
  if (resp.outcome === 'error') {
    return { tone: 'error', text: resp.error ?? 'Could not verify the experiment.' };
  }
  if (resp.outcome === 'refused') {
    return { tone: 'error', text: resp.refused_reason ?? 'Refused.' };
  }
  if (resp.fresh) {
    return {
      tone: 'success',
      text: `Fresh experiment "${resp.name || resp.experiment_id}" — ready to use.`,
    };
  }
  const why = (resp.reasons ?? []).join(' ') || 'The experiment is not fresh.';
  return { tone: 'warning', text: why };
}

export function creationMessage(resp: CreationResponse): ToneMessage {
  if (resp.outcome === 'created') {
    return {
      tone: 'success',
      text: `Created experiment ${resp.experiment_id}${resp.name ? ` ("${resp.name}")` : ''}.`,
    };
  }
  if (resp.outcome === 'refused') {
    return { tone: 'error', text: resp.refused_reason ?? 'Refused.' };
  }
  const prereq = resp.prerequisite ? ` Prerequisite: ${resp.prerequisite}.` : '';
  return { tone: 'error', text: `${resp.error ?? 'Could not create the experiment.'}${prereq}` };
}

export function registerMessage(resp: RegisterResponse): ToneMessage {
  switch (resp.outcome) {
    case 'registered':
      return {
        tone: 'success',
        text: `Registered "${resp.agent_name}" — it now appears in the agent switcher.`,
      };
    case 'refused':
      return {
        tone: 'error',
        text: `Not registered — ${resp.refused_reason ?? 'a fail-closed check blocked it'}.`,
      };
    default:
      return { tone: 'error', text: `Error — ${resp.error ?? 'the agent could not be registered'}.` };
  }
}

// A validation response only resolves a fresh experiment when it is actually fresh.
export function resolvedFromValidation(resp: ValidationResponse): ResolvedExperiment | null {
  if (resp.outcome === 'validated' && resp.fresh) {
    return { experiment_id: resp.experiment_id, name: resp.name ?? '', fresh: true };
  }
  return null;
}

export function resolvedFromCreation(resp: CreationResponse): ResolvedExperiment | null {
  if (resp.outcome === 'created' && resp.experiment_id) {
    return { experiment_id: resp.experiment_id, name: resp.name ?? '', fresh: true };
  }
  return null;
}

// ---------------------------------------------------------------------------
// Data-gate view model (two-tier: rendered VERBATIM from the Python response)
// ---------------------------------------------------------------------------

export interface DataGatePerGoal {
  key: string;
  label: string;
  scorer: string;
  summary: string;
}

export interface DataGateView {
  summary: string; // overall note, verbatim from Python
  gates: GateRequirement[]; // union gates, rendered verbatim (label + needed)
  perGoal: DataGatePerGoal[]; // per-goal descriptions, verbatim from Python
}

// Build the data-gate page's view model from the Python `requirements` response.
// Every gate fact is passed through VERBATIM — the overall note, the per-gate
// `needed` strings (which carry the real threshold numbers), and each goal's
// `summary`. No readiness threshold or gate-bundle text is ever authored in
// TypeScript, and nothing is derived from `requires_labels` (two-tier discipline).
export function dataGateView(req: RequirementsResponse): DataGateView {
  return {
    summary: req.summary ?? '',
    gates: req.union_gates,
    perGoal: req.selected.map((g) => ({
      key: g.key,
      label: g.label,
      scorer: g.scorer,
      summary: g.summary ?? '',
    })),
  };
}
