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
  requires_labels: boolean;
  gates: GateRequirement[];
  summary: string;
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
  // Python-composed, rendered VERBATIM (two-tier). The full workspace deep-link of
  // the created experiment ('' when the host could not be resolved) and a copy-paste
  // tracing snippet. The client builds NEITHER — no host, path, or snippet is
  // constructed in TS (a workspace value in TS would break reusability).
  experiment_url?: string;
  tracing_hint?: string;
  annotations_table?: string;
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

// --- Free-form requirements intake (slice 2) -------------------------------
// The engine (ail.requirements) owns every routing / kind / target fact; these
// shapes are the JSON it prints, rendered VERBATIM. Nothing here re-derives which
// dimension is a judge vs a metric, nor invents a target — two-tier discipline.

export interface PreviewedDimension {
  name: string;
  description: string;
  user_priority: number;
  kind: string; // 'deterministic_l0' | 'memalign_judge' — from Python
  role: string; // 'objective' | 'guardrail' — from Python
  metric: string | null;
  judge_name: string | null;
  direction: string; // 'minimize' | 'maximize' — from Python
}

// The composed objective target — a SUGGESTION the human must set/acknowledge.
// `value` is Python's signed relative default; the client pre-fills an editable
// field with it (never a TS constant) and sends back whatever the human commits.
export interface SuggestedTarget {
  value: number;
  kind: string;
  is_suggestion: boolean;
}

export interface RequirementsPreviewResponse {
  outcome: 'requirements_preview' | 'error';
  requirements_text?: string;
  cohort?: string;
  agent_name?: string;
  describe?: string;
  objective_metric?: string;
  direction?: string;
  requires_quality?: boolean;
  dimensions?: PreviewedDimension[];
  judges_to_author?: string[];
  deterministic_metrics?: string[];
  suggested_target?: SuggestedTarget | null;
  error?: string | null;
  action?: string; // present on an ErrorResult
}

export interface RequirementsConfirmResponse {
  outcome: 'requirements_confirmed' | 'refused' | 'error';
  agent_name?: string;
  experiment_id?: string;
  cohort?: string;
  objective_metric?: string;
  objective_target?: number | null;
  authored_judges?: string[];
  persisted?: boolean;
  // The confirmed goal, serialized by Python to the registry `goal_config` shape (the
  // keys the continuous-RLM lane reads). The wizard threads this onto the register
  // payload so a requirements-confirmed goal steers the loop; null on non-success.
  // Opaque here (two-tier: Python owns the goal knobs) — relayed verbatim, never built.
  goal_config?: Record<string, unknown> | null;
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
    description: 'Connect the experiment and describe the agent, project, and requirements in one place.',
  },
  {
    key: 'goals',
    title: 'Goals',
    description: 'Choose built-in goals or turn your requirements into custom feedback-trained judges.',
  },
  {
    key: 'data_gate',
    title: 'Data gates',
    description: 'Accept that optimization will not act until the readiness gates are met.',
  },
  {
    key: 'register',
    title: 'Register',
    description: 'Review the complete setup and register it in the agent switcher.',
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
  reviewerExperimentId: string;
  goals: string[];
  accepted: boolean;
  agentName: string;
  // The repo/path the open-ended executor edits — REQUIRED for the executor to run
  // this agent (an AGENT_TASK against an agent with no target_workspace fails closed).
  targetWorkspace: string;
  // Explicit project-relative file that an approved GEPA candidate may rewrite.
  // The companion resolves it under targetWorkspace and verifies hashes before use.
  optimizationTargetPath: string;
  optimizationValidationCommand: string;
  // The fully-qualified OTEL annotations table the memory-distiller job reads —
  // REQUIRED for the memory job (it skips an agent with no annotations_table).
  annotationsTable: string;
  // The confirmed requirements goal in the registry goal_config shape, when the
  // requirements path was used (else null → the catalog path, RLM neutral). Opaque:
  // set verbatim from the confirm response, threaded onto register — never built here.
  goalConfig: Record<string, unknown> | null;
  // Natural-language intake is part of the catalog flow. Quality dimensions are
  // authored as MemAlign judges only after explicit preview + confirmation.
  requirementsText: string;
  customJudgeNames: string[];
}

export const initialWizardState: WizardState = {
  stepIndex: 0,
  experimentMode: 'validate',
  experimentIdInput: '',
  experimentNameInput: '',
  resolved: null,
  reviewerExperimentId: '',
  goals: [],
  accepted: false,
  agentName: '',
  targetWorkspace: '',
  optimizationTargetPath: '',
  optimizationValidationCommand: '',
  annotationsTable: '',
  goalConfig: null,
  requirementsText: '',
  customJudgeNames: [],
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
      if (!state.resolved?.fresh) return { ok: false, reason: 'Validate or create a fresh experiment to continue.' };
      return state.agentName.trim() ? { ok: true, reason: null } : { ok: false, reason: 'Name the agent to continue.' };
    case 'goals':
      return state.goals.length > 0 || state.goalConfig !== null
        ? { ok: true, reason: null }
        : { ok: false, reason: 'Select a catalog goal or confirm your custom goals.' };
    case 'data_gate':
      return state.accepted
        ? { ok: true, reason: null }
        : { ok: false, reason: 'Accept the data prerequisites to continue.' };
    case 'register':
      if (!state.agentName.trim()) return { ok: false, reason: 'Enter a unique agent name.' };
      if (!state.reviewerExperimentId.trim()) return { ok: false, reason: 'Create the isolated reviewer experiment.' };
      if (Boolean(state.optimizationTargetPath.trim()) !== Boolean(state.optimizationValidationCommand.trim())) {
        return {
          ok: false,
          reason: 'Provide both the GEPA target path and its validation command, or leave both blank.',
        };
      }
      return { ok: true, reason: null };
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

// The extended registry fields the wizard CAPTURES and threads onto the register
// payload — never fabricated. `target_workspace` (executor) + `annotations_table`
// (memory job) are what make the agent fully functional; `goal_config` is the
// requirements-confirmed goal (opaque — relayed verbatim from Python, never built in
// TS). All optional; the actor is NEVER sent (the server resolves it authenticated).
export interface RegisterExtras {
  targetWorkspace?: string;
  annotationsTable?: string;
  goalConfig?: Record<string, unknown> | null;
  reviewerExperimentId?: string;
  optimizationTargetPath?: string;
  optimizationValidationCommand?: string;
}

export interface RegisterBody {
  agent_name: string;
  experiment_id: string;
  goals: string[];
  target_workspace?: string;
  annotations_table?: string;
  goal_config?: Record<string, unknown>;
  reviewer_experiment_id?: string;
  optimization_target?: {
    kind: 'claude_skill';
    path: string;
    validation_command: string;
  };
}

export const registerBody = (
  agentName: string,
  experimentId: string,
  goals: readonly string[],
  extras: RegisterExtras = {}
): RegisterBody => {
  const body: RegisterBody = {
    agent_name: agentName.trim(),
    experiment_id: experimentId.trim(),
    goals: [...goals],
  };
  // Omit a blank/absent field so Python persists it as None (a
  // registered-but-not-fully-functional agent) rather than an empty-string table/path.
  const targetWorkspace = extras.targetWorkspace?.trim();
  if (targetWorkspace) body.target_workspace = targetWorkspace;
  const annotationsTable = extras.annotationsTable?.trim();
  if (annotationsTable) body.annotations_table = annotationsTable;
  if (extras.goalConfig) body.goal_config = extras.goalConfig;
  const reviewerExperimentId = extras.reviewerExperimentId?.trim();
  if (reviewerExperimentId) body.reviewer_experiment_id = reviewerExperimentId;
  const optimizationTargetPath = extras.optimizationTargetPath?.trim();
  const optimizationValidationCommand = extras.optimizationValidationCommand?.trim();
  if (optimizationTargetPath && optimizationValidationCommand) {
    body.optimization_target = {
      kind: 'claude_skill',
      path: optimizationTargetPath,
      validation_command: optimizationValidationCommand,
    };
  }
  return body;
};

export const previewRequirementsBody = (
  requirementsText: string,
  agentName: string
): { requirements_text: string; agent_name: string } => ({
  requirements_text: requirementsText.trim(),
  agent_name: agentName.trim(),
});

// The confirm body carries the human's target VERBATIM — the client never computes,
// clamps, or defaults it here (two-tier: Python validates the sign against the
// derived direction and fails closed on a mismatch).
export const confirmRequirementsBody = (
  requirementsText: string,
  agentName: string,
  experimentId: string,
  objectiveTarget: number
): { requirements_text: string; agent_name: string; experiment_id: string; objective_target: number } => ({
  requirements_text: requirementsText.trim(),
  agent_name: agentName.trim(),
  experiment_id: experimentId.trim(),
  objective_target: objectiveTarget,
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

// The ready-to-use details of a freshly CREATED experiment, relayed VERBATIM from
// Python (two-tier): the workspace deep-link and the tracing snippet. Null unless the
// creation actually succeeded — never a fabricated link. The client authors NEITHER
// string: no host, path, or snippet is constructed in TS. An absent url/hint (e.g. the
// host was unresolvable server-side) reads back as '' — still no TS-authored value.
export interface CreationDetails {
  url: string;
  hint: string;
}

export function creationDetails(resp: CreationResponse): CreationDetails | null {
  if (resp.outcome !== 'created') return null;
  return { url: resp.experiment_url ?? '', hint: resp.tracing_hint ?? '' };
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

// The initial catalog response carries each goal's Python-computed gate bundle.
// Projecting a checkbox selection from that cache makes Data Gates update in the
// same render, without another serverless onboarding job. All labels, thresholds,
// needed strings, and summaries remain verbatim Python outputs.
export function requirementsForGoals(
  req: RequirementsResponse | null,
  goals: readonly string[]
): RequirementsResponse | null {
  if (!req) return null;
  const chosen = new Set(goals);
  const selected: GoalRequirement[] = req.catalog
    .filter((goal) => chosen.has(goal.key))
    .map((goal) => ({
      key: goal.key,
      label: goal.label,
      objective_metric: goal.objective_metric,
      scorer: goal.scorer,
      scorer_kind: goal.scorer_kind,
      requires_quality: goal.requires_quality,
      requires_labels: goal.requires_labels,
      guardrail_judges: goal.guardrail_judges,
      optional_quality_judge: goal.optional_quality_judge,
      gates: goal.gates,
      summary: goal.summary,
    }));
  const gates = new Map<string, GateRequirement>();
  for (const goal of selected) for (const gate of goal.gates) if (!gates.has(gate.name)) gates.set(gate.name, gate);
  return {
    ...req,
    selected,
    union_gates: [...gates.values()],
    requires_labels: selected.some((goal) => goal.requires_labels),
    summary: selected
      .map((goal) => goal.summary)
      .filter(Boolean)
      .join(' '),
  };
}

// ---------------------------------------------------------------------------
// Free-form requirements plan view model (two-tier: rendered VERBATIM from Python)
// ---------------------------------------------------------------------------

export interface RequirementsPlanView {
  describe: string; // plan.describe() text, verbatim
  objectiveMetric: string;
  direction: string;
  requiresQuality: boolean;
  dimensions: PreviewedDimension[]; // routed dimensions, verbatim
  judgesToAuthor: string[]; // judge names Python would author
  deterministicMetrics: string[]; // L0 metric names (no judge)
  suggestedTarget: SuggestedTarget | null; // Python's suggestion, NOT a TS default
}

// Project a preview response into the render model. Every routing/target fact is
// passed through VERBATIM: the client authors no `kind`, no `direction`, no
// threshold, and — crucially — no target. The suggested target is whatever Python
// sent (or null); the UI pre-fills the editable field from it, never a constant.
export function requirementsPlanView(resp: RequirementsPreviewResponse): RequirementsPlanView {
  return {
    describe: resp.describe ?? '',
    objectiveMetric: resp.objective_metric ?? '',
    direction: resp.direction ?? '',
    requiresQuality: resp.requires_quality ?? false,
    dimensions: resp.dimensions ?? [],
    judgesToAuthor: resp.judges_to_author ?? [],
    deterministicMetrics: resp.deterministic_metrics ?? [],
    suggestedTarget: resp.suggested_target ?? null,
  };
}

// The preview verdict, honestly: an engine error (garbage blob, mis-mapped metric,
// unconfigured LLM) is surfaced verbatim — never a fabricated plan.
export function previewRequirementsMessage(resp: RequirementsPreviewResponse): ToneMessage | null {
  if (resp.outcome === 'error') {
    return { tone: 'error', text: resp.error ?? 'Could not preview the requirements.' };
  }
  return null;
}

// The confirm result, honestly: success names the authored judges and whether the
// goal was persisted; a refusal (anonymous actor / no target) or an error (wrong-sign
// target / infra) is never dressed up as a confirmation.
export function confirmRequirementsMessage(resp: RequirementsConfirmResponse): ToneMessage {
  switch (resp.outcome) {
    case 'requirements_confirmed': {
      const judges = (resp.authored_judges ?? []).join(', ') || 'none';
      const persisted = resp.persisted ? 'and the goal was persisted to the loop' : 'but the goal was NOT persisted';
      return { tone: 'success', text: `Authored judges: ${judges} — ${persisted}.` };
    }
    case 'refused':
      return { tone: 'error', text: `Not confirmed — ${resp.refused_reason ?? 'a fail-closed check blocked it'}.` };
    default:
      return { tone: 'error', text: `Error — ${resp.error ?? 'the requirements could not be confirmed'}.` };
  }
}
