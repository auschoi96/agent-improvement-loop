import { describe, it, expect } from 'vitest';
import {
  WIZARD_STEPS,
  initialWizardState,
  toggleGoal,
  stepValidation,
  canAdvance,
  isLastStep,
  clampStep,
  requirementsBody,
  validateExperimentBody,
  createExperimentBody,
  registerBody,
  previewRequirementsBody,
  confirmRequirementsBody,
  freshnessMessage,
  creationMessage,
  creationDetails,
  registerMessage,
  requirementsPlanView,
  requirementsForGoals,
  previewRequirementsMessage,
  confirmRequirementsMessage,
  resolvedFromValidation,
  resolvedFromCreation,
  dataGateView,
  type RequirementsPreviewResponse,
  type RequirementsResponse,
  type WizardState,
} from './onboarding';

const FRESH = { experiment_id: 'exp-1', name: 'agent', fresh: true };

function stateAt(stepKey: (typeof WIZARD_STEPS)[number]['key'], over: Partial<WizardState> = {}): WizardState {
  const stepIndex = WIZARD_STEPS.findIndex((s) => s.key === stepKey);
  return { ...initialWizardState, stepIndex, ...over };
}

describe('toggleGoal', () => {
  it('adds an unselected goal and removes a selected one', () => {
    expect(toggleGoal([], 'cost')).toEqual(['cost']);
    expect(toggleGoal(['cost'], 'accuracy')).toEqual(['cost', 'accuracy']);
    expect(toggleGoal(['cost', 'accuracy'], 'cost')).toEqual(['accuracy']);
  });
});

describe('stepValidation — fail-closed per step', () => {
  it('experiment step needs a resolved FRESH experiment and agent name', () => {
    expect(stepValidation(stateAt('experiment')).ok).toBe(false);
    expect(stepValidation(stateAt('experiment', { resolved: { ...FRESH, fresh: false } })).ok).toBe(false);
    expect(stepValidation(stateAt('experiment', { resolved: FRESH })).ok).toBe(false);
    expect(stepValidation(stateAt('experiment', { resolved: FRESH, agentName: 'my_agent' })).ok).toBe(true);
  });

  it('goals step needs at least one goal', () => {
    expect(stepValidation(stateAt('goals')).ok).toBe(false);
    expect(canAdvance(stateAt('goals', { goals: ['cost'] }))).toBe(true);
    expect(canAdvance(stateAt('goals', { goalConfig: { objective_metric: 'custom_quality' } }))).toBe(true);
  });

  it('data-gate step needs explicit acceptance', () => {
    expect(stepValidation(stateAt('data_gate')).ok).toBe(false);
    expect(stepValidation(stateAt('data_gate')).reason).toMatch(/accept/i);
    expect(canAdvance(stateAt('data_gate', { accepted: true }))).toBe(true);
  });

  it('register step needs a non-empty agent name', () => {
    expect(canAdvance(stateAt('register', { agentName: '   ' }))).toBe(false);
    expect(canAdvance(stateAt('register', { agentName: 'my_agent', reviewerExperimentId: 'review-exp' }))).toBe(true);
  });
});

describe('stepper navigation helpers', () => {
  it('isLastStep is only true on the final step', () => {
    expect(isLastStep(stateAt('experiment'))).toBe(false);
    expect(isLastStep(stateAt('register'))).toBe(true);
  });

  it('clampStep keeps the index in range', () => {
    expect(clampStep(-3)).toBe(0);
    expect(clampStep(99)).toBe(WIZARD_STEPS.length - 1);
    expect(clampStep(2)).toBe(2);
  });
});

describe('request builders — trim, correct shape, no actor', () => {
  it('trims inputs and never includes an actor', () => {
    expect(requirementsBody(['cost'])).toEqual({ goals: ['cost'] });
    expect(validateExperimentBody('  exp-1 ')).toEqual({ experiment_id: 'exp-1' });
    expect(createExperimentBody('  Fresh ')).toEqual({ name: 'Fresh' });
    const body = registerBody('  my_agent ', ' exp-1 ', ['accuracy', 'cost']);
    expect(body).toEqual({ agent_name: 'my_agent', experiment_id: 'exp-1', goals: ['accuracy', 'cost'] });
    expect('actor' in body).toBe(false);
  });

  it('threads the executor workspace + memory table (trimmed), never an actor', () => {
    const body = registerBody('my_agent', 'exp-1', ['cost'], {
      targetWorkspace: '  /Workspace/Repos/me/my_agent ',
      annotationsTable: ' catalog.schema.otel_annotations ',
      optimizationTargetPath: ' .claude/skills/agent/SKILL.md ',
      optimizationValidationCommand: ' python -m pytest -q ',
    });
    expect(body).toEqual({
      agent_name: 'my_agent',
      experiment_id: 'exp-1',
      goals: ['cost'],
      target_workspace: '/Workspace/Repos/me/my_agent',
      annotations_table: 'catalog.schema.otel_annotations',
      optimization_target: {
        kind: 'claude_skill',
        path: '.claude/skills/agent/SKILL.md',
        validation_command: 'python -m pytest -q',
      },
    });
    expect('actor' in body).toBe(false);
  });

  it('threads a requirements-confirmed goal_config verbatim when present', () => {
    const goalConfig = {
      objective_metric: 'no_hallucinated_tool_calls',
      goal_direction: 'maximize',
      goal_target: 0.25,
      goal_target_kind: 'relative',
      guardrail_judge: ['no_hallucinated_tool_calls:4.0'],
    };
    const body = registerBody('my_agent', 'exp-1', ['cost'], { goalConfig });
    expect(body.goal_config).toEqual(goalConfig);
  });

  it('omits blank/absent extended fields so Python persists them as None (back-compat)', () => {
    const blank = registerBody('my_agent', 'exp-1', ['cost'], {
      targetWorkspace: '   ',
      annotationsTable: '',
      goalConfig: null,
    });
    expect(blank).toEqual({ agent_name: 'my_agent', experiment_id: 'exp-1', goals: ['cost'] });
    expect('target_workspace' in blank).toBe(false);
    expect('annotations_table' in blank).toBe(false);
    expect('goal_config' in blank).toBe(false);
    // No extras at all is byte-for-byte the pre-Slice-4 shape.
    expect(registerBody('my_agent', 'exp-1', ['cost'])).toEqual({
      agent_name: 'my_agent',
      experiment_id: 'exp-1',
      goals: ['cost'],
    });
  });
});

describe('freshnessMessage — honest verdict', () => {
  it('is success only for a genuinely fresh experiment', () => {
    expect(freshnessMessage({ outcome: 'validated', experiment_id: 'e', name: 'x', fresh: true }).tone).toBe('success');
  });

  it('is a warning that names why for a non-fresh experiment', () => {
    const msg = freshnessMessage({
      outcome: 'validated',
      experiment_id: 'e',
      fresh: false,
      reasons: ['experiment already has 12 trace(s)'],
    });
    expect(msg.tone).toBe('warning');
    expect(msg.text).toMatch(/12 trace/);
  });

  it('is an error on an access failure — never dressed up as fresh', () => {
    const msg = freshnessMessage({ outcome: 'error', experiment_id: 'e', error: 'no CAN_VIEW' });
    expect(msg.tone).toBe('error');
    expect(msg.text).toMatch(/CAN_VIEW/);
  });
});

describe('creationMessage — honest, surfaces the prerequisite', () => {
  it('is success on a real creation', () => {
    const msg = creationMessage({ outcome: 'created', experiment_id: 'exp-42', name: 'Fresh' });
    expect(msg.tone).toBe('success');
    expect(msg.text).toMatch(/exp-42/);
  });

  it('is an error that names the SP prerequisite on a denied create', () => {
    const msg = creationMessage({
      outcome: 'error',
      error: 'not authorized to create',
      prerequisite: 'app SP needs experiment-create authority',
    });
    expect(msg.tone).toBe('error');
    expect(msg.text).toMatch(/experiment-create authority/);
  });
});

describe('creationDetails — url+hint relayed VERBATIM from Python (no workspace value in TS)', () => {
  it('passes the Python-provided url and hint through unchanged on a real creation', () => {
    // An arbitrary workspace host a TS constant could not plausibly be — it must come
    // from Python, proving the UI renders (not fabricates) the deep-link + snippet.
    const url = 'https://arbitrary-workspace-9f3.cloud.databricks.example/ml/experiments/exp-42';
    const hint = "mlflow.set_experiment(experiment_id='exp-42')  # then enable autolog";
    const details = creationDetails({
      outcome: 'created',
      experiment_id: 'exp-42',
      experiment_url: url,
      tracing_hint: hint,
    });
    expect(details).toEqual({ url, hint });
  });

  it('is null unless the creation succeeded — never fabricates a link', () => {
    expect(creationDetails({ outcome: 'error', error: 'denied' })).toBeNull();
    expect(creationDetails({ outcome: 'refused', refused_reason: 'anonymous' })).toBeNull();
  });

  it('tolerates an absent url/hint (host unresolvable server-side) with empty strings', () => {
    // Still no TS-authored value: an unresolvable host reads back as '' from Python.
    expect(creationDetails({ outcome: 'created', experiment_id: 'exp-9' })).toEqual({ url: '', hint: '' });
  });
});

describe('registerMessage — honest', () => {
  it('success only on a real registration', () => {
    expect(registerMessage({ outcome: 'registered', agent_name: 'my_agent' }).tone).toBe('success');
  });
  it('a refusal (e.g. duplicate) is an error, not a fake success', () => {
    const msg = registerMessage({ outcome: 'refused', refused_reason: 'already registered' });
    expect(msg.tone).toBe('error');
    expect(msg.text).toMatch(/already registered/);
  });
  it('an engine error is an error', () => {
    expect(registerMessage({ outcome: 'error', error: 'warehouse down' }).tone).toBe('error');
  });
});

describe('resolved-experiment gating', () => {
  it('only a fresh validation resolves the experiment', () => {
    expect(resolvedFromValidation({ outcome: 'validated', experiment_id: 'e', fresh: true })).not.toBeNull();
    expect(resolvedFromValidation({ outcome: 'validated', experiment_id: 'e', fresh: false })).toBeNull();
    expect(resolvedFromValidation({ outcome: 'error', experiment_id: 'e' })).toBeNull();
  });
  it('only a real creation resolves the experiment', () => {
    expect(resolvedFromCreation({ outcome: 'created', experiment_id: 'exp-9' })).not.toBeNull();
    expect(resolvedFromCreation({ outcome: 'error', error: 'denied' })).toBeNull();
  });
});

describe('dataGateView — renders Python gate facts verbatim (no TS thresholds/bundles)', () => {
  // A requirements response whose gate strings are DISTINCTIVE sentinels: if the
  // client ever re-authored the thresholds/bundle in TS instead of rendering the
  // Python strings, these exact sentinels could not appear.
  const req: RequirementsResponse = {
    outcome: 'requirements',
    thresholds: {
      baseline_min_traces: 10,
      prove_min_traces: 50,
      quality_min_labels: 20,
      scored_coverage_floor: 0.5,
    },
    catalog: [],
    requires_labels: true,
    summary: 'PY_OVERALL_NOTE_with_20_labels',
    union_gates: [
      { name: 'human_labels', label: 'Human labels to align the judge', needed: 'PY_NEEDED_20_labels', threshold: 20 },
    ],
    selected: [
      {
        key: 'accuracy',
        label: 'Accuracy',
        objective_metric: 'correctness',
        scorer: 'MemAlign judge (correctness)',
        scorer_kind: 'memalign_judge',
        requires_quality: true,
        requires_labels: true,
        guardrail_judges: ['correctness'],
        optional_quality_judge: null,
        gates: [],
        summary: 'PY_ACCURACY_SUMMARY',
      },
    ],
  };

  it('passes the overall note, per-gate needed text, and per-goal summary through verbatim', () => {
    const view = dataGateView(req);
    expect(view.summary).toBe('PY_OVERALL_NOTE_with_20_labels');
    expect(view.gates[0].needed).toBe('PY_NEEDED_20_labels');
    expect(view.perGoal[0].summary).toBe('PY_ACCURACY_SUMMARY');
  });

  it('does not derive per-goal text from requires_labels — it uses the Python summary', () => {
    // Even with requires_labels flipped, the rendered per-goal text is the Python
    // summary verbatim (proving the old hardcoded "traces + 20 labels…" bundle is gone).
    const flipped: RequirementsResponse = {
      ...req,
      selected: [{ ...req.selected[0], requires_labels: false, summary: 'PY_SUMMARY_WINS' }],
    };
    expect(dataGateView(flipped).perGoal[0].summary).toBe('PY_SUMMARY_WINS');
  });
});

describe('requirementsForGoals — instant gate projection from Python catalog facts', () => {
  it('updates the selected gates without another response and preserves Python strings verbatim', () => {
    const base: RequirementsResponse = {
      outcome: 'requirements',
      thresholds: {
        baseline_min_traces: 10,
        prove_min_traces: 50,
        quality_min_labels: 20,
        scored_coverage_floor: 0.5,
      },
      catalog: [
        {
          key: 'accuracy',
          label: 'Accuracy',
          objective_metric: 'correctness',
          scorer: 'MemAlign judge (correctness)',
          scorer_kind: 'memalign_judge',
          requires_quality: true,
          requires_labels: true,
          guardrail_judges: ['correctness'],
          optional_quality_judge: null,
          description: 'PY_DESCRIPTION',
          gates: [{ name: 'human_labels', label: 'Human labels', needed: 'PY_NEED_LABELS', threshold: 20 }],
          summary: 'PY_GOAL_SUMMARY',
        },
      ],
      selected: [],
      union_gates: [],
      requires_labels: false,
      summary: '',
    };

    const selected = requirementsForGoals(base, ['accuracy']);
    expect(selected?.union_gates[0].needed).toBe('PY_NEED_LABELS');
    expect(selected?.selected[0].summary).toBe('PY_GOAL_SUMMARY');
    expect(selected?.requires_labels).toBe(true);
  });
});

describe('requirementsPlanView — the routed plan is rendered from Python, nothing derived in TS', () => {
  // A preview whose routing facts are DISTINCTIVE sentinels: if the client ever
  // re-derived a dimension's kind/role/direction, the objective, the judge/metric
  // split, or the target instead of rendering the Python response, these exact
  // sentinels could not appear.
  const preview: RequirementsPreviewResponse = {
    outcome: 'requirements_preview',
    requirements_text: 'be great',
    agent_name: 'my_agent',
    describe: 'PY_DESCRIBE_TEXT confirmed=False',
    objective_metric: 'PY_OBJECTIVE_METRIC',
    direction: 'PY_DIRECTION',
    requires_quality: true,
    dimensions: [
      {
        name: 'no hallucinated tool calls',
        description: 'never invent tools',
        user_priority: 1,
        kind: 'memalign_judge',
        role: 'objective',
        metric: null,
        judge_name: 'PY_JUDGE_NAME',
        direction: 'maximize',
      },
      {
        name: 'latency',
        description: 'be fast',
        user_priority: 2,
        kind: 'deterministic_l0',
        role: 'guardrail',
        metric: 'PY_METRIC_NAME',
        judge_name: null,
        direction: 'minimize',
      },
    ],
    judges_to_author: ['PY_JUDGE_NAME'],
    deterministic_metrics: ['PY_METRIC_NAME'],
    // A distinctive, non-round SUGGESTED target: a hardcoded TS default could not
    // reproduce it — it can only come from the response.
    suggested_target: { value: 0.4242, kind: 'PY_TARGET_KIND', is_suggestion: true },
  };

  it('passes objective, dimensions, kinds/roles/directions, split, and describe through verbatim', () => {
    const view = requirementsPlanView(preview);
    expect(view.objectiveMetric).toBe('PY_OBJECTIVE_METRIC');
    expect(view.direction).toBe('PY_DIRECTION');
    expect(view.requiresQuality).toBe(true);
    expect(view.describe).toBe('PY_DESCRIBE_TEXT confirmed=False');
    expect(view.dimensions.map((d) => [d.role, d.kind, d.direction, d.metric ?? d.judge_name])).toEqual([
      ['objective', 'memalign_judge', 'maximize', 'PY_JUDGE_NAME'],
      ['guardrail', 'deterministic_l0', 'minimize', 'PY_METRIC_NAME'],
    ]);
    expect(view.judgesToAuthor).toEqual(['PY_JUDGE_NAME']);
    expect(view.deterministicMetrics).toEqual(['PY_METRIC_NAME']);
  });

  it('the suggested target comes from the response (a TS constant could not produce 0.4242)', () => {
    const view = requirementsPlanView(preview);
    expect(view.suggestedTarget).toEqual({ value: 0.4242, kind: 'PY_TARGET_KIND', is_suggestion: true });
  });

  it('never fabricates a target when Python sent none', () => {
    const view = requirementsPlanView({ ...preview, suggested_target: null });
    expect(view.suggestedTarget).toBeNull();
  });
});

describe('preview / confirm request builders — trim, no actor, target relayed verbatim', () => {
  it('preview body carries the trimmed text + agent name and no actor', () => {
    const body = previewRequirementsBody('  be fast  ', '  my_agent ');
    expect(body).toEqual({ requirements_text: 'be fast', agent_name: 'my_agent' });
    expect('actor' in body).toBe(false);
  });

  it('confirm body relays the human target EXACTLY (TS neither defaults nor clamps it)', () => {
    const body = confirmRequirementsBody('  be safe ', ' my_agent ', ' exp-1 ', -0.5);
    expect(body).toEqual({
      requirements_text: 'be safe',
      agent_name: 'my_agent',
      experiment_id: 'exp-1',
      objective_target: -0.5,
    });
    expect('actor' in body).toBe(false);
    // A different human value passes through unchanged — no TS-side normalization.
    expect(confirmRequirementsBody('x', 'a', 'e', 0.9999).objective_target).toBe(0.9999);
  });
});

describe('previewRequirementsMessage / confirmRequirementsMessage — honest verdicts', () => {
  it('a preview engine error is surfaced verbatim, never a fabricated plan', () => {
    const msg = previewRequirementsMessage({ outcome: 'error', error: 'LICENSE_TO_ILL: bad blob' });
    expect(msg?.tone).toBe('error');
    expect(msg?.text).toMatch(/LICENSE_TO_ILL/);
    expect(previewRequirementsMessage({ outcome: 'requirements_preview' })).toBeNull();
  });

  it('confirm success names the authored judges + persistence honestly', () => {
    const msg = confirmRequirementsMessage({
      outcome: 'requirements_confirmed',
      authored_judges: ['no_hallucinated_tool_calls', 'response_conciseness'],
      persisted: true,
    });
    expect(msg.tone).toBe('success');
    expect(msg.text).toMatch(/no_hallucinated_tool_calls/);
    expect(msg.text).toMatch(/persisted/);
  });

  it('a refusal (no target) is an error, not a fake success', () => {
    const msg = confirmRequirementsMessage({ outcome: 'refused', refused_reason: 'target required' });
    expect(msg.tone).toBe('error');
    expect(msg.text).toMatch(/target required/);
  });

  it('an engine error (wrong-sign target) is an error', () => {
    expect(confirmRequirementsMessage({ outcome: 'error', error: 'negative relative target' }).tone).toBe('error');
  });
});
