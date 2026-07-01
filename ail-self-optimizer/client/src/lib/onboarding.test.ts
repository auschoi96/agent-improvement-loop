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
  freshnessMessage,
  creationMessage,
  registerMessage,
  resolvedFromValidation,
  resolvedFromCreation,
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
  it('experiment step needs a resolved FRESH experiment', () => {
    expect(stepValidation(stateAt('experiment')).ok).toBe(false);
    expect(stepValidation(stateAt('experiment', { resolved: { ...FRESH, fresh: false } })).ok).toBe(false);
    expect(stepValidation(stateAt('experiment', { resolved: FRESH })).ok).toBe(true);
  });

  it('goals step needs at least one goal', () => {
    expect(stepValidation(stateAt('goals')).ok).toBe(false);
    expect(canAdvance(stateAt('goals', { goals: ['cost'] }))).toBe(true);
  });

  it('data-gate step needs explicit acceptance', () => {
    expect(stepValidation(stateAt('data_gate')).ok).toBe(false);
    expect(stepValidation(stateAt('data_gate')).reason).toMatch(/accept/i);
    expect(canAdvance(stateAt('data_gate', { accepted: true }))).toBe(true);
  });

  it('register step needs a non-empty agent name', () => {
    expect(canAdvance(stateAt('register', { agentName: '   ' }))).toBe(false);
    expect(canAdvance(stateAt('register', { agentName: 'my_agent' }))).toBe(true);
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
