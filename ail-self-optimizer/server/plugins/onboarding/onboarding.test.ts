import { describe, it, expect, vi } from 'vitest';
import {
  handleRequirements,
  handleValidateExperiment,
  handleCreateExperiment,
  handleRegisterAgent,
  handlePreviewRequirements,
  handleConfirmRequirements,
  readActor,
  type OnboardingHttpRequest,
  type OnboardingHttpResponse,
} from './onboarding';
import type { OnboardingAction, OnboardingBridge, OnboardingResult } from './bridge';

function fakeRes() {
  const captured: { code: number; body: unknown } = { code: 0, body: undefined };
  const res: OnboardingHttpResponse = {
    status(code: number) {
      captured.code = code;
      return res;
    },
    json(body: unknown) {
      captured.body = body;
    },
  };
  return { res, captured };
}

function req(headers: Record<string, string>, body: unknown): OnboardingHttpRequest {
  return { headers, body };
}

const AUTH = { 'x-forwarded-email': 'onboarder@databricks.com' };

function recordingBridge(result: OnboardingResult = { outcome: 'registered' }) {
  const calls: OnboardingAction[] = [];
  const bridge: OnboardingBridge = (input) => {
    calls.push(input);
    return Promise.resolve(result);
  };
  return { bridge, calls };
}

describe('readActor — authenticated identity from forwarded headers', () => {
  it('prefers the OBO email, falls back to the user id, else null (fail-closed)', () => {
    expect(readActor(req({ 'x-forwarded-email': 'a@b.com' }, {}))).toBe('a@b.com');
    expect(readActor(req({ 'x-forwarded-user': 'u123' }, {}))).toBe('u123');
    expect(readActor(req({}, {}))).toBeNull();
    expect(readActor(req({ 'x-forwarded-email': '  ' }, {}))).toBeNull();
  });
});

describe('all onboarding routes are fail-closed authenticated', () => {
  const cases: Array<
    [string, (r: OnboardingHttpRequest, res: OnboardingHttpResponse, b: OnboardingBridge) => Promise<void>, unknown]
  > = [
    ['requirements', handleRequirements, { goals: ['cost'] }],
    ['validate', handleValidateExperiment, { experiment_id: 'exp-1' }],
    ['create', handleCreateExperiment, { name: 'Fresh' }],
    ['register', handleRegisterAgent, { agent_name: 'a', experiment_id: 'exp-1', goals: ['cost'] }],
    ['preview_requirements', handlePreviewRequirements, { requirements_text: 'be fast', agent_name: 'a' }],
    [
      'confirm_requirements',
      handleConfirmRequirements,
      { requirements_text: 'be fast', agent_name: 'a', experiment_id: 'exp-1', objective_target: -0.3 },
    ],
  ];
  for (const [label, handler, body] of cases) {
    it(`${label} refuses an unauthenticated request (401) and never calls the engine`, async () => {
      const { bridge, calls } = recordingBridge();
      const { res, captured } = fakeRes();
      await handler(req({}, body), res, bridge);
      expect(captured.code).toBe(401);
      expect((captured.body as { outcome: string }).outcome).toBe('refused');
      expect(calls).toHaveLength(0);
    });
  }
});

describe('handleRequirements — the goal catalog + gate facts come from the engine', () => {
  it('serves the generated complete catalog without starting a serverless job', async () => {
    const { bridge, calls } = recordingBridge();
    const { res, captured } = fakeRes();
    await handleRequirements(req(AUTH, { goals: [] }), res, bridge);
    expect(calls).toHaveLength(0);
    expect(captured.code).toBe(200);
    expect(captured.body).toMatchObject({ outcome: 'requirements', requires_labels: false });
    expect((captured.body as { catalog: unknown[] }).catalog).toHaveLength(4);
  });

  it('passes the selected goals + authenticated actor to the engine', async () => {
    const { bridge, calls } = recordingBridge({ outcome: 'requirements', requires_labels: true });
    const { res, captured } = fakeRes();
    await handleRequirements(req(AUTH, { goals: ['accuracy', 'cost'] }), res, bridge);
    expect(calls).toHaveLength(1);
    expect(calls[0]).toMatchObject({
      action: 'requirements',
      actor: 'onboarder@databricks.com',
      goals: ['accuracy', 'cost'],
    });
    expect(captured.code).toBe(200);
    expect((captured.body as { outcome: string }).outcome).toBe('requirements');
  });
});

describe('handleValidateExperiment — fail-closed + honest freshness verdict', () => {
  it('refuses a missing experiment_id (400) and never calls the engine', async () => {
    const { bridge, calls } = recordingBridge();
    const { res, captured } = fakeRes();
    await handleValidateExperiment(req(AUTH, {}), res, bridge);
    expect(captured.code).toBe(400);
    expect(calls).toHaveLength(0);
  });

  it('surfaces a NON-FRESH experiment verdict verbatim (never dressed up as fresh)', async () => {
    const { bridge, calls } = recordingBridge({
      outcome: 'validated',
      fresh: false,
      trace_count: 12,
      reasons: ['experiment already has 12 trace(s)'],
    });
    const { res, captured } = fakeRes();
    await handleValidateExperiment(req(AUTH, { experiment_id: 'exp-9' }), res, bridge);
    expect(calls[0]).toMatchObject({ action: 'validate_experiment', experiment_id: 'exp-9' });
    expect(captured.code).toBe(200);
    expect(captured.body).toMatchObject({ outcome: 'validated', fresh: false });
  });
});

describe('handleCreateExperiment — fail-closed create', () => {
  it('refuses a missing name (400)', async () => {
    const { bridge, calls } = recordingBridge();
    const { res, captured } = fakeRes();
    await handleCreateExperiment(req(AUTH, {}), res, bridge);
    expect(captured.code).toBe(400);
    expect(calls).toHaveLength(0);
  });

  it('surfaces an engine permission error verbatim — never a fabricated creation', async () => {
    const { bridge } = recordingBridge({
      outcome: 'error',
      error: 'the app service principal is not authorized to create MLflow experiment',
      prerequisite: 'app service principal needs experiment-create authority in the workspace',
    });
    const { res, captured } = fakeRes();
    await handleCreateExperiment(req(AUTH, { name: 'Fresh agent' }), res, bridge);
    expect(captured.code).toBe(200);
    expect(captured.body).toMatchObject({ outcome: 'error' });
    expect((captured.body as { outcome: string }).outcome).not.toBe('created');
  });

  it('forwards explicit idempotent reuse only for internal reviewer provisioning', async () => {
    const { bridge, calls } = recordingBridge({ outcome: 'created', experiment_id: 'review-exp' });
    const { res, captured } = fakeRes();
    await handleCreateExperiment(req(AUTH, { name: 'agent-ail-internal', allow_existing: true }), res, bridge);
    expect(calls[0]).toMatchObject({
      action: 'create_experiment',
      name: 'agent-ail-internal',
      allow_existing: true,
    });
    expect(captured.code).toBe(200);
  });

  it('does not allow an ordinary subject experiment create to reuse an existing name', async () => {
    const { bridge, calls } = recordingBridge({ outcome: 'created', experiment_id: 'subject-exp' });
    const { res } = fakeRes();
    await handleCreateExperiment(req(AUTH, { name: 'ordinary-subject', allow_existing: true }), res, bridge);
    expect(calls[0].allow_existing).toBeUndefined();
  });

  it('a bridge (engine) failure is an honest 502 error, never a fake creation', async () => {
    const bridge: OnboardingBridge = vi.fn().mockRejectedValue(new Error('onboarding-service exited 1'));
    const { res, captured } = fakeRes();
    await handleCreateExperiment(req(AUTH, { name: 'Fresh agent' }), res, bridge);
    expect(captured.code).toBe(502);
    expect((captured.body as { outcome: string; error: string }).outcome).toBe('error');
    expect((captured.body as { error: string }).error).toMatch(/exited 1/);
  });
});

describe('handleRegisterAgent — authenticated write, body identity ignored', () => {
  it('registers with the AUTHENTICATED actor; a spoofed body actor is ignored', async () => {
    const { bridge, calls } = recordingBridge({ outcome: 'registered', agent_name: 'my_agent' });
    const { res, captured } = fakeRes();
    await handleRegisterAgent(
      req(AUTH, {
        agent_name: 'my_agent',
        experiment_id: 'exp-1',
        goals: ['accuracy', 'cost'],
        actor: 'attacker@evil.com',
      }),
      res,
      bridge
    );
    expect(calls).toHaveLength(1);
    expect(calls[0].actor).toBe('onboarder@databricks.com');
    expect(calls[0]).toMatchObject({ action: 'register_agent', agent_name: 'my_agent', experiment_id: 'exp-1' });
    expect(captured.code).toBe(200);
    expect((captured.body as { outcome: string }).outcome).toBe('registered');
  });

  it('forwards the executor workspace, memory table, and goal_config; actor stays server-set', async () => {
    const { bridge, calls } = recordingBridge({ outcome: 'registered', agent_name: 'my_agent' });
    const { res } = fakeRes();
    const goal_config = {
      objective_metric: 'no_hallucinated_tool_calls',
      goal_direction: 'maximize',
      goal_target: 0.25,
    };
    await handleRegisterAgent(
      req(AUTH, {
        agent_name: 'my_agent',
        experiment_id: 'exp-1',
        goals: ['cost'],
        target_workspace: '  /Workspace/Repos/me/my_agent ',
        annotations_table: ' catalog.schema.otel_annotations ',
        goal_config,
        actor: 'attacker@evil.com',
      }),
      res,
      bridge
    );
    expect(calls).toHaveLength(1);
    expect(calls[0].actor).toBe('onboarder@databricks.com');
    expect(calls[0].target_workspace).toBe('/Workspace/Repos/me/my_agent');
    expect(calls[0].annotations_table).toBe('catalog.schema.otel_annotations');
    expect(calls[0].goal_config).toEqual(goal_config);
  });

  it('omits blank/malformed extended fields so the engine sees them absent', async () => {
    const { bridge, calls } = recordingBridge({ outcome: 'registered', agent_name: 'my_agent' });
    const { res } = fakeRes();
    await handleRegisterAgent(
      req(AUTH, {
        agent_name: 'my_agent',
        experiment_id: 'exp-1',
        goals: ['cost'],
        target_workspace: '   ',
        annotations_table: 42,
        goal_config: [1, 2, 3],
      }),
      res,
      bridge
    );
    expect(calls).toHaveLength(1);
    expect(calls[0].target_workspace).toBeUndefined();
    expect(calls[0].annotations_table).toBeUndefined();
    expect(calls[0].goal_config).toBeUndefined();
  });

  it('refuses missing agent_name / experiment_id (400) and never calls the engine', async () => {
    const { bridge, calls } = recordingBridge();
    const { res, captured } = fakeRes();
    await handleRegisterAgent(req(AUTH, { goals: ['cost'] }), res, bridge);
    expect(captured.code).toBe(400);
    expect(calls).toHaveLength(0);
  });

  it('refuses an empty goal selection (400)', async () => {
    const { bridge, calls } = recordingBridge();
    const { res, captured } = fakeRes();
    await handleRegisterAgent(req(AUTH, { agent_name: 'a', experiment_id: 'exp-1', goals: [] }), res, bridge);
    expect(captured.code).toBe(400);
    expect(calls).toHaveLength(0);
  });

  it('surfaces an engine REFUSED (e.g. duplicate name) verbatim — never a fake register', async () => {
    const { bridge } = recordingBridge({
      outcome: 'refused',
      refused_reason: "an agent named 'my_agent' is already registered",
    });
    const { res, captured } = fakeRes();
    await handleRegisterAgent(
      req(AUTH, { agent_name: 'my_agent', experiment_id: 'exp-1', goals: ['cost'] }),
      res,
      bridge
    );
    expect(captured.code).toBe(200);
    expect((captured.body as { outcome: string }).outcome).toBe('refused');
  });

  it('a bridge (registry write) failure is an honest 502 error, never a fake register', async () => {
    const bridge: OnboardingBridge = vi.fn().mockRejectedValue(new Error('onboarding-service exited 1'));
    const { res, captured } = fakeRes();
    await handleRegisterAgent(req(AUTH, { agent_name: 'a', experiment_id: 'exp-1', goals: ['cost'] }), res, bridge);
    expect(captured.code).toBe(502);
    expect((captured.body as { outcome: string }).outcome).toBe('error');
  });
});

describe('handlePreviewRequirements — free-form intake preview, actor server-set', () => {
  it('forwards the raw requirements text + AUTHENTICATED actor; a body actor is ignored', async () => {
    const { bridge, calls } = recordingBridge({ outcome: 'requirements_preview', dimensions: [] });
    const { res, captured } = fakeRes();
    await handlePreviewRequirements(
      req(AUTH, {
        requirements_text: 'correctness matters most; keep latency low',
        agent_name: 'my_agent',
        actor: 'attacker@evil.com',
      }),
      res,
      bridge
    );
    expect(calls).toHaveLength(1);
    expect(calls[0].actor).toBe('onboarder@databricks.com'); // NOT the body actor
    expect(calls[0]).toMatchObject({
      action: 'preview_requirements',
      requirements_text: 'correctness matters most; keep latency low',
      agent_name: 'my_agent',
    });
    expect(captured.code).toBe(200);
    expect((captured.body as { outcome: string }).outcome).toBe('requirements_preview');
  });

  it('refuses a missing requirements_text (400) and never calls the engine', async () => {
    const { bridge, calls } = recordingBridge();
    const { res, captured } = fakeRes();
    await handlePreviewRequirements(req(AUTH, { agent_name: 'a' }), res, bridge);
    expect(captured.code).toBe(400);
    expect(calls).toHaveLength(0);
  });
});

describe('handleConfirmRequirements — authored write, actor server-set, honest target relay', () => {
  it('forwards the text + human target + AUTHENTICATED actor; a body actor is ignored', async () => {
    const { bridge, calls } = recordingBridge({ outcome: 'requirements_confirmed', persisted: true });
    const { res, captured } = fakeRes();
    await handleConfirmRequirements(
      req(AUTH, {
        requirements_text: 'never hallucinate a tool call',
        agent_name: 'my_agent',
        experiment_id: 'exp-1',
        objective_target: 0.25,
        actor: 'attacker@evil.com',
      }),
      res,
      bridge
    );
    expect(calls).toHaveLength(1);
    expect(calls[0].actor).toBe('onboarder@databricks.com'); // NOT the body actor
    expect(calls[0]).toMatchObject({
      action: 'confirm_requirements',
      requirements_text: 'never hallucinate a tool call',
      agent_name: 'my_agent',
      experiment_id: 'exp-1',
      objective_target: 0.25,
    });
    expect(captured.code).toBe(200);
    expect((captured.body as { outcome: string }).outcome).toBe('requirements_confirmed');
  });

  it('omits a non-numeric objective_target so the engine refuses honestly (never a fake target)', async () => {
    const { bridge, calls } = recordingBridge({ outcome: 'refused' });
    const { res } = fakeRes();
    await handleConfirmRequirements(
      req(AUTH, {
        requirements_text: 'be fast',
        agent_name: 'my_agent',
        experiment_id: 'exp-1',
        objective_target: 'not-a-number',
      }),
      res,
      bridge
    );
    expect(calls).toHaveLength(1);
    expect(calls[0].objective_target).toBeUndefined();
  });

  it('refuses missing requirements_text / agent_name / experiment_id (400)', async () => {
    const { bridge, calls } = recordingBridge();
    const { res, captured } = fakeRes();
    await handleConfirmRequirements(req(AUTH, { requirements_text: 'be fast', agent_name: 'a' }), res, bridge);
    expect(captured.code).toBe(400);
    expect(calls).toHaveLength(0);
  });
});
