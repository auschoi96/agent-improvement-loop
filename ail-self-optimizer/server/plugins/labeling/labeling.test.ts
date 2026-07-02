import { describe, it, expect, vi } from 'vitest';
import {
  handleDimensions,
  handleLabel,
  readLabeler,
  type LabelingHttpRequest,
  type LabelingHttpResponse,
} from './labeling';
import type { LabelingAction, LabelingBridge, LabelingResult } from './bridge';

function fakeRes() {
  const captured: { code: number; body: unknown } = { code: 0, body: undefined };
  const res: LabelingHttpResponse = {
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

function req(headers: Record<string, string>, body: unknown): LabelingHttpRequest {
  return { headers, body };
}

const AUTH = { 'x-forwarded-email': 'labeler@databricks.com' };

function recordingBridge(result: LabelingResult = { outcome: 'labeled' }) {
  const calls: LabelingAction[] = [];
  const bridge: LabelingBridge = (input) => {
    calls.push(input);
    return Promise.resolve(result);
  };
  return { bridge, calls };
}

describe('readLabeler — authenticated identity from forwarded headers', () => {
  it('prefers the OBO email, falls back to the user id, else null (fail-closed)', () => {
    expect(readLabeler(req({ 'x-forwarded-email': 'a@b.com' }, {}))).toBe('a@b.com');
    expect(readLabeler(req({ 'x-forwarded-user': 'u123' }, {}))).toBe('u123');
    expect(readLabeler(req({}, {}))).toBeNull();
    expect(readLabeler(req({ 'x-forwarded-email': '  ' }, {}))).toBeNull();
  });
});

describe('both labeling routes are fail-closed authenticated', () => {
  const cases: Array<
    [string, (r: LabelingHttpRequest, res: LabelingHttpResponse, b: LabelingBridge) => Promise<void>, unknown]
  > = [
    ['dimensions', handleDimensions, { experiment_id: 'exp-1' }],
    ['label', handleLabel, { experiment_id: 'exp-1', trace_id: 't1', name: 'correctness', value: 'pass' }],
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

describe('handleDimensions — registered dimensions + progress come from the engine', () => {
  it('refuses a missing experiment_id (400) and never calls the engine', async () => {
    const { bridge, calls } = recordingBridge();
    const { res, captured } = fakeRes();
    await handleDimensions(req(AUTH, {}), res, bridge);
    expect(captured.code).toBe(400);
    expect(calls).toHaveLength(0);
  });

  it('passes the experiment + authenticated actor and returns the result verbatim', async () => {
    const { bridge, calls } = recordingBridge({
      outcome: 'dimensions',
      label_floor: 20,
      dimensions: [{ name: 'correctness', labels_so_far: 3, label_floor: 20, remaining: 17 }],
    });
    const { res, captured } = fakeRes();
    await handleDimensions(req(AUTH, { experiment_id: 'exp-9' }), res, bridge);
    expect(calls).toHaveLength(1);
    expect(calls[0]).toMatchObject({
      action: 'dimensions',
      actor: 'labeler@databricks.com',
      experiment_id: 'exp-9',
    });
    expect(captured.code).toBe(200);
    expect(captured.body).toMatchObject({ outcome: 'dimensions', label_floor: 20 });
  });

  it('surfaces an engine ERROR (cannot determine judges) verbatim — never invented dimensions', async () => {
    const { bridge } = recordingBridge({
      outcome: 'error',
      error: 'cannot determine the registered judges for this experiment; refusing to invent',
    });
    const { res, captured } = fakeRes();
    await handleDimensions(req(AUTH, { experiment_id: 'exp-1' }), res, bridge);
    expect(captured.code).toBe(200);
    expect((captured.body as { outcome: string }).outcome).toBe('error');
    expect((captured.body as { outcome: string }).outcome).not.toBe('dimensions');
  });

  it('a bridge (engine) failure is an honest 502 error, never a fabricated read', async () => {
    const bridge: LabelingBridge = vi.fn().mockRejectedValue(new Error('labeling-service exited 1'));
    const { res, captured } = fakeRes();
    await handleDimensions(req(AUTH, { experiment_id: 'exp-1' }), res, bridge);
    expect(captured.code).toBe(502);
    expect((captured.body as { outcome: string; error: string }).outcome).toBe('error');
    expect((captured.body as { error: string }).error).toMatch(/exited 1/);
  });
});

describe('handleLabel — authenticated write, body identity ignored, fail-closed', () => {
  it('labels with the AUTHENTICATED actor; a spoofed body labeler is ignored', async () => {
    const { bridge, calls } = recordingBridge({ outcome: 'labeled', name: 'correctness' });
    const { res, captured } = fakeRes();
    await handleLabel(
      req(AUTH, {
        experiment_id: 'exp-1',
        trace_id: 't1',
        name: 'correctness',
        value: 'pass',
        rationale: 'clear evidence',
        actor: 'attacker@evil.com',
        labeler: 'attacker@evil.com',
      }),
      res,
      bridge
    );
    expect(calls).toHaveLength(1);
    expect(calls[0].actor).toBe('labeler@databricks.com'); // authenticated identity, not the body value
    expect(calls[0]).toMatchObject({
      action: 'label',
      experiment_id: 'exp-1',
      trace_id: 't1',
      name: 'correctness',
      value: 'pass',
      rationale: 'clear evidence',
    });
    expect(captured.code).toBe(200);
    expect((captured.body as { outcome: string }).outcome).toBe('labeled');
  });

  it('forwards a falsey numeric value (0) without dropping it', async () => {
    const { bridge, calls } = recordingBridge();
    const { res } = fakeRes();
    await handleLabel(req(AUTH, { experiment_id: 'exp-1', trace_id: 't1', name: 'modularity', value: 0 }), res, bridge);
    expect(calls).toHaveLength(1);
    expect(calls[0].value).toBe(0);
  });

  it('refuses missing experiment_id / trace_id / name (400) and never calls the engine', async () => {
    const { bridge, calls } = recordingBridge();
    const { res, captured } = fakeRes();
    await handleLabel(req(AUTH, { experiment_id: 'exp-1', name: 'correctness', value: 'pass' }), res, bridge);
    expect(captured.code).toBe(400);
    expect(calls).toHaveLength(0);
  });

  it('refuses a missing value (400) and never calls the engine', async () => {
    const { bridge, calls } = recordingBridge();
    const { res, captured } = fakeRes();
    await handleLabel(req(AUTH, { experiment_id: 'exp-1', trace_id: 't1', name: 'correctness' }), res, bridge);
    expect(captured.code).toBe(400);
    expect(calls).toHaveLength(0);
  });

  it('surfaces an engine REFUSED (unknown judge name) verbatim — never a fake label', async () => {
    const { bridge } = recordingBridge({
      outcome: 'refused',
      refused_reason: "'not_a_judge' is not a registered judge",
    });
    const { res, captured } = fakeRes();
    await handleLabel(
      req(AUTH, { experiment_id: 'exp-1', trace_id: 't1', name: 'not_a_judge', value: 'pass' }),
      res,
      bridge
    );
    expect(captured.code).toBe(200);
    expect((captured.body as { outcome: string }).outcome).toBe('refused');
    expect((captured.body as { outcome: string }).outcome).not.toBe('labeled');
  });

  it('a bridge (write) failure is an honest 502 error, never a fake label', async () => {
    const bridge: LabelingBridge = vi.fn().mockRejectedValue(new Error('labeling-service exited 1'));
    const { res, captured } = fakeRes();
    await handleLabel(
      req(AUTH, { experiment_id: 'exp-1', trace_id: 't1', name: 'correctness', value: 'pass' }),
      res,
      bridge
    );
    expect(captured.code).toBe(502);
    expect((captured.body as { outcome: string }).outcome).toBe('error');
  });
});
