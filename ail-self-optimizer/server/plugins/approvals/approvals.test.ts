import { describe, it, expect, vi } from 'vitest';
import { handleDecision, readApprover, type DecisionHttpRequest, type DecisionHttpResponse } from './approvals';
import type { ApplyBridge, BridgeResult, DecisionInput } from './bridge';

function fakeRes() {
  const captured: { code: number; body: unknown } = { code: 0, body: undefined };
  const res: DecisionHttpResponse = {
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

function req(headers: Record<string, string>, body: unknown): DecisionHttpRequest {
  return { headers, body };
}

const AUTH = { 'x-forwarded-email': 'reviewer@databricks.com' };

function recordingBridge(result: BridgeResult = { outcome: 'applied', created_view: 'cat.sch.v' }) {
  const calls: DecisionInput[] = [];
  const bridge: ApplyBridge = (input) => {
    calls.push(input);
    return Promise.resolve(result);
  };
  return { bridge, calls };
}

describe('readApprover — authenticated identity from forwarded headers', () => {
  it('prefers the OBO email, falls back to the user id, else null (fail-closed)', () => {
    expect(readApprover(req({ 'x-forwarded-email': 'a@b.com' }, {}))).toBe('a@b.com');
    expect(readApprover(req({ 'x-forwarded-user': 'u123' }, {}))).toBe('u123');
    expect(readApprover(req({}, {}))).toBeNull();
    expect(readApprover(req({ 'x-forwarded-email': '  ' }, {}))).toBeNull();
  });
});

describe('handleDecision — fail-closed authentication', () => {
  it('refuses an unauthenticated request (401) and never calls the engine', async () => {
    const { bridge, calls } = recordingBridge();
    const { res, captured } = fakeRes();
    await handleDecision(req({}, { proposal_id: 'p1', agent_name: 'a', decision: 'approve' }), res, bridge);
    expect(captured.code).toBe(401);
    expect((captured.body as { outcome: string }).outcome).toBe('refused');
    expect(calls).toHaveLength(0);
  });
});

describe('handleDecision — approve', () => {
  it('calls the engine with the AUTHENTICATED approver + a server-set timestamp', async () => {
    const { bridge, calls } = recordingBridge();
    const { res, captured } = fakeRes();
    // A spoofed approver in the body must be ignored — the header identity wins.
    await handleDecision(
      req(AUTH, { proposal_id: 'p1', agent_name: 'claude_code', decision: 'approve', approver: 'attacker@evil.com' }),
      res,
      bridge
    );
    expect(calls).toHaveLength(1);
    expect(calls[0].approver).toBe('reviewer@databricks.com');
    expect(calls[0].decision).toBe('approve');
    expect(calls[0].decided_at).toMatch(/^\d{4}-\d{2}-\d{2}T/);
    expect(captured.code).toBe(200);
    expect((captured.body as { outcome: string }).outcome).toBe('applied');
  });
});

describe('handleDecision — reject requires a reason', () => {
  it('refuses a reject with no reason (400) and never calls the engine', async () => {
    const { bridge, calls } = recordingBridge();
    const { res, captured } = fakeRes();
    await handleDecision(req(AUTH, { proposal_id: 'p1', agent_name: 'a', decision: 'reject' }), res, bridge);
    expect(captured.code).toBe(400);
    expect(calls).toHaveLength(0);
  });

  it('passes the reason through on a valid reject', async () => {
    const { bridge, calls } = recordingBridge({ outcome: 'rejected' });
    const { res, captured } = fakeRes();
    await handleDecision(
      req(AUTH, { proposal_id: 'p1', agent_name: 'a', decision: 'reject', reason: 'rule mis-fired' }),
      res,
      bridge
    );
    expect(calls[0].decision).toBe('reject');
    expect(calls[0].reason).toBe('rule mis-fired');
    expect((captured.body as { outcome: string }).outcome).toBe('rejected');
  });
});

describe('handleDecision — validation + engine surfacing', () => {
  it('refuses a missing proposal_id / agent_name (400)', async () => {
    const { bridge, calls } = recordingBridge();
    const { res, captured } = fakeRes();
    await handleDecision(req(AUTH, { decision: 'approve' }), res, bridge);
    expect(captured.code).toBe(400);
    expect(calls).toHaveLength(0);
  });

  it('refuses an unknown decision (400)', async () => {
    const { bridge, calls } = recordingBridge();
    const { res, captured } = fakeRes();
    await handleDecision(req(AUTH, { proposal_id: 'p1', agent_name: 'a', decision: 'delete' }), res, bridge);
    expect(captured.code).toBe(400);
    expect(calls).toHaveLength(0);
  });

  it('surfaces an engine ApplyRefused verbatim (200 + refused outcome)', async () => {
    const { bridge } = recordingBridge({ outcome: 'refused', refused_reason: 'apply-time gate re-check failed' });
    const { res, captured } = fakeRes();
    await handleDecision(req(AUTH, { proposal_id: 'p1', agent_name: 'a', decision: 'approve' }), res, bridge);
    expect(captured.code).toBe(200);
    expect(captured.body).toMatchObject({ outcome: 'refused', refused_reason: 'apply-time gate re-check failed' });
  });

  it('returns 502 when the bridge itself fails — an honest error, never a fake apply', async () => {
    const bridge: ApplyBridge = vi.fn().mockRejectedValue(new Error('apply-service exited 1'));
    const { res, captured } = fakeRes();
    await handleDecision(req(AUTH, { proposal_id: 'p1', agent_name: 'a', decision: 'approve' }), res, bridge);
    expect(captured.code).toBe(502);
    expect((captured.body as { outcome: string; error: string }).outcome).toBe('error');
    expect((captured.body as { error: string }).error).toMatch(/exited 1/);
  });
});
