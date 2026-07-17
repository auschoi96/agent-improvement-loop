import { describe, it, expect } from 'vitest';
import { deferredJobVerifyBridge, resolveVerifyTransport, selectVerifyBridge, type VerifyInput } from './verify_bridge';

const INPUT: VerifyInput = {
  proposal_id: 'prop-1',
  agent_name: 'claude_code',
  requested_by: 'reviewer@databricks.com',
  requested_at: '2026-07-03T12:00:00.000Z',
};

describe('resolveVerifyTransport — env-driven transport selection', () => {
  it('defaults to the subprocess transport', () => {
    expect(resolveVerifyTransport({})).toBe('subprocess');
  });
  it('selects the job transport when AIL_VERIFY_TRANSPORT=job', () => {
    expect(resolveVerifyTransport({ AIL_VERIFY_TRANSPORT: 'job' })).toBe('job');
  });
  it('selects the job transport when AIL_VERIFY_JOB_ID is set', () => {
    expect(resolveVerifyTransport({ AIL_VERIFY_JOB_ID: '4242' })).toBe('job');
  });
  it('treats a blank AIL_VERIFY_JOB_ID as unset (subprocess)', () => {
    expect(resolveVerifyTransport({ AIL_VERIFY_JOB_ID: '   ' })).toBe('subprocess');
  });
});

describe('deferredJobVerifyBridge — fails closed, never a fake request', () => {
  it('rejects with an honest "not yet wired" error and requests nothing', async () => {
    const bridge = deferredJobVerifyBridge();
    await expect(bridge(INPUT)).rejects.toThrow(/not yet wired/);
    await expect(bridge(INPUT)).rejects.toThrow(/Failing closed/);
  });
});

describe('selectVerifyBridge — the deployed (job) image fails closed until wired', () => {
  it('the job transport returns the deferred, fail-closed bridge', async () => {
    const bridge = selectVerifyBridge({ AIL_VERIFY_TRANSPORT: 'job' });
    await expect(bridge(INPUT)).rejects.toThrow(/not yet wired/);
  });
});
