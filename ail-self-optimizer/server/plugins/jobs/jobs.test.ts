import { describe, it, expect } from 'vitest';
import { handleActivity, type ActivityHttpResponse } from './jobs';
import type { JobsActivityBridge, JobsActivityResult } from './bridge';

function fakeRes() {
  const captured: { code: number; body: unknown } = { code: 0, body: undefined };
  const res: ActivityHttpResponse = {
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

describe('handleActivity — read-only, returns the bridge result verbatim', () => {
  it('returns 200 with the bridge result (real per-job sections)', async () => {
    const result: JobsActivityResult = {
      jobs: [
        {
          name: 'ail-apply-service',
          status: 'ok',
          job_id: 7,
          runs: [{ run_id: 1, life_cycle_state: 'TERMINATED', result_state: 'SUCCESS' }],
        },
        { name: 'ail-l0-publish-scheduled', status: 'not_found' },
      ],
    };
    const bridge: JobsActivityBridge = () => Promise.resolve(result);
    const { res, captured } = fakeRes();

    await handleActivity(res, bridge);

    expect(captured.code).toBe(200);
    expect(captured.body).toEqual(result); // verbatim — no relabeling
  });

  it('surfaces an honest unavailable body when the bridge throws — never a fabricated run', async () => {
    const bridge: JobsActivityBridge = () => Promise.reject(new Error('SDK unavailable'));
    const { res, captured } = fakeRes();

    await handleActivity(res, bridge);

    expect(captured.code).toBe(200);
    expect(captured.body).toMatchObject({ jobs: [], fatal_error: 'SDK unavailable' });
  });
});
