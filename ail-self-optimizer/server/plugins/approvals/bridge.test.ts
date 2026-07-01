import { describe, it, expect } from 'vitest';
import { jobTriggerApplyBridge, resolveApplyTransport, type JobTriggerClient, type DecisionInput } from './bridge';

const INPUT: DecisionInput = {
  proposal_id: 'prop-1',
  agent_name: 'claude_code',
  decision: 'approve',
  approver: 'reviewer@databricks.com',
  decided_at: '2026-06-30T12:00:00.000Z',
  // reason intentionally omitted -> the bridge must send '' as the job param
};

// A real ApplyServiceResult (what the job wrote to agent_apply_results) — the bridge
// must return THIS verbatim, never a fabricated one.
const REAL_APPLIED = {
  outcome: 'applied',
  proposal_id: 'prop-1',
  agent_name: 'claude_code',
  decision: 'approve',
  approver: 'reviewer@databricks.com',
  decided_at: '2026-06-30T12:00:00.000Z',
  created_view: 'cat.sch.mv',
  status: 'applied',
};

type RunLike = Awaited<ReturnType<JobTriggerClient['getRun']>>;
type StatementLike = Awaited<ReturnType<JobTriggerClient['executeStatement']>>;

interface FakeOpts {
  runId?: number | undefined;
  runs: RunLike[];
  statement?: StatementLike;
}

function fakeClient(opts: FakeOpts) {
  const calls = {
    runNow: [] as Array<Parameters<JobTriggerClient['runNow']>[0]>,
    getRun: 0,
    executeStatement: [] as Array<Parameters<JobTriggerClient['executeStatement']>[0]>,
  };
  let runIdx = 0;
  const client: JobTriggerClient = {
    runNow(req) {
      calls.runNow.push(req);
      return Promise.resolve({ run_id: 'runId' in opts ? opts.runId : 99 });
    },
    getRun() {
      calls.getRun += 1;
      const i = Math.min(runIdx, opts.runs.length - 1);
      runIdx += 1;
      return Promise.resolve(opts.runs[i]);
    },
    executeStatement(req) {
      calls.executeStatement.push(req);
      return Promise.resolve(opts.statement ?? { status: { state: 'SUCCEEDED' }, result: { data_array: [] } });
    },
    getStatement() {
      return Promise.resolve({ status: { state: 'SUCCEEDED' }, result: { data_array: [] } });
    },
  };
  return { client, calls };
}

const RUNNING: RunLike = { state: { life_cycle_state: 'RUNNING' } };
const SUCCESS: RunLike = { state: { life_cycle_state: 'TERMINATED', result_state: 'SUCCESS' } };

function bridgeWith(fake: ReturnType<typeof fakeClient>, over: Record<string, unknown> = {}) {
  return jobTriggerApplyBridge({
    jobId: 4242,
    warehouseId: 'wh-1',
    catalog: 'cat',
    schema: 'sch',
    timeoutMs: 200,
    pollIntervalMs: 1,
    statementPollMs: 1,
    clientFactory: () => fake.client,
    ...over,
  });
}

describe('jobTriggerApplyBridge — success returns the engine result verbatim', () => {
  it('triggers, polls to SUCCESS, reads the result row, and returns it', async () => {
    const fake = fakeClient({
      runId: 7,
      runs: [RUNNING, SUCCESS], // exercises the poll loop
      statement: { status: { state: 'SUCCEEDED' }, result: { data_array: [[JSON.stringify(REAL_APPLIED)]] } },
    });
    const bridge = bridgeWith(fake);

    const result = await bridge(INPUT);

    expect(result).toEqual(REAL_APPLIED); // verbatim, not fabricated
    // the decision was passed as job parameters, with an idempotency token
    expect(fake.calls.runNow).toHaveLength(1);
    expect(fake.calls.runNow[0].job_id).toBe(4242);
    expect(fake.calls.runNow[0].job_parameters).toMatchObject({
      proposal_id: 'prop-1',
      agent_name: 'claude_code',
      decision: 'approve',
      approver: 'reviewer@databricks.com',
      reason: '', // omitted reason -> empty string, never undefined
      decided_at: '2026-06-30T12:00:00.000Z',
    });
    expect(fake.calls.runNow[0].idempotency_token).toBeTruthy();
    // the read is scoped to this decision's (proposal_id, decided_at) key
    expect(fake.calls.executeStatement).toHaveLength(1);
    expect(fake.calls.executeStatement[0].statement).toContain("proposal_id = 'prop-1'");
    expect(fake.calls.executeStatement[0].statement).toContain("decided_at = '2026-06-30T12:00:00.000Z'");
  });
});

describe('jobTriggerApplyBridge — fail-closed (never a fabricated apply)', () => {
  it('a FAILED run rejects and never reads a result', async () => {
    const fake = fakeClient({
      runs: [{ state: { life_cycle_state: 'TERMINATED', result_state: 'FAILED', state_message: 'boom' } }],
    });
    const bridge = bridgeWith(fake);
    await expect(bridge(INPUT)).rejects.toThrow(/TERMINATED\/FAILED/);
    expect(fake.calls.executeStatement).toHaveLength(0);
  });

  it('a non-terminal run at the timeout rejects and never reads a result', async () => {
    const fake = fakeClient({ runs: [RUNNING] });
    const bridge = bridgeWith(fake, { timeoutMs: 15 });
    await expect(bridge(INPUT)).rejects.toThrow(/still RUNNING after 15ms/);
    expect(fake.calls.executeStatement).toHaveLength(0);
  });

  it('a SUCCESS run with no result row rejects (result unretrievable)', async () => {
    const fake = fakeClient({
      runs: [SUCCESS],
      statement: { status: { state: 'SUCCEEDED' }, result: { data_array: [] } },
    });
    const bridge = bridgeWith(fake);
    await expect(bridge(INPUT)).rejects.toThrow(/wrote no result row/);
    expect(fake.calls.executeStatement).toHaveLength(1);
  });

  it('a failed result-read statement rejects (result unretrievable)', async () => {
    const fake = fakeClient({
      runs: [SUCCESS],
      statement: { status: { state: 'FAILED', error: { message: 'perm denied' } } },
    });
    const bridge = bridgeWith(fake);
    await expect(bridge(INPUT)).rejects.toThrow(/perm denied|FAILED/);
  });

  it('an unparseable result row rejects (never a fabricated outcome)', async () => {
    const fake = fakeClient({
      runs: [SUCCESS],
      statement: { status: { state: 'SUCCEEDED' }, result: { data_array: [['not json {']] } },
    });
    const bridge = bridgeWith(fake);
    await expect(bridge(INPUT)).rejects.toThrow(/unparseable/);
  });

  it('a trigger returning no run id rejects', async () => {
    const fake = fakeClient({ runId: undefined, runs: [SUCCESS] });
    const bridge = bridgeWith(fake);
    await expect(bridge(INPUT)).rejects.toThrow(/no run id/);
  });

  it('rejects when the apply job id is not configured', async () => {
    const prev = process.env.AIL_APPLY_JOB_ID;
    delete process.env.AIL_APPLY_JOB_ID;
    try {
      const fake = fakeClient({ runs: [SUCCESS] });
      const bridge = bridgeWith(fake, { jobId: undefined });
      await expect(bridge(INPUT)).rejects.toThrow(/AIL_APPLY_JOB_ID is not set/);
    } finally {
      if (prev !== undefined) process.env.AIL_APPLY_JOB_ID = prev;
    }
  });

  it('rejects blank/invalid apply job ids without triggering and accepts a valid env id', async () => {
    const prev = process.env.AIL_APPLY_JOB_ID;
    try {
      process.env.AIL_APPLY_JOB_ID = '   ';
      const blank = fakeClient({ runs: [SUCCESS] });
      const blankBridge = jobTriggerApplyBridge({
        warehouseId: 'wh-1',
        clientFactory: () => blank.client,
      });
      await expect(blankBridge(INPUT)).rejects.toThrow(/AIL_APPLY_JOB_ID is not set/);
      expect(blank.calls.runNow).toHaveLength(0);

      process.env.AIL_APPLY_JOB_ID = 'not-a-job';
      const invalid = fakeClient({ runs: [SUCCESS] });
      const invalidBridge = jobTriggerApplyBridge({
        warehouseId: 'wh-1',
        clientFactory: () => invalid.client,
      });
      await expect(invalidBridge(INPUT)).rejects.toThrow(/AIL_APPLY_JOB_ID is invalid/);
      expect(invalid.calls.runNow).toHaveLength(0);

      process.env.AIL_APPLY_JOB_ID = '4242';
      const valid = fakeClient({
        runs: [SUCCESS],
        statement: { status: { state: 'SUCCEEDED' }, result: { data_array: [[JSON.stringify(REAL_APPLIED)]] } },
      });
      const validBridge = jobTriggerApplyBridge({
        warehouseId: 'wh-1',
        catalog: 'cat',
        schema: 'sch',
        clientFactory: () => valid.client,
      });
      await expect(validBridge(INPUT)).resolves.toEqual(REAL_APPLIED);
      expect(valid.calls.runNow[0].job_id).toBe(4242);
    } finally {
      if (prev === undefined) delete process.env.AIL_APPLY_JOB_ID;
      else process.env.AIL_APPLY_JOB_ID = prev;
    }
  });

  it('rejects when the warehouse for the result read is not configured', async () => {
    const prev = process.env.DATABRICKS_WAREHOUSE_ID;
    delete process.env.DATABRICKS_WAREHOUSE_ID;
    try {
      const fake = fakeClient({ runs: [SUCCESS] });
      const bridge = bridgeWith(fake, { warehouseId: undefined });
      await expect(bridge(INPUT)).rejects.toThrow(/DATABRICKS_WAREHOUSE_ID is not set/);
    } finally {
      if (prev !== undefined) process.env.DATABRICKS_WAREHOUSE_ID = prev;
    }
  });
});

describe('resolveApplyTransport — env-driven transport selection', () => {
  it('defaults to the subprocess transport', () => {
    expect(resolveApplyTransport({})).toBe('subprocess');
  });
  it('selects the job transport when AIL_APPLY_TRANSPORT=job', () => {
    expect(resolveApplyTransport({ AIL_APPLY_TRANSPORT: 'job' })).toBe('job');
  });
  it('selects the job transport when AIL_APPLY_JOB_ID is set', () => {
    expect(resolveApplyTransport({ AIL_APPLY_JOB_ID: '4242' })).toBe('job');
  });
  it('treats a blank AIL_APPLY_JOB_ID as unset (subprocess)', () => {
    expect(resolveApplyTransport({ AIL_APPLY_JOB_ID: '   ' })).toBe('subprocess');
  });
});
