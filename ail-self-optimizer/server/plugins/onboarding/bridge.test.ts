import { describe, it, expect, beforeAll, afterAll, afterEach, vi } from 'vitest';
import { mkdtempSync, rmSync, writeFileSync, chmodSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import {
  jobTriggerOnboardingBridge,
  readJobOnboardingStatus,
  spawnPythonOnboardingBridge,
  selectOnboardingBridge,
  type OnboardingAction,
  type OnboardingJobClient,
} from './bridge';

// Hermetic subprocess test: the bridge hardcodes `-m ail.onboarding.service`, so we
// point `pythonBin` at tiny stub interpreters that ignore those args and stand in
// for the Python engine — proving the bridge's happy path (stdin action -> parsed
// stdout result) and its fail-closed handling (non-zero exit / unparseable output)
// without needing the `ail` package or a live workspace.

let dir: string;
const stub = (name: string, script: string): string => {
  const p = path.join(dir, name);
  writeFileSync(p, `#!/bin/sh\n${script}\n`);
  chmodSync(p, 0o755);
  return p;
};

let echoStub: string;
let failStub: string;
let garbageStub: string;

const INPUT: OnboardingAction = { action: 'requirements', actor: 'a@b.com', goals: ['cost'] };

beforeAll(() => {
  dir = mkdtempSync(path.join(tmpdir(), 'ail-onboarding-bridge-'));
  echoStub = stub('echo.sh', 'cat'); // echo stdin JSON back as the "result"
  failStub = stub('fail.sh', 'echo "boom" 1>&2\nexit 1');
  garbageStub = stub('garbage.sh', "printf 'not json {'");
});

afterAll(() => {
  rmSync(dir, { recursive: true, force: true });
});

afterEach(() => {
  vi.unstubAllEnvs();
});

describe('spawnPythonOnboardingBridge — subprocess transport', () => {
  it('writes the action on stdin and parses the JSON result from stdout', async () => {
    const bridge = spawnPythonOnboardingBridge({ pythonBin: echoStub, srcPath: dir });
    const result = await bridge(INPUT);
    // the echo stub returns the action verbatim -> the bridge parsed it as the result
    expect(result).toMatchObject({ action: 'requirements', actor: 'a@b.com' });
  });

  it('rejects a non-zero exit (never a fabricated result)', async () => {
    const bridge = spawnPythonOnboardingBridge({ pythonBin: failStub, srcPath: dir });
    await expect(bridge(INPUT)).rejects.toThrow(/exited 1/);
  });

  it('rejects unparseable stdout (never a fabricated result)', async () => {
    const bridge = spawnPythonOnboardingBridge({ pythonBin: garbageStub, srcPath: dir });
    await expect(bridge(INPUT)).rejects.toThrow(/unparseable/);
  });

  it('rejects when the interpreter cannot be spawned', async () => {
    const bridge = spawnPythonOnboardingBridge({
      pythonBin: path.join(dir, 'does-not-exist'),
      srcPath: dir,
    });
    await expect(bridge(INPUT)).rejects.toBeInstanceOf(Error);
  });
});

describe('selectOnboardingBridge', () => {
  it('returns the subprocess transport (a callable bridge)', () => {
    expect(typeof selectOnboardingBridge()).toBe('function');
  });
});

const jobClient = (overrides: Partial<OnboardingJobClient> = {}): OnboardingJobClient => ({
  runNow: vi.fn().mockResolvedValue({ run_id: 41 }),
  getRun: vi.fn().mockResolvedValue({ state: { life_cycle_state: 'RUNNING' } }),
  getRunOutput: vi.fn().mockResolvedValue({}),
  executeStatement: vi.fn().mockImplementation(({ statement }: { statement: string }) =>
    Promise.resolve({
      statement_id: 'stmt-1',
      status: { state: 'SUCCEEDED' },
      result: { data_array: statement.startsWith('SELECT run_id') ? [['41']] : [] },
    })
  ),
  getStatement: vi.fn().mockRejectedValue(new Error('not expected')),
  ...overrides,
});

const configureJobTransport = (): void => {
  vi.stubEnv('AIL_ONBOARDING_JOB_ID', '123');
  vi.stubEnv('DATABRICKS_WAREHOUSE_ID', 'wh-1');
  vi.stubEnv('AIL_CATALOG', 'cat');
  vi.stubEnv('AIL_SCHEMA', 'sch');
};

describe('job onboarding transport', () => {
  it('stores the governed payload and submits only an opaque request id', async () => {
    configureJobTransport();
    let submitted: Parameters<OnboardingJobClient['runNow']>[0] | undefined;
    const runNow: OnboardingJobClient['runNow'] = (request) => {
      submitted = request;
      return Promise.resolve({ run_id: 41 });
    };
    const statements: string[] = [];
    const executeStatement: OnboardingJobClient['executeStatement'] = (request) => {
      statements.push(request.statement);
      return Promise.resolve({
        statement_id: 'stmt-1',
        status: { state: 'SUCCEEDED' },
        result: { data_array: [] },
      });
    };
    const client = jobClient({ runNow, executeStatement });
    const result = await jobTriggerOnboardingBridge(client)(INPUT);
    expect(result).toMatchObject({ outcome: 'pending' });
    expect(result.request_id).toEqual(expect.any(String));
    expect(submitted?.job_id).toBe(123);
    expect(submitted?.job_parameters).toEqual({ request_id: result.request_id });
    expect(statements.some((statement) => statement.includes('INSERT INTO') && statement.includes(INPUT.actor))).toBe(
      true
    );
    expect(statements.every((statement) => !statement.includes('payload_base64'))).toBe(true);
  });

  it('redacts the governed payload when Lakeflow returns no run id', async () => {
    configureJobTransport();
    const statements: string[] = [];
    const client = jobClient({
      runNow: vi.fn().mockResolvedValue({}),
      executeStatement: vi.fn().mockImplementation(({ statement }: { statement: string }) => {
        statements.push(statement);
        return Promise.resolve({ status: { state: 'SUCCEEDED' }, result: { data_array: [] } });
      }),
    });

    await expect(jobTriggerOnboardingBridge(client)(INPUT)).rejects.toThrow(/no run id/);
    expect(statements.some((statement) => statement.includes('SET payload_json = NULL'))).toBe(true);
  });

  it('returns the persisted result before consulting run state', async () => {
    configureJobTransport();
    const getRun = vi.fn().mockResolvedValue({ state: { life_cycle_state: 'RUNNING' } });
    const client = jobClient({
      getRun,
      executeStatement: vi.fn().mockImplementation(({ statement }: { statement: string }) =>
        Promise.resolve({
          status: { state: 'SUCCEEDED' },
          result: {
            data_array: statement.startsWith('SELECT run_id')
              ? [['41']]
              : [[JSON.stringify({ outcome: 'registered', agent_name: 'agent-a' })]],
          },
        })
      ),
    });
    await expect(readJobOnboardingStatus('request-1', INPUT.actor, client)).resolves.toEqual({
      outcome: 'registered',
      agent_name: 'agent-a',
    });
    expect(getRun).not.toHaveBeenCalled();
  });

  it('stays pending while active and through a successful result-table visibility race', async () => {
    configureJobTransport();
    const active = jobClient();
    await expect(readJobOnboardingStatus('request-1', INPUT.actor, active)).resolves.toMatchObject({
      outcome: 'pending',
      request_id: 'request-1',
    });

    const terminal = jobClient({
      getRun: vi.fn().mockResolvedValue({
        state: { life_cycle_state: 'TERMINATED', result_state: 'SUCCESS' },
      }),
    });
    await expect(readJobOnboardingStatus('request-1', INPUT.actor, terminal)).resolves.toMatchObject({
      outcome: 'pending',
      request_id: 'request-1',
    });
  });

  it('falls back to the successful task JSON when the result row is not visible', async () => {
    configureJobTransport();
    const client = jobClient({
      getRun: vi.fn().mockResolvedValue({
        state: { life_cycle_state: 'TERMINATED', result_state: 'SUCCESS' },
        tasks: [{ run_id: 99 }],
      }),
      getRunOutput: vi.fn().mockResolvedValue({
        logs: 'warning: noisy startup\n{"outcome":"validated","experiment_id":"exp-1","fresh":true}\n',
      }),
    });

    await expect(readJobOnboardingStatus('request-1', INPUT.actor, client)).resolves.toEqual({
      outcome: 'validated',
      experiment_id: 'exp-1',
      fresh: true,
    });
  });

  it('surfaces a failed terminal run honestly when no result exists', async () => {
    configureJobTransport();
    const terminal = jobClient({
      getRun: vi.fn().mockResolvedValue({
        state: { life_cycle_state: 'INTERNAL_ERROR', result_state: 'FAILED', state_message: 'boom' },
      }),
    });

    await expect(readJobOnboardingStatus('request-1', INPUT.actor, terminal)).resolves.toEqual({
      outcome: 'error',
      error: 'onboarding job ended INTERNAL_ERROR/FAILED: boom',
    });
  });

  it('refuses status access when the authenticated actor does not own the request', async () => {
    configureJobTransport();
    const getRun = vi.fn().mockResolvedValue({ state: { life_cycle_state: 'RUNNING' } });
    const client = jobClient({
      getRun,
      executeStatement: vi.fn().mockResolvedValue({ status: { state: 'SUCCEEDED' }, result: { data_array: [] } }),
    });
    await expect(readJobOnboardingStatus('request-1', 'attacker@example.com', client)).rejects.toThrow(/owned/);
    expect(getRun).not.toHaveBeenCalled();
  });
});
