import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { mkdtempSync, rmSync, writeFileSync, chmodSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import {
  adaptLabelingApiClient,
  spawnPythonLabelingBridge,
  restLabelingBridge,
  selectLabelingBridge,
  resolveLabelingTransport,
  type LabelingAction,
  type LabelingRestClient,
} from './bridge';

// Hermetic subprocess test: the bridge hardcodes `-m ail.labeling.service`, so we
// point `pythonBin` at tiny stub interpreters that ignore those args and stand in for
// the Python engine — proving the bridge's happy path (stdin action -> parsed stdout
// result) and its fail-closed handling (non-zero exit / unparseable output / spawn
// failure) without needing the `ail` package or a live workspace.

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

const INPUT: LabelingAction = {
  action: 'label',
  actor: 'a@b.com',
  experiment_id: 'exp-1',
  trace_id: 't1',
  name: 'correctness',
  value: 'pass',
};

beforeAll(() => {
  dir = mkdtempSync(path.join(tmpdir(), 'ail-labeling-bridge-'));
  echoStub = stub('echo.sh', 'cat'); // echo stdin JSON back as the "result"
  failStub = stub('fail.sh', 'echo "boom" 1>&2\nexit 1');
  garbageStub = stub('garbage.sh', "printf 'not json {'");
});

afterAll(() => {
  rmSync(dir, { recursive: true, force: true });
});

describe('spawnPythonLabelingBridge — subprocess transport', () => {
  it('writes the action on stdin and parses the JSON result from stdout', async () => {
    const bridge = spawnPythonLabelingBridge({ pythonBin: echoStub, srcPath: dir });
    const result = await bridge(INPUT);
    // the echo stub returns the action verbatim -> the bridge parsed it as the result
    expect(result).toMatchObject({ action: 'label', actor: 'a@b.com', name: 'correctness' });
  });

  it('rejects a non-zero exit (never a fabricated result)', async () => {
    const bridge = spawnPythonLabelingBridge({ pythonBin: failStub, srcPath: dir });
    await expect(bridge(INPUT)).rejects.toThrow(/exited 1/);
  });

  it('rejects unparseable stdout (never a fabricated result)', async () => {
    const bridge = spawnPythonLabelingBridge({ pythonBin: garbageStub, srcPath: dir });
    await expect(bridge(INPUT)).rejects.toThrow(/unparseable/);
  });

  it('rejects when the interpreter cannot be spawned', async () => {
    const bridge = spawnPythonLabelingBridge({
      pythonBin: path.join(dir, 'does-not-exist'),
      srcPath: dir,
    });
    await expect(bridge(INPUT)).rejects.toBeInstanceOf(Error);
  });
});

describe('selectLabelingBridge', () => {
  it('returns the subprocess transport (a callable bridge)', () => {
    expect(typeof selectLabelingBridge()).toBe('function');
  });
});

// ---------------------------------------------------------------------------
// restLabelingBridge — the deployed (Node-only) MLflow REST transport.
// ---------------------------------------------------------------------------
// Driven by a FAKE LabelingRestClient (no live workspace, no HTTP), exactly as the
// approvals jobTriggerApplyBridge is fake-driven. The endpoints/shapes the real adapter
// speaks are grounded live (see the bridge's header comment); these tests prove the
// bridge's fail-closed logic and the name-match + server-set-actor + two-tier invariants.

type CreateCall = { location: string; traceId: string; body: unknown };
interface FakeRestOpts {
  scorers?: Array<{ name?: string; serialized_scorer?: string }> | Error;
  traces?: Array<Record<string, unknown>> | Error;
  created?: { assessment_id?: string } | Error;
}
function fakeRestClient(opts: FakeRestOpts) {
  const calls = { listScorers: 0, searchTraces: 0, createAssessment: [] as CreateCall[] };
  const client: LabelingRestClient = {
    listScorers() {
      calls.listScorers += 1;
      if (opts.scorers instanceof Error) return Promise.reject(opts.scorers);
      return Promise.resolve(opts.scorers ?? []);
    },
    searchTraces() {
      calls.searchTraces += 1;
      if (opts.traces instanceof Error) return Promise.reject(opts.traces);
      return Promise.resolve(opts.traces ?? []);
    },
    createAssessment(location, traceId, body) {
      calls.createAssessment.push({ location, traceId, body });
      if (opts.created instanceof Error) return Promise.reject(opts.created);
      return Promise.resolve(opts.created ?? {});
    },
  };
  return { client, calls };
}

// A recent trace carrying zero (or the given) HUMAN assessment names.
function traceInfo(bare: string, humanNames: string[] = []) {
  return {
    trace_id: bare,
    trace_location: { uc_table_prefix: { catalog_name: 'cat', schema_name: 'sch', table_prefix: 'pfx' } },
    request_preview: `req ${bare}`,
    request_time: '2026-07-08T00:00:00Z',
    assessments: humanNames.map((n) => ({ assessment_name: n, source: { source_type: 'HUMAN', source_id: 'x' } })),
  };
}

const DIMS: LabelingAction = { action: 'dimensions', actor: 'labeler@databricks.com', experiment_id: 'exp-1' };
const LABEL: LabelingAction = {
  action: 'label',
  actor: 'labeler@databricks.com',
  experiment_id: 'exp-1',
  trace_id: 'trace:/cat.sch.pfx/tid1',
  name: 'correctness',
  value: 'yes',
  rationale: 'clear evidence',
};

// The (open) result shapes the bridge returns — a typed lens over LabelingResult so the
// assertions are `unknown`-free (no `any`, matching the eslint bar the rest of the suite
// holds).
interface DimResult {
  outcome: string;
  label_floor?: number;
  dimensions?: Array<{
    name: string;
    labels_so_far: number;
    label_floor?: number;
    remaining?: number;
    complete?: boolean;
  }>;
  traces?: Array<{ trace_id: string; labeled: Record<string, boolean> }>;
  scanned?: number;
  error?: string;
}
interface WriteResult {
  outcome: string;
  name?: string;
  error?: string;
  refused_reason?: string;
  labels_so_far?: number;
  label_floor?: number;
  remaining?: number;
}

describe('restLabelingBridge — dimensions (read) from judges + scanned traces', () => {
  it('lists registered judges, counts HUMAN labels, and builds the worklist', async () => {
    const fake = fakeRestClient({
      scorers: [{ name: 'correctness' }, { name: 'modularity' }],
      traces: [traceInfo('tid1', ['correctness']), traceInfo('tid2', [])],
    });
    const bridge = restLabelingBridge({ warehouseId: 'wh-1', labelFloor: 33, clientFactory: () => fake.client });
    const res = (await bridge(DIMS)) as DimResult;

    expect(res.outcome).toBe('dimensions');
    expect(res.label_floor).toBe(33); // relayed verbatim, not hardcoded
    const correctness = res.dimensions?.find((d) => d.name === 'correctness');
    expect(correctness).toMatchObject({ labels_so_far: 1, label_floor: 33, remaining: 32, complete: false });
    // Both traces are on the worklist (each still misses at least one dimension);
    // the worklist trace id is the FULL v4 id the write needs.
    const traces = res.traces ?? [];
    expect(traces).toHaveLength(2);
    expect(traces[0]?.trace_id).toBe('trace:/cat.sch.pfx/tid1');
    expect(traces[0]?.labeled).toEqual({ correctness: true, modularity: false });
    expect(res.scanned).toBe(2);
  });

  it('excludes deterministic custom-code scorers from human labeling dimensions', async () => {
    const fake = fakeRestClient({
      scorers: [
        { name: 'accuracy_and_correctness' },
        {
          name: 'duration_seconds',
          serialized_scorer: JSON.stringify({
            original_func_name: 'duration_seconds_scorer',
            call_source: 'return 1.0',
          }),
        },
      ],
      traces: [traceInfo('tid1', [])],
    });
    const bridge = restLabelingBridge({ warehouseId: 'wh-1', clientFactory: () => fake.client });
    const res = (await bridge(DIMS)) as DimResult;

    expect(res.outcome).toBe('dimensions');
    expect(res.dimensions?.map((dimension) => dimension.name)).toEqual(['accuracy_and_correctness']);
    expect(res.traces?.[0]?.labeled).toEqual({ accuracy_and_correctness: false });
  });

  it('two-tier: with no floor relayed, omits label_floor entirely (never a hardcoded number)', async () => {
    const fake = fakeRestClient({ scorers: [{ name: 'correctness' }], traces: [traceInfo('tid1', [])] });
    const bridge = restLabelingBridge({ warehouseId: 'wh-1', clientFactory: () => fake.client });
    const res = (await bridge(DIMS)) as DimResult;
    expect(res.outcome).toBe('dimensions');
    expect(res).not.toHaveProperty('label_floor');
    const dim0 = (res.dimensions ?? [])[0];
    expect(dim0).toBeDefined();
    expect(dim0).not.toHaveProperty('label_floor');
    // no fabricated digit anywhere a floor would be
    expect(JSON.stringify(res)).not.toMatch(/"label_floor"/);
  });

  it('fails closed (honest error) when the registered judges cannot be determined — never invents dimensions', async () => {
    const fake = fakeRestClient({ scorers: new Error('scorer backend down') });
    const bridge = restLabelingBridge({ warehouseId: 'wh-1', clientFactory: () => fake.client });
    const res = (await bridge(DIMS)) as DimResult;
    expect(res.outcome).toBe('error');
    expect(res.outcome).not.toBe('dimensions');
    expect(res.error).toMatch(/registered judges|invent/i);
    expect(fake.calls.searchTraces).toBe(0);
  });

  it('fails closed (honest error, use MLflow UI) when the trace scan fails', async () => {
    const fake = fakeRestClient({ scorers: [{ name: 'correctness' }], traces: new Error('warehouse asleep') });
    const bridge = restLabelingBridge({ warehouseId: 'wh-1', clientFactory: () => fake.client });
    const res = (await bridge(DIMS)) as DimResult;
    expect(res.outcome).toBe('error');
    expect(res.error).toMatch(/MLflow Traces UI/);
  });
});

describe('adaptLabelingApiClient — paginated MLflow trace search', () => {
  it('never exceeds the 1000-row API limit and follows next_page_token', async () => {
    const requests: Array<{ path: string; method: string; payload?: Record<string, unknown> }> = [];
    let page = 0;
    const request: Parameters<typeof adaptLabelingApiClient>[0] = (apiRequest) => {
      requests.push({
        path: apiRequest.path,
        method: apiRequest.method,
        ...(apiRequest.payload && typeof apiRequest.payload === 'object'
          ? { payload: apiRequest.payload as Record<string, unknown> }
          : {}),
      });
      if (apiRequest.method === 'POST') {
        page += 1;
        return Promise.resolve({ name: `operation-${page}` });
      }
      const pathParts = apiRequest.path.split('-');
      const operation = Number(pathParts[pathParts.length - 1]);
      if (operation === 1) {
        return Promise.resolve({
          done: true,
          response: {
            trace_infos: Array.from({ length: 1_000 }, (_, index) => traceInfo(`first-${index}`)),
            next_page_token: 'page-2',
          },
        });
      }
      return Promise.resolve({
        done: true,
        response: {
          trace_infos: Array.from({ length: 500 }, (_, index) => traceInfo(`second-${index}`)),
        },
      });
    };

    const client = adaptLabelingApiClient(request, {
      warehouseId: 'wh-1',
      searchTimeoutMs: 1_000,
      searchPollMs: 0,
    });
    const traces = await client.searchTraces('exp-1', 1_500);
    const starts = requests.filter((request) => request.method === 'POST');

    expect(traces).toHaveLength(1_500);
    expect(starts.map((request) => request.payload?.max_results)).toEqual([1_000, 500]);
    expect(starts[0]?.payload).not.toHaveProperty('page_token');
    expect(starts[1]?.payload).toMatchObject({ page_token: 'page-2' });
  });

  it('surfaces a failed async search operation', async () => {
    const request: Parameters<typeof adaptLabelingApiClient>[0] = (apiRequest) =>
      Promise.resolve(
        apiRequest.method === 'POST'
          ? { name: 'operation-1' }
          : { done: true, error: { error_code: 'INTERNAL_ERROR', message: 'warehouse failed' } }
      );
    const client = adaptLabelingApiClient(request, {
      warehouseId: 'wh-1',
      searchTimeoutMs: 1_000,
      searchPollMs: 0,
    });

    await expect(client.searchTraces('exp-1', 100)).rejects.toThrow(/warehouse failed/);
  });
});

describe('restLabelingBridge — label (write) is name-matched, HUMAN, server-set actor, fail-closed', () => {
  it('writes a confirmed HUMAN assessment named for the judge and returns labeled', async () => {
    const fake = fakeRestClient({
      scorers: [{ name: 'correctness' }, { name: 'modularity' }],
      created: { assessment_id: 'a-123' },
    });
    const bridge = restLabelingBridge({ warehouseId: 'wh-1', labelFloor: 20, clientFactory: () => fake.client });
    const res = (await bridge(LABEL)) as WriteResult;

    expect(res.outcome).toBe('labeled');
    expect(res.name).toBe('correctness');
    expect(fake.calls.createAssessment).toHaveLength(1);
    const call = fake.calls.createAssessment[0];
    expect(call?.location).toBe('cat.sch.pfx'); // parsed from the v4 trace id
    expect(call?.traceId).toBe('tid1');
    // name-match + HUMAN source + AUTHENTICATED actor as source_id (never from the body)
    expect(call?.body).toMatchObject({
      assessment_name: 'correctness',
      trace_id: 'trace:/cat.sch.pfx/tid1',
      source: { source_type: 'HUMAN', source_id: 'labeler@databricks.com' },
      feedback: { value: 'yes' },
      rationale: 'clear evidence',
    });
    // Rapid-fire: the write does NOT do a second trace scan for inline progress — it
    // returns immediately; the panel refetches `dimensions` to update the cards. The
    // write's `searchTraces` count stays 0.
    expect(fake.calls.searchTraces).toBe(0);
    expect(res).not.toHaveProperty('labels_so_far');
  });

  it('refuses a name that is not a registered judge — never writes (the name-match guard)', async () => {
    const fake = fakeRestClient({ scorers: [{ name: 'correctness' }], created: { assessment_id: 'a-1' } });
    const bridge = restLabelingBridge({ warehouseId: 'wh-1', clientFactory: () => fake.client });
    const res = (await bridge({ ...LABEL, name: 'not_a_judge' })) as WriteResult;
    expect(res.outcome).toBe('refused');
    expect(res.refused_reason).toMatch(/not a registered judge/);
    expect(fake.calls.createAssessment).toHaveLength(0);
  });

  it('appends an alignment event only after MLflow confirms the HUMAN assessment', async () => {
    const fake = fakeRestClient({ scorers: [{ name: 'correctness' }], created: { assessment_id: 'a-1' } });
    const events: unknown[] = [];
    const bridge = restLabelingBridge({
      warehouseId: 'wh-1',
      clientFactory: () => fake.client,
      eventSink: (event) => {
        events.push(event);
        return Promise.resolve();
      },
    });
    const res = (await bridge(LABEL)) as WriteResult;
    expect(res.outcome).toBe('labeled');
    expect(events).toEqual([
      expect.objectContaining({ experimentId: LABEL.experiment_id, assessmentId: 'a-1', judgeName: LABEL.name }),
    ]);
  });

  it('keeps a confirmed label successful when event append defers to daily recovery', async () => {
    const fake = fakeRestClient({ scorers: [{ name: 'correctness' }], created: { assessment_id: 'a-1' } });
    const bridge = restLabelingBridge({
      warehouseId: 'wh-1',
      clientFactory: () => fake.client,
      eventSink: () => Promise.reject(new Error('warehouse starting')),
    });
    const res = (await bridge(LABEL)) as WriteResult & { event_warning?: string };
    expect(res.outcome).toBe('labeled');
    expect(res.event_warning).toMatch(/daily|deferred|wake-up/i);
  });

  it('FAIL-CLOSED: a write with no returned assessment_id is an honest error, never a fabricated label', async () => {
    const fake = fakeRestClient({ scorers: [{ name: 'correctness' }], created: {} }); // no assessment_id
    const bridge = restLabelingBridge({ warehouseId: 'wh-1', clientFactory: () => fake.client });
    const res = (await bridge(LABEL)) as WriteResult;
    expect(res.outcome).toBe('error');
    expect(res.outcome).not.toBe('labeled');
    expect(res.error).toMatch(/no assessment id|never fabricating|MLflow Traces UI/i);
  });

  it('a write failure is an honest error, never a fake label', async () => {
    const fake = fakeRestClient({ scorers: [{ name: 'correctness' }], created: new Error('PERMISSION_DENIED') });
    const bridge = restLabelingBridge({ warehouseId: 'wh-1', clientFactory: () => fake.client });
    const res = (await bridge(LABEL)) as WriteResult;
    expect(res.outcome).toBe('error');
    expect(res.error).toMatch(/PERMISSION_DENIED/);
  });

  it('cannot confirm the judge set → honest error and NEVER attempts the write', async () => {
    const fake = fakeRestClient({ scorers: new Error('list scorers 500') });
    const bridge = restLabelingBridge({ warehouseId: 'wh-1', clientFactory: () => fake.client });
    const res = (await bridge(LABEL)) as WriteResult;
    expect(res.outcome).toBe('error');
    expect(res.error).toMatch(/cannot be confirmed|registered judges/i);
    expect(fake.calls.createAssessment).toHaveLength(0);
  });

  it('refuses a non-v4 trace id (points to the MLflow Traces UI) and never writes', async () => {
    const fake = fakeRestClient({ scorers: [{ name: 'correctness' }], created: { assessment_id: 'a-1' } });
    const bridge = restLabelingBridge({ warehouseId: 'wh-1', clientFactory: () => fake.client });
    const res = (await bridge({ ...LABEL, trace_id: 'bare-id-no-location' })) as WriteResult;
    expect(res.outcome).toBe('error');
    expect(res.error).toMatch(/MLflow Traces UI|v4/);
    expect(fake.calls.createAssessment).toHaveLength(0);
  });

  it('refuses an anonymous label (no authenticated actor) and never writes', async () => {
    const fake = fakeRestClient({ scorers: [{ name: 'correctness' }] });
    const bridge = restLabelingBridge({ warehouseId: 'wh-1', clientFactory: () => fake.client });
    const res = (await bridge({ ...LABEL, actor: '' })) as WriteResult;
    expect(res.outcome).toBe('refused');
    expect(res.refused_reason).toMatch(/anonymous/);
    expect(fake.calls.listScorers).toBe(0);
    expect(fake.calls.createAssessment).toHaveLength(0);
  });
});

describe('restLabelingBridge — unavailable deps fail closed to an honest state (never a fake success)', () => {
  it('no SQL warehouse → honest "deployed labeling unavailable — use the MLflow Traces UI" (both actions)', async () => {
    let built = 0;
    const bridge = restLabelingBridge({
      warehouseId: '',
      clientFactory: () => {
        built += 1;
        return fakeRestClient({}).client;
      },
    });
    const dimRes = (await bridge(DIMS)) as DimResult;
    const labelRes = (await bridge(LABEL)) as WriteResult;
    expect(dimRes.outcome).toBe('error');
    expect(dimRes.error).toMatch(/deployed labeling unavailable.*MLflow Traces UI/i);
    expect(labelRes.outcome).toBe('error');
    expect(labelRes.outcome).not.toBe('labeled');
    expect(labelRes.error).toMatch(/MLflow Traces UI/);
    expect(built).toBe(0); // never even constructs a client
  });
});

describe('resolveLabelingTransport — env-driven transport selection', () => {
  it('defaults to the subprocess transport', () => {
    expect(resolveLabelingTransport({})).toBe('subprocess');
  });
  it('selects the REST transport when AIL_LABELING_TRANSPORT=rest', () => {
    expect(resolveLabelingTransport({ AIL_LABELING_TRANSPORT: 'rest' })).toBe('rest');
  });
  it('treats any other value as the subprocess transport', () => {
    expect(resolveLabelingTransport({ AIL_LABELING_TRANSPORT: 'job' })).toBe('subprocess');
  });
});
