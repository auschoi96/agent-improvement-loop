import { spawn } from 'node:child_process';
import { createHash, randomUUID } from 'node:crypto';
import path from 'node:path';
import { WorkspaceClient } from '@databricks/sdk-experimental';

// The JSON action the authenticated onboarding route hands the (Python) engine.
// `actor` is set SERVER-SIDE by the route (from the authenticated request) — it is
// never trusted from the browser (mirrors the approvals write-path's `approver`).
export interface OnboardingAction {
  action:
    | 'requirements'
    | 'validate_experiment'
    | 'create_experiment'
    | 'register_agent'
    | 'preview_requirements'
    | 'confirm_requirements'
    | 'bootstrap_agent';
  actor: string;
  goals?: string[];
  experiment_id?: string;
  name?: string;
  agent_name?: string;
  // Free-form requirements intake (preview_requirements / confirm_requirements).
  // The engine owns extraction/routing/target facts (two-tier); the client only
  // relays the raw text and the human's explicit objective target back.
  requirements_text?: string;
  objective_target?: number;
  cohort?: string;
  // Extended registry fields for register_agent (Slice 4). The engine sets them on
  // the persisted Agent and validates their types fail-closed. `goal_config` is the
  // requirements-confirmed goal (opaque here — relayed verbatim); the two others are
  // the executor's target workspace and the memory job's annotations table.
  goal_config?: Record<string, unknown>;
  reviewer_experiment_id?: string;
  annotations_table?: string;
  target_workspace?: string;
  trace_catalog?: string;
  trace_schema?: string;
  trace_table_prefix?: string;
  allow_existing?: boolean;
}

// The JSON ail.onboarding.service prints — a typed onboarding result. Kept open
// (outcome + passthrough fields) so the route returns it verbatim and the client
// wizard renders it (never re-deriving the goal/gate/registry facts in TS).
export interface OnboardingResult {
  outcome: string;
  [key: string]: unknown;
}

// The seam the route calls to run the (Python) onboarding engine server-side.
// Injectable so the route is unit-testable with a fake bridge (no subprocess, no
// live write) — exactly as ail.loop.apply_service's ApplyBridge seam is.
export type OnboardingBridge = (input: OnboardingAction) => Promise<OnboardingResult>;

interface SpawnBridgeOptions {
  /** Python interpreter that has the `ail` package importable. */
  pythonBin?: string;
  /** Extra PYTHONPATH entry so `python -m ail.onboarding.service` resolves in dev. */
  srcPath?: string;
  /** Hard timeout for the subprocess (ms). */
  timeoutMs?: number;
}

// The default bridge: run `python -m ail.onboarding.service`, write the JSON action
// on stdin, read the typed result JSON from stdout. The engine runs under the app's
// identity (the framework service principal / the dev profile); it performs the
// permission-sensitive read/writes and returns an HONEST result — an experiment is
// only "created" when MLflow created it, an agent only "registered" when the
// registry write succeeded (see src/ail/onboarding/service.py).
//
// Deployment note (mirrors docs/LOOP_CONTROLLER.md for approvals): the deployed
// Databricks App image is Node-only — the `ail` wheel runs as serverless Jobs — so
// this subprocess bridge is the local-dev / self-hosted transport. A Databricks
// Job-trigger transport for the deployed image (the analogue of the approvals
// `jobTriggerApplyBridge`) is a documented FOLLOW-ON: it needs a wheel-task entry
// point + a job resource, which are out of scope for this slice (new plugin +
// client only). The `OnboardingBridge` seam is unchanged when that lands.
export function spawnPythonOnboardingBridge(options: SpawnBridgeOptions = {}): OnboardingBridge {
  const pythonBin =
    options.pythonBin ?? process.env.AIL_ONBOARDING_PYTHON_BIN ?? process.env.AIL_APPLY_PYTHON_BIN ?? 'python3';
  // The app runs from ail-self-optimizer/; the ail package source is ../src.
  const srcPath =
    options.srcPath ??
    process.env.AIL_ONBOARDING_SRC_PATH ??
    process.env.AIL_APPLY_SRC_PATH ??
    path.resolve(process.cwd(), '..', 'src');
  const timeoutMs = options.timeoutMs ?? 120_000;

  return (input: OnboardingAction) =>
    new Promise<OnboardingResult>((resolve, reject) => {
      const existing = process.env.PYTHONPATH;
      const child = spawn(pythonBin, ['-m', 'ail.onboarding.service'], {
        env: { ...process.env, PYTHONPATH: existing ? `${srcPath}${path.delimiter}${existing}` : srcPath },
      });

      let stdout = '';
      let stderr = '';
      const timer = setTimeout(() => {
        child.kill('SIGKILL');
        reject(new Error(`onboarding-service timed out after ${timeoutMs}ms`));
      }, timeoutMs);

      // Handle stdin stream errors: a child that exits WITHOUT reading its stdin
      // (a fast-failing engine, or a stub that ignores input) makes the write below
      // emit EPIPE on this stream. Without a handler that EPIPE surfaces as an
      // UNHANDLED exception AFTER the tests finish — vitest catches it as an uncaught
      // error and fails the AppKit gate even though every test passed (a timing race
      // that passes locally, fails in CI). The child's exit code + stdout are the
      // real signal (via the 'close'/'error' handlers), so a broken input pipe is
      // safely swallowed here. The approvals subprocess bridge never needed this
      // because its tests exercise only the job transport, not spawnPython*.
      child.stdin.on('error', () => {});
      child.stdout.on('data', (d: Buffer) => (stdout += d.toString()));
      child.stderr.on('data', (d: Buffer) => (stderr += d.toString()));
      child.on('error', (err) => {
        clearTimeout(timer);
        reject(err);
      });
      child.on('close', (code) => {
        clearTimeout(timer);
        if (code !== 0) {
          reject(new Error(`onboarding-service exited ${code}: ${stderr.trim() || stdout.trim()}`));
          return;
        }
        try {
          resolve(JSON.parse(stdout) as OnboardingResult);
        } catch {
          reject(new Error(`onboarding-service returned unparseable output: ${stdout.slice(0, 500)}`));
        }
      });

      // Guard the write itself: if the pipe is already torn down the synchronous
      // call can throw. Swallow it — the 'close'/'error' handlers above settle the
      // promise from the child's real exit code / output, never a write-after-close.
      try {
        child.stdin.write(JSON.stringify(input));
        child.stdin.end();
      } catch {
        // stdin already closed; the outcome comes from the child's exit + stdout.
      }
    });
}

interface RunStateLike {
  life_cycle_state?: string;
  result_state?: string;
  state_message?: string;
}

export interface OnboardingJobClient {
  runNow(req: {
    job_id: number;
    job_parameters: Record<string, string>;
    idempotency_token?: string;
  }): Promise<{ run_id?: number }>;
  getRun(req: { run_id: number }): Promise<{
    state?: RunStateLike;
    tasks?: Array<{ run_id?: number }>;
  }>;
  getRunOutput(req: { run_id: number }): Promise<{ logs?: string }>;
  executeStatement(req: { warehouse_id: string; statement: string; wait_timeout?: string }): Promise<{
    statement_id?: string;
    status?: { state?: string; error?: { message?: string } };
    result?: { data_array?: string[][] };
  }>;
  getStatement(req: { statement_id: string }): Promise<{
    statement_id?: string;
    status?: { state?: string; error?: { message?: string } };
    result?: { data_array?: string[][] };
  }>;
}

const TERMINAL_STATES = new Set(['TERMINATED', 'SKIPPED', 'INTERNAL_ERROR']);
const wait = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));
const quote = (value: string): string => `'${value.replace(/'/g, "''")}'`;

function onboardingClient(): OnboardingJobClient {
  const workspace = new WorkspaceClient({});
  return {
    runNow: (req) => workspace.jobs.runNow(req),
    getRun: (req) => workspace.jobs.getRun(req),
    getRunOutput: (req) => workspace.jobs.getRunOutput(req),
    executeStatement: (req) => workspace.statementExecution.executeStatement(req),
    getStatement: (req) => workspace.statementExecution.getStatement(req),
  };
}

function resultFromTaskLogs(logs: string | undefined): OnboardingResult | null {
  if (!logs) return null;
  const lines = logs.split(/\r?\n/);
  for (let index = lines.length - 1; index >= 0; index -= 1) {
    const line = lines[index].trim();
    if (!line.startsWith('{')) continue;
    try {
      const parsed = JSON.parse(line) as OnboardingResult;
      if (typeof parsed.outcome === 'string') return parsed;
    } catch {
      // Warnings and ordinary logs may contain braces; only a parseable result wins.
    }
  }
  return null;
}

async function readOnboardingResult(
  client: OnboardingJobClient,
  requestId: string,
  warehouseId: string,
  catalog: string,
  schema: string
): Promise<string | null> {
  const statement =
    `SELECT result_json FROM \`${catalog}\`.\`${schema}\`.agent_onboarding_results ` +
    `WHERE request_id = ${quote(requestId)} ORDER BY recorded_at DESC LIMIT 1`;
  let response = await client.executeStatement({ warehouse_id: warehouseId, statement, wait_timeout: '50s' });
  const deadline = Date.now() + 60_000;
  while (response.status?.state === 'PENDING' || response.status?.state === 'RUNNING') {
    if (Date.now() >= deadline || !response.statement_id) return null;
    await wait(1_000);
    response = await client.getStatement({ statement_id: response.statement_id });
  }
  if (response.status?.state !== 'SUCCEEDED') {
    const message = response.status?.error?.message ?? `onboarding result read ${response.status?.state ?? 'failed'}`;
    const normalized = message.toLowerCase();
    if (normalized.includes('table_or_view_not_found') || normalized.includes('does not exist')) return null;
    throw new Error(message);
  }
  return response.result?.data_array?.[0]?.[0] ?? null;
}

export function jobTriggerOnboardingBridge(client: OnboardingJobClient = onboardingClient()): OnboardingBridge {
  const jobId = Number(process.env.AIL_ONBOARDING_JOB_ID);
  const warehouseId = process.env.DATABRICKS_WAREHOUSE_ID;
  const catalog = process.env.AIL_CATALOG;
  const schema = process.env.AIL_SCHEMA;
  return async (input) => {
    if (!Number.isFinite(jobId)) throw new Error('AIL_ONBOARDING_JOB_ID is not configured');
    if (!warehouseId || !catalog || !schema) {
      throw new Error('DATABRICKS_WAREHOUSE_ID, AIL_CATALOG, and AIL_SCHEMA are required for onboarding');
    }
    const requestId = randomUUID();
    const payload = Buffer.from(JSON.stringify(input), 'utf8').toString('base64');
    const started = await client.runNow({
      job_id: jobId,
      job_parameters: { request_id: requestId, payload_base64: payload },
      idempotency_token: createHash('sha256').update(requestId).digest('hex'),
    });
    if (started.run_id == null) throw new Error(`onboarding job ${jobId} returned no run id`);
    return { outcome: 'pending', request_id: requestId, run_id: started.run_id };
  };
}

export async function readJobOnboardingStatus(
  requestId: string,
  runId: number,
  client: OnboardingJobClient = onboardingClient()
): Promise<OnboardingResult> {
  const warehouseId = process.env.DATABRICKS_WAREHOUSE_ID;
  const catalog = process.env.AIL_CATALOG;
  const schema = process.env.AIL_SCHEMA;
  if (!requestId || !Number.isFinite(runId)) throw new Error('request_id and run_id are required');
  if (!warehouseId || !catalog || !schema) {
    throw new Error('DATABRICKS_WAREHOUSE_ID, AIL_CATALOG, and AIL_SCHEMA are required for onboarding');
  }
  const json = await readOnboardingResult(client, requestId, warehouseId, catalog, schema);
  if (json) return JSON.parse(json) as OnboardingResult;
  const run = await client.getRun({ run_id: runId });
  const lifecycle = run.state?.life_cycle_state;
  if (lifecycle && TERMINAL_STATES.has(lifecycle)) {
    // The Job can become SUCCESS milliseconds before the result-table INSERT is
    // visible to this warehouse read. Keep polling through that read-after-write
    // window; a genuine missing result will hit the client's bounded 15-minute
    // timeout instead of surfacing a false terminal error immediately.
    if (lifecycle === 'TERMINATED' && run.state?.result_state === 'SUCCESS') {
      const taskRunIds = (run.tasks ?? []).flatMap((task) => (task.run_id == null ? [] : [task.run_id])).reverse();
      for (const taskRunId of taskRunIds) {
        try {
          const output = await client.getRunOutput({ run_id: taskRunId });
          const result = resultFromTaskLogs(output.logs);
          if (result) return result;
        } catch {
          // The durable result-table read remains primary. If task output is briefly
          // unavailable, preserve the bounded pending behavior and retry both paths.
        }
      }
      return { outcome: 'pending', request_id: requestId, run_id: runId };
    }
    return {
      outcome: 'error',
      error: `onboarding job ended ${lifecycle}/${run.state?.result_state ?? '—'}: ${run.state?.state_message ?? ''}`,
    };
  }
  return { outcome: 'pending', request_id: requestId, run_id: runId };
}

// Transport selection — the subprocess transport for this slice. Kept as a function
// (not a bare export) so a future Node-only Job transport can be selected by env
// here without touching the route, exactly as approvals' selectApplyBridge does.
export function selectOnboardingBridge(): OnboardingBridge {
  return process.env.AIL_ONBOARDING_TRANSPORT === 'job' || process.env.AIL_ONBOARDING_JOB_ID
    ? jobTriggerOnboardingBridge()
    : spawnPythonOnboardingBridge();
}
