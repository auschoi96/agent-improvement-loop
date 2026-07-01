import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { WorkspaceClient } from '@databricks/sdk-experimental';

// The decision the authenticated route hands the apply engine. `approver` and
// `decided_at` are set SERVER-SIDE by the route (from the authenticated request) —
// never trusted from the browser.
export interface DecisionInput {
  proposal_id: string;
  agent_name: string;
  decision: 'approve' | 'reject';
  reason?: string;
  approver: string;
  decided_at: string;
}

// The JSON ail.loop.apply_service prints — an ApplyServiceResult. Kept open (outcome
// + passthrough fields) so the route returns it verbatim and the client renders it.
export interface BridgeResult {
  outcome: string;
  [key: string]: unknown;
}

// The seam the route calls to run the (Python) apply engine server-side. Injectable
// so the route is unit-testable with a fake bridge (no subprocess, no live write).
export type ApplyBridge = (input: DecisionInput) => Promise<BridgeResult>;

interface SpawnBridgeOptions {
  /** Python interpreter that has the `ail` package importable. */
  pythonBin?: string;
  /** Extra PYTHONPATH entry so `python -m ail.loop.apply_service` resolves in dev. */
  srcPath?: string;
  /** Hard timeout for the subprocess (ms). */
  timeoutMs?: number;
}

// The default bridge: run `python -m ail.loop.apply_service`, write the decision as
// JSON on stdin, read the ApplyServiceResult JSON from stdout. The engine runs under
// the framework service principal (the deployed app's identity / the dev profile);
// it re-checks the proof + gate and performs the gated apply. The UI only triggers.
//
// Deployment note (docs/LOOP_CONTROLLER.md): the deployed Databricks App image is
// Node-only — the `ail` wheel runs as serverless Jobs — so this subprocess bridge is
// the local-dev/self-hosted transport. The Node-only transport is now BUILT as
// `jobTriggerApplyBridge` below (a Databricks Job trigger running the *same*
// ail.loop.apply_service engine); `selectApplyBridge` picks between them by env. The
// engine, the authenticated route, and the queue are unchanged — only the transport.
export function spawnPythonApplyBridge(options: SpawnBridgeOptions = {}): ApplyBridge {
  const pythonBin = options.pythonBin ?? process.env.AIL_APPLY_PYTHON_BIN ?? 'python3';
  // The app runs from ail-self-optimizer/; the ail package source is ../src.
  const srcPath = options.srcPath ?? process.env.AIL_APPLY_SRC_PATH ?? path.resolve(process.cwd(), '..', 'src');
  const timeoutMs = options.timeoutMs ?? 120_000;

  return (input: DecisionInput) =>
    new Promise<BridgeResult>((resolve, reject) => {
      const existing = process.env.PYTHONPATH;
      const child = spawn(pythonBin, ['-m', 'ail.loop.apply_service'], {
        env: { ...process.env, PYTHONPATH: existing ? `${srcPath}${path.delimiter}${existing}` : srcPath },
      });

      let stdout = '';
      let stderr = '';
      const timer = setTimeout(() => {
        child.kill('SIGKILL');
        reject(new Error(`apply-service timed out after ${timeoutMs}ms`));
      }, timeoutMs);

      child.stdout.on('data', (d: Buffer) => (stdout += d.toString()));
      child.stderr.on('data', (d: Buffer) => (stderr += d.toString()));
      child.on('error', (err) => {
        clearTimeout(timer);
        reject(err);
      });
      child.on('close', (code) => {
        clearTimeout(timer);
        if (code !== 0) {
          reject(new Error(`apply-service exited ${code}: ${stderr.trim() || stdout.trim()}`));
          return;
        }
        try {
          resolve(JSON.parse(stdout) as BridgeResult);
        } catch {
          reject(new Error(`apply-service returned unparseable output: ${stdout.slice(0, 500)}`));
        }
      });

      child.stdin.write(JSON.stringify(input));
      child.stdin.end();
    });
}

// ---------------------------------------------------------------------------
// Databricks Job-trigger bridge — the DEPLOYED (Node-only) transport.
// ---------------------------------------------------------------------------
//
// The deployed Databricks App image is Node-only: the `ail` wheel is not importable
// there (docs/DEPLOY.md), so `spawnPythonApplyBridge` cannot run. This bridge is the
// same `ApplyBridge` seam over a different transport: it triggers the pre-deployed
// `ail-apply-job` (a serverless python_wheel_task running the SAME
// `ail.loop.apply_service.run_decision`), polls to a terminal run state, and returns
// the engine's REAL result. The engine, the authenticated route, and the queue are
// unchanged — only the transport differs (docs/LOOP_CONTROLLER.md).
//
// Result retrieval (LOAD-BEARING, fail-closed): a serverless wheel task does NOT
// stream stdout back to the trigger, so the job writes its real ApplyServiceResult
// (full JSON) to the `agent_apply_results` UC Delta table keyed by (proposal_id,
// decided_at) — see src/ail/jobs/apply_job.py — and this bridge reads that row back
// AFTER the run reaches a terminal SUCCESS. It NEVER fabricates a success: a failed
// run, a run still non-terminal at the timeout, a non-SUCCESS terminal state, or a
// missing/unparseable result row all REJECT (the route surfaces that as an honest
// outcome:"error", exactly as it does for a subprocess failure).

// The framework catalog.schema the app reads from (and where apply_job writes the
// result row). Mirrors the Python DEFAULT_CATALOG/DEFAULT_SCHEMA and the app's
// config/queries/*.sql; override via env for a different workspace.
const DEFAULT_APPLY_CATALOG = 'austin_choi_omni_agent_catalog';
const DEFAULT_APPLY_SCHEMA = 'agent_improvement_loop';

// The result-handoff table apply_job writes and this bridge reads back. MUST match
// APPLY_RESULTS_TABLE in src/ail/jobs/apply_job.py.
const APPLY_RESULTS_TABLE = 'agent_apply_results';

// Terminal run lifecycle states (RunState.life_cycle_state). Only TERMINATED with
// result_state === 'SUCCESS' is a clean success; the other terminals are failures.
const TERMINAL_LIFECYCLE = new Set(['TERMINATED', 'SKIPPED', 'INTERNAL_ERROR']);

// The minimal slice of the Databricks SDK the bridge uses — narrow on purpose so a
// FAKE client is trivial to inject in tests (no live workspace). The real
// WorkspaceClient satisfies this structurally via `adaptWorkspaceClient`.
interface RunStateLike {
  life_cycle_state?: string;
  result_state?: string;
  state_message?: string;
}
interface RunLike {
  run_id?: number;
  state?: RunStateLike;
  run_page_url?: string;
}
interface StatementLike {
  statement_id?: string;
  status?: { state?: string; error?: { message?: string } };
  result?: { data_array?: string[][] };
}
export interface JobTriggerClient {
  runNow(req: {
    job_id: number;
    job_parameters: Record<string, string>;
    idempotency_token?: string;
  }): Promise<{ run_id?: number }>;
  getRun(req: { run_id: number }): Promise<RunLike>;
  executeStatement(req: { warehouse_id: string; statement: string; wait_timeout?: string }): Promise<StatementLike>;
  getStatement(req: { statement_id: string }): Promise<StatementLike>;
  host?: string;
}

export interface JobTriggerBridgeOptions {
  /** Numeric id of the pre-deployed `ail-apply-job` (env: AIL_APPLY_JOB_ID). */
  jobId?: number;
  /** SQL warehouse used to read the result row back (env: DATABRICKS_WAREHOUSE_ID). */
  warehouseId?: string;
  /** UC catalog holding `agent_apply_results` (env: AIL_APPLY_CATALOG). */
  catalog?: string;
  /** UC schema holding `agent_apply_results` (env: AIL_APPLY_SCHEMA). */
  schema?: string;
  /** Hard bound on polling the run to a terminal state, ms (env: AIL_APPLY_JOB_TIMEOUT_MS). */
  timeoutMs?: number;
  /** Interval between run-state polls, ms (env: AIL_APPLY_JOB_POLL_MS). */
  pollIntervalMs?: number;
  /** Bound on polling the result-read statement to SUCCEEDED, ms. */
  statementTimeoutMs?: number;
  /** Interval between result-read statement polls, ms. */
  statementPollMs?: number;
  /** Injectable client factory (tests pass a fake; default builds a WorkspaceClient). */
  clientFactory?: () => JobTriggerClient;
}

const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

function parseApplyJobId(raw: string | undefined): { jobId?: number; error?: string } {
  if (raw === undefined) {
    return {};
  }
  const trimmed = raw.trim();
  if (!trimmed) {
    return {};
  }
  const jobId = Number(trimmed);
  if (!Number.isFinite(jobId)) {
    return {
      error: `AIL_APPLY_JOB_ID is invalid — expected a numeric job id, got ${JSON.stringify(raw)}`,
    };
  }
  return { jobId };
}

// Render a value as a safe SQL string literal (double single-quotes). proposal_id /
// decided_at are server-controlled, but the read must not be injectable regardless.
function sqlLit(value: string): string {
  return `'${value.replace(/'/g, "''")}'`;
}

// Adapt the real SDK client to the narrow JobTriggerClient the bridge needs.
function adaptWorkspaceClient(ws: WorkspaceClient): JobTriggerClient {
  return {
    runNow: (req) => ws.jobs.runNow(req),
    getRun: (req) => ws.jobs.getRun(req),
    executeStatement: (req) => ws.statementExecution.executeStatement(req),
    getStatement: (req) => ws.statementExecution.getStatement(req),
    host: ws.config.host,
  };
}

// Read the run's real ApplyServiceResult JSON back from `agent_apply_results`. The
// job writes exactly one row per decision (keyed by proposal_id + the server-set
// decided_at); we take the most recent for that key. Returns the raw JSON string, or
// null when no row is present yet (the job SUCCEEDED but did not record — fail-closed).
async function readResultJson(
  client: JobTriggerClient,
  input: DecisionInput,
  warehouseId: string,
  catalog: string,
  schema: string,
  statementTimeoutMs: number,
  statementPollMs: number
): Promise<string | null> {
  const table = `\`${catalog}\`.\`${schema}\`.${APPLY_RESULTS_TABLE}`;
  const statement =
    `SELECT result_json FROM ${table} ` +
    `WHERE proposal_id = ${sqlLit(input.proposal_id)} AND decided_at = ${sqlLit(input.decided_at)} ` +
    `ORDER BY recorded_at DESC LIMIT 1`;

  let resp = await client.executeStatement({ warehouse_id: warehouseId, statement, wait_timeout: '50s' });
  const deadline = Date.now() + statementTimeoutMs;
  while (resp.status?.state === 'PENDING' || resp.status?.state === 'RUNNING') {
    if (Date.now() >= deadline) {
      throw new Error(`result read for proposal ${input.proposal_id} did not complete in ${statementTimeoutMs}ms`);
    }
    await sleep(statementPollMs);
    if (!resp.statement_id) break;
    resp = await client.getStatement({ statement_id: resp.statement_id });
  }
  if (resp.status?.state !== 'SUCCEEDED') {
    const detail = resp.status?.error?.message ? `: ${resp.status.error.message}` : '';
    throw new Error(`result read for proposal ${input.proposal_id} ${resp.status?.state ?? 'unknown'}${detail}`);
  }
  const rows = resp.result?.data_array ?? [];
  if (rows.length === 0 || rows[0].length === 0 || rows[0][0] == null) {
    return null;
  }
  return rows[0][0];
}

export function jobTriggerApplyBridge(options: JobTriggerBridgeOptions = {}): ApplyBridge {
  const parsedJobId = parseApplyJobId(process.env.AIL_APPLY_JOB_ID);
  const jobId = options.jobId ?? parsedJobId.jobId;
  const warehouseId = options.warehouseId ?? process.env.DATABRICKS_WAREHOUSE_ID;
  const catalog = options.catalog ?? process.env.AIL_APPLY_CATALOG ?? DEFAULT_APPLY_CATALOG;
  const schema = options.schema ?? process.env.AIL_APPLY_SCHEMA ?? DEFAULT_APPLY_SCHEMA;
  const timeoutMs = options.timeoutMs ?? (Number(process.env.AIL_APPLY_JOB_TIMEOUT_MS) || 300_000);
  const pollIntervalMs = options.pollIntervalMs ?? (Number(process.env.AIL_APPLY_JOB_POLL_MS) || 3_000);
  const statementTimeoutMs = options.statementTimeoutMs ?? 60_000;
  const statementPollMs = options.statementPollMs ?? 1_000;
  const clientFactory = options.clientFactory ?? (() => adaptWorkspaceClient(new WorkspaceClient({})));

  return async (input: DecisionInput): Promise<BridgeResult> => {
    if (options.jobId === undefined && parsedJobId.error) {
      throw new Error(parsedJobId.error);
    }
    if (jobId === undefined || Number.isNaN(jobId)) {
      throw new Error('AIL_APPLY_JOB_ID is not set — cannot trigger the apply job (deployed transport)');
    }
    if (!warehouseId) {
      throw new Error('DATABRICKS_WAREHOUSE_ID is not set — cannot read the apply-job result back');
    }
    const client = clientFactory();

    // Idempotency: derive a stable token from (proposal_id, decided_at). A retried
    // trigger of the SAME decision reuses the same run rather than launching a
    // second (defense-in-depth alongside the job's max_concurrent_runs:1 and the
    // engine refusing a non-pending proposal).
    const idempotencyToken = createHash('sha256')
      .update(`${input.proposal_id}:${input.decided_at}`)
      .digest('hex')
      .slice(0, 64);

    const started = await client.runNow({
      job_id: jobId,
      job_parameters: {
        proposal_id: input.proposal_id,
        agent_name: input.agent_name,
        decision: input.decision,
        approver: input.approver,
        reason: input.reason ?? '',
        decided_at: input.decided_at,
      },
      idempotency_token: idempotencyToken,
    });
    const runId = started.run_id;
    if (runId === undefined || runId === null) {
      throw new Error(`apply job ${jobId} trigger returned no run id`);
    }

    const deadline = Date.now() + timeoutMs;
    let run: RunLike;
    for (;;) {
      run = await client.getRun({ run_id: runId });
      const life = run.state?.life_cycle_state;
      if (life && TERMINAL_LIFECYCLE.has(life)) break;
      if (Date.now() >= deadline) {
        throw new Error(`apply job run ${runId} still ${life ?? 'PENDING'} after ${timeoutMs}ms — not applied`);
      }
      await sleep(pollIntervalMs);
    }

    // Terminal but not a clean success => honest error; never a fabricated apply.
    if (run.state?.life_cycle_state !== 'TERMINATED' || run.state?.result_state !== 'SUCCESS') {
      const msg = run.state?.state_message ? `: ${run.state.state_message}` : '';
      throw new Error(
        `apply job run ${runId} ended ${run.state?.life_cycle_state}/${run.state?.result_state ?? '—'}${msg}`
      );
    }

    // SUCCESS: retrieve the engine's REAL result out-of-band and return it verbatim.
    const json = await readResultJson(client, input, warehouseId, catalog, schema, statementTimeoutMs, statementPollMs);
    if (json === null) {
      throw new Error(
        `apply job run ${runId} SUCCEEDED but wrote no result row for proposal ${input.proposal_id} — refusing to fabricate an outcome`
      );
    }
    try {
      return JSON.parse(json) as BridgeResult;
    } catch {
      throw new Error(`apply job run ${runId} result row was unparseable JSON: ${json.slice(0, 500)}`);
    }
  };
}

// ---------------------------------------------------------------------------
// Transport selection — env-driven, so the SAME route + engine work in both images.
// ---------------------------------------------------------------------------
//
// Env contract:
//   AIL_APPLY_TRANSPORT=job  (or) AIL_APPLY_JOB_ID set  -> Databricks Job trigger
//   otherwise                                            -> local subprocess (default)
export type ApplyTransport = 'job' | 'subprocess';

export function resolveApplyTransport(env: NodeJS.ProcessEnv = process.env): ApplyTransport {
  if (env.AIL_APPLY_TRANSPORT === 'job' || (env.AIL_APPLY_JOB_ID && env.AIL_APPLY_JOB_ID.trim())) {
    return 'job';
  }
  return 'subprocess';
}

export function selectApplyBridge(env: NodeJS.ProcessEnv = process.env): ApplyBridge {
  return resolveApplyTransport(env) === 'job' ? jobTriggerApplyBridge() : spawnPythonApplyBridge();
}
