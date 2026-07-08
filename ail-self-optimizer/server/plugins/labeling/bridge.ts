import { spawn } from 'node:child_process';
import path from 'node:path';
import { WorkspaceClient } from '@databricks/sdk-experimental';

// The JSON action the authenticated labeling route hands the (Python) engine.
// `actor` is set SERVER-SIDE by the route (from the authenticated request) — it is
// never trusted from the browser (mirrors the approvals write-path's `approver` and
// the onboarding write-path's `actor`). The engine uses it as the label's HUMAN
// source; any `labeler` in the browser payload is ignored.
export interface LabelingAction {
  action: 'dimensions' | 'label';
  actor: string;
  experiment_id: string;
  // `label` only:
  trace_id?: string;
  name?: string; // the dimension == the registered judge name (the alignment key)
  value?: unknown; // 1–5 / 'pass'|'fail' / free-form — the human's grade
  rationale?: string;
}

// The JSON ail.labeling.service prints — a typed labeling result. Kept open
// (outcome + passthrough fields) so the route returns it verbatim and the client
// renders it (never re-deriving the registered judges or the label floor in TS).
export interface LabelingResult {
  outcome: string;
  [key: string]: unknown;
}

// The seam the route calls to run the (Python) labeling engine server-side.
// Injectable so the route is unit-testable with a fake bridge (no subprocess, no
// live write) — exactly as the onboarding/apply bridges are.
export type LabelingBridge = (input: LabelingAction) => Promise<LabelingResult>;

interface SpawnBridgeOptions {
  /** Python interpreter that has the `ail` package importable. */
  pythonBin?: string;
  /** Extra PYTHONPATH entry so `python -m ail.labeling.service` resolves in dev. */
  srcPath?: string;
  /** Hard timeout for the subprocess (ms). */
  timeoutMs?: number;
}

// The default bridge: run `python -m ail.labeling.service`, write the JSON action on
// stdin, read the typed result JSON from stdout. The engine runs under the app's
// identity (the framework service principal / the dev profile); it performs the
// permission-sensitive MLflow read/write and returns an HONEST result — a trace is
// only "labeled" when mlflow.log_feedback succeeded (see src/ail/labeling/service.py).
// This is grounded, not guessed: it is the SAME transport the onboarding plugin uses
// for its MLflow writes (a Python subprocess over the MLflow SDK), reusing the L1/L2
// helpers so the sacred name-match write is never reimplemented in TypeScript.
//
// Deployment note (mirrors the onboarding/approvals bridges): the deployed Databricks
// App image is Node-only — the `ail` wheel runs as serverless Jobs — so this
// subprocess bridge is the local-dev / self-hosted transport. The Node-only deployed
// transport is now BUILT as `restLabelingBridge` below (Node-native MLflow assessments
// REST, NOT a per-grade Job trigger — labeling is rapid-fire and job-startup latency
// per grade would be unusable); `selectLabelingBridge` picks between them by env. The
// engine's write shape, the authenticated route, and the client are unchanged — only
// the transport differs.
export function spawnPythonLabelingBridge(options: SpawnBridgeOptions = {}): LabelingBridge {
  const pythonBin =
    options.pythonBin ??
    process.env.AIL_LABELING_PYTHON_BIN ??
    process.env.AIL_ONBOARDING_PYTHON_BIN ??
    process.env.AIL_APPLY_PYTHON_BIN ??
    'python3';
  // The app runs from ail-self-optimizer/; the ail package source is ../src.
  const srcPath =
    options.srcPath ??
    process.env.AIL_LABELING_SRC_PATH ??
    process.env.AIL_ONBOARDING_SRC_PATH ??
    process.env.AIL_APPLY_SRC_PATH ??
    path.resolve(process.cwd(), '..', 'src');
  const timeoutMs = options.timeoutMs ?? 120_000;

  return (input: LabelingAction) =>
    new Promise<LabelingResult>((resolve, reject) => {
      const existing = process.env.PYTHONPATH;
      const child = spawn(pythonBin, ['-m', 'ail.labeling.service'], {
        env: { ...process.env, PYTHONPATH: existing ? `${srcPath}${path.delimiter}${existing}` : srcPath },
      });

      let stdout = '';
      let stderr = '';
      const timer = setTimeout(() => {
        child.kill('SIGKILL');
        reject(new Error(`labeling-service timed out after ${timeoutMs}ms`));
      }, timeoutMs);

      // Handle stdin stream errors: a child that exits WITHOUT reading its stdin makes
      // the write below emit EPIPE on this stream. Without a handler that EPIPE
      // surfaces as an UNHANDLED exception AFTER the tests finish — vitest catches it
      // as an uncaught error and fails the gate even though every test passed (a timing
      // race that passes locally, fails in CI). The child's exit code + stdout are the
      // real signal (via 'close'/'error'), so a broken input pipe is safely swallowed.
      // (Same guard the onboarding subprocess bridge needs for its stub-interpreter tests.)
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
          reject(new Error(`labeling-service exited ${code}: ${stderr.trim() || stdout.trim()}`));
          return;
        }
        try {
          resolve(JSON.parse(stdout) as LabelingResult);
        } catch {
          reject(new Error(`labeling-service returned unparseable output: ${stdout.slice(0, 500)}`));
        }
      });

      // Guard the write itself: if the pipe is already torn down the synchronous call
      // can throw. Swallow it — the 'close'/'error' handlers settle the promise from the
      // child's real exit code / output, never a write-after-close.
      try {
        child.stdin.write(JSON.stringify(input));
        child.stdin.end();
      } catch {
        // stdin already closed; the outcome comes from the child's exit + stdout.
      }
    });
}

// ---------------------------------------------------------------------------
// Node-native REST bridge — the DEPLOYED (Node-only) transport.
// ---------------------------------------------------------------------------
//
// The deployed Databricks App image is Node-only: the `ail` wheel is not importable
// there (docs/DEPLOY.md §7, docs/LABELING_UI.md), so `spawnPythonLabelingBridge`
// cannot run. This bridge is the same `LabelingBridge` seam over a Node-native
// transport: it talks to the Databricks-managed MLflow REST API directly (via the
// `@databricks/sdk-experimental` WorkspaceClient — the app's ambient service-principal
// auth, exactly as the approvals `jobTriggerApplyBridge` builds its client). Unlike
// approvals it does NOT trigger a Databricks Job per action: labeling is rapid-fire
// (one write per grade) and per-grade job-startup latency would be unusable.
//
// The three REST calls are GROUNDED against the live workspace (profile dais-demo),
// not guessed — the exact endpoints, request bodies, and responses were captured from
// the MLflow 3.14 Python SDK and confirmed end-to-end (create → read-back → delete):
//
//   * list judges  — GET /api/2.0/managed-evals/scheduled-scorers/{experiment_id}
//                     → { scheduled_scorers: { scorers: [{ name }] } }. This is what
//                     mlflow.genai.scorers.list_scorers (which ail.judges.registration
//                     .list_registered_scorers wraps) resolves to on a Databricks
//                     backend, so the registered-judge set matches the engine's.
//   * scan traces  — POST /api/4.0/mlflow/traces/search-long-running (async) then poll
//                     GET /api/4.0/mlflow/traces/search/operations/{id}; the operation
//                     response carries `trace_infos[]` with their `assessments` inline.
//   * write label  — POST /api/4.0/mlflow/traces/{location}/{trace_id}/assessments
//                     with { assessment_name, trace_id, source:{source_type:"HUMAN",
//                     source_id}, feedback:{value}, rationale } — the wire-equivalent
//                     of ail.judges.labeling.record_label's mlflow.log_feedback. The
//                     response's `assessment_id` is the ONLY proof of a real write.
//
// Fail-closed / no fabrication (HARD requirement): a label is `labeled` ONLY when the
// write response returns an `assessment_id`; a missing warehouse, an unresolvable
// trace location, a scorer-list failure, an unknown judge name, a write error, or a
// write with no returned id all yield an honest `refused`/`error` result — never a
// fabricated `labeled`. When a dependency cannot be confirmed the message tells the
// user to fall back to the MLflow Traces UI. The written assessment's name equals the
// judge/dimension name and its source is HUMAN with the AUTHENTICATED labeler as
// source_id (`actor`, set by the route from x-forwarded-* headers — never the body).

//: How many recent traces the read side scans (mirrors ail.labeling.service
//: DEFAULT_SCAN_LIMIT — a pagination bound, not a readiness threshold).
const DEFAULT_SCAN_LIMIT = 200;
//: How many "needs a label" traces the worklist returns (mirrors DEFAULT_WORKLIST_LIMIT).
const DEFAULT_WORKLIST_LIMIT = 50;

const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

// The narrow slice of the MLflow REST API the bridge uses — narrow on purpose so a
// FAKE client is trivial to inject in tests (no live workspace, no HTTP), exactly as
// the approvals `JobTriggerClient` is. The real client (`adaptWorkspaceClient`) closes
// over the WorkspaceClient + the SQL warehouse id and hides the async search-poll flow.
interface RestScorer {
  name?: string;
}
interface RestAssessmentSource {
  source_type?: string;
  source_id?: string;
}
interface RestAssessment {
  assessment_name?: string;
  source?: RestAssessmentSource;
}
interface RestTraceLocation {
  uc_table_prefix?: { catalog_name?: string; schema_name?: string; table_prefix?: string };
}
interface RestTraceInfo {
  trace_id?: string;
  trace_location?: RestTraceLocation;
  request_preview?: string;
  request_time?: string;
  assessments?: RestAssessment[];
}
export interface LabelingRestClient {
  /** Registered judge names for the experiment (the name-match set + the dimensions). */
  listScorers(experimentId: string): Promise<RestScorer[]>;
  /** Most-recent trace infos (with inline assessments) for the experiment, newest first. */
  searchTraces(experimentId: string, maxResults: number): Promise<RestTraceInfo[]>;
  /** Create one HUMAN assessment on a trace; returns the created id (write confirmation). */
  createAssessment(location: string, traceId: string, body: AssessmentWriteBody): Promise<{ assessment_id?: string }>;
}

// The v4 assessments write body — the wire-equivalent of record_label's log_feedback.
interface AssessmentWriteBody {
  assessment_name: string;
  trace_id: string;
  source: { source_type: 'HUMAN'; source_id: string };
  feedback: { value: unknown };
  rationale?: string;
}

export interface RestBridgeOptions {
  /** SQL warehouse id for the v4 UC tracing endpoints (env: DATABRICKS_WAREHOUSE_ID). */
  warehouseId?: string;
  /**
   * The label floor to surface (env: AIL_LABEL_FLOOR). Two-tier: this is a RELAY of the
   * Python `ail.readiness.ReadinessThresholds.quality_min_labels`, NEVER a number
   * authored here. When unset the result omits it and the client renders a neutral
   * `—` (see client/src/lib/labeling.ts) — an honest missing value, never a fabricated
   * floor. Deployers may set it from the engine (`python -c "import ail.readiness as r;
   * print(r.ReadinessThresholds().quality_min_labels)"`) to light up the target.
   */
  labelFloor?: number;
  /** Recent traces to scan for progress + worklist (mirrors the engine scan limit). */
  scanLimit?: number;
  /** Max "needs a label" traces returned in the worklist (mirrors the engine limit). */
  worklistLimit?: number;
  /** Bound on polling the async trace search to a terminal state, ms. */
  searchTimeoutMs?: number;
  /** Interval between search-operation polls, ms. */
  searchPollMs?: number;
  /** Injectable client factory (tests pass a fake; default builds a WorkspaceClient). */
  clientFactory?: () => LabelingRestClient;
}

function parseLabelFloor(raw: string | undefined): number | undefined {
  if (raw === undefined) return undefined;
  const trimmed = raw.trim();
  if (!trimmed) return undefined;
  const n = Number(trimmed);
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : undefined;
}

// Parse a v4 (UC-stored) MLflow trace id `trace:/<location>/<trace_id>` into its parts —
// the same split MLflow's `parse_trace_id_v4` does. The v4 assessments endpoint needs
// the location and the bare id in the URL path. Returns null for a non-v4 id (so the
// write fails closed rather than guess a v3 endpoint that was not grounded here).
function parseV4TraceId(traceId: string): { location: string; traceId: string } | null {
  const prefix = 'trace:/';
  if (!traceId.startsWith(prefix)) return null;
  const rest = traceId.slice(prefix.length);
  const slash = rest.indexOf('/');
  if (slash <= 0) return null;
  const location = rest.slice(0, slash);
  const bare = rest.slice(slash + 1);
  if (!location || !bare) return null;
  return { location, traceId: bare };
}

// The UC location string (`catalog.schema.table_prefix`) from a trace_info's
// trace_location, so the worklist can hand back the full v4 id the write needs.
function locationOf(loc: RestTraceLocation | undefined): string | null {
  const uc = loc?.uc_table_prefix;
  if (!uc?.catalog_name || !uc.schema_name || !uc.table_prefix) return null;
  return `${uc.catalog_name}.${uc.schema_name}.${uc.table_prefix}`;
}

// The judge names a trace already carries a HUMAN assessment for — the exact read
// convention ail.labeling.service._human_labeled_names uses (source_type == HUMAN,
// keyed by name). Used to compute per-dimension progress and the worklist; never to
// fabricate a label.
function humanLabeledNames(assessments: RestAssessment[] | undefined): Set<string> {
  const names = new Set<string>();
  for (const a of assessments ?? []) {
    if (a.source?.source_type === 'HUMAN' && a.assessment_name) names.add(a.assessment_name);
  }
  return names;
}

function isEmptyValue(value: unknown): boolean {
  return value === undefined || value === null || (typeof value === 'string' && !value.trim());
}

function cleanValue(value: unknown): unknown {
  return typeof value === 'string' ? value.trim() : value;
}

function previewOf(text: string | undefined): string | null {
  if (text === undefined || text === null) return null;
  const s = String(text);
  return s.length <= 240 ? s : `${s.slice(0, 237)}…`;
}

const errText = (err: unknown): string => (err instanceof Error ? err.message : String(err));

// Resolve the registered judge names, de-duped and non-empty, exactly as the engine's
// list_registered_scorers does. Throws when the set cannot be determined so the caller
// fails closed (never invents dimensions / never writes an unconfirmable name).
async function listJudgeNames(client: LabelingRestClient, experimentId: string): Promise<string[]> {
  const scorers = await client.listScorers(experimentId);
  const names: string[] = [];
  for (const s of scorers) {
    if (s.name && !names.includes(s.name)) names.push(s.name);
  }
  return names;
}

// The read side: registered judged dimensions + per-dimension progress + the "needs a
// label" worklist — the DimensionsResult shape ail.labeling.service prints and
// client/src/lib/labeling.ts renders. Fail-closed: an unknown judge set or an
// unreadable trace scan is an honest `error`, never invented dimensions.
async function runRestDimensions(
  client: LabelingRestClient,
  input: LabelingAction,
  cfg: { labelFloor: number | undefined; scanLimit: number; worklistLimit: number }
): Promise<LabelingResult> {
  const exp = input.experiment_id.trim();
  if (!exp) return { outcome: 'error', error: 'an experiment id is required' };

  let judges: string[];
  try {
    judges = await listJudgeNames(client, exp);
  } catch (err) {
    return {
      outcome: 'error',
      error:
        `cannot determine the registered judges for this experiment (${errText(err)}); refusing to ` +
        'invent labeling dimensions. Register at least one judge (ail.judges authoring) and ensure ' +
        'the app can list scorers, or label in the MLflow Traces UI.',
    };
  }

  let traces: RestTraceInfo[];
  try {
    traces = await client.searchTraces(exp, cfg.scanLimit);
  } catch (err) {
    return {
      outcome: 'error',
      error: `cannot read this experiment's traces to build the labeling worklist (${errText(err)}). Use the MLflow Traces UI to label.`,
    };
  }

  const counts = new Map<string, number>(judges.map((n) => [n, 0]));
  const worklist: unknown[] = [];
  let scanned = 0;
  for (const t of traces) {
    scanned += 1;
    const bare = t.trace_id;
    const location = locationOf(t.trace_location);
    if (!bare || !location) continue;
    const human = humanLabeledNames(t.assessments);
    const labeled: Record<string, boolean> = {};
    for (const name of judges) {
      const has = human.has(name);
      labeled[name] = has;
      if (has) counts.set(name, (counts.get(name) ?? 0) + 1);
    }
    if (judges.length > 0 && !judges.every((n) => labeled[n]) && worklist.length < cfg.worklistLimit) {
      worklist.push({
        trace_id: `trace:/${location}/${bare}`,
        request_time: t.request_time ?? null,
        preview: previewOf(t.request_preview),
        labeled,
      });
    }
  }

  const dimensions = judges.map((name) => {
    const soFar = counts.get(name) ?? 0;
    // The value control (numeric/pass-fail hint) degrades to a free-form field on this
    // transport — the engine/local transport reads it from the judge's L1 label schema;
    // the client falls back to free text when `input` is null (never blocks labeling).
    const dim: Record<string, unknown> = {
      name,
      labels_so_far: soFar,
      input: null,
      // Neutral summary — deliberately carries NO floor number (two-tier: a floor number
      // is never authored in TS). The client renders the `N / floor` progress from the
      // numbers below via progressLabel, using the relayed floor or a neutral `—`.
      summary: `Human labels named for the ${name} judge — MemAlign pairs your labels to it by this name.`,
    };
    if (cfg.labelFloor !== undefined) {
      dim.label_floor = cfg.labelFloor;
      dim.remaining = Math.max(0, cfg.labelFloor - soFar);
      dim.complete = cfg.labelFloor - soFar <= 0;
    } else {
      dim.complete = false;
    }
    return dim;
  });

  const result: LabelingResult = {
    outcome: 'dimensions',
    experiment_id: exp,
    dimensions,
    traces: worklist,
    scanned,
    scan_capped: scanned >= cfg.scanLimit,
    actor: input.actor,
    summary:
      judges.length === 0
        ? ''
        : `Label traces along the ${judges.length} registered judged ` +
          `${judges.length === 1 ? 'dimension' : 'dimensions'}. Your labels are written as HUMAN ` +
          'assessments named for the judge, which is what MemAlign pairs them by.',
  };
  if (cfg.labelFloor !== undefined) result.label_floor = cfg.labelFloor;
  return result;
}

// The write side: record ONE HUMAN label, name-matched to a registered judge, or refuse.
// Fail-closed at every step; a label is `labeled` ONLY when the REST write confirms an
// assessment_id. The labeler is the AUTHENTICATED `actor` the route injected.
async function runRestLabel(client: LabelingRestClient, input: LabelingAction): Promise<LabelingResult> {
  const experiment_id = input.experiment_id;
  const name = (input.name ?? '').trim();
  const traceId = (input.trace_id ?? '').trim();
  const labeler = (input.actor ?? '').trim();
  const value = input.value;

  const refused = (reason: string): LabelingResult => ({
    outcome: 'refused',
    experiment_id,
    trace_id: traceId,
    name,
    value,
    labeler,
    refused_reason: reason,
  });
  const errored = (error: string): LabelingResult => ({
    outcome: 'error',
    experiment_id,
    trace_id: traceId,
    name,
    value,
    labeler,
    error,
  });

  // Fail-closed pre-checks (the route validates too; re-checked here so the engine and
  // this transport agree, and so a direct bridge call cannot skip them).
  if (!labeler) return refused('refusing an anonymous label — no authenticated labeler identity');
  if (!traceId) return refused('a trace id is required');
  if (!name) return refused('a dimension name is required');
  if (isEmptyValue(value)) return refused('a label value is required');

  // Name-match guard (the load-bearing invariant): the label name MUST be a registered
  // judge, or it could never align. If the set cannot be confirmed, refuse to write.
  let judges: string[];
  try {
    judges = await listJudgeNames(client, experiment_id);
  } catch (err) {
    return errored(
      `cannot determine the registered judges to validate the label name (${errText(err)}); ` +
        'refusing to write a label whose name cannot be confirmed to match a registered judge. ' +
        'Use the MLflow Traces UI.'
    );
  }
  if (!judges.includes(name)) {
    return refused(
      `${JSON.stringify(name)} is not a registered judge — refusing to write a label that could ` +
        'never align (a label must be named for a registered judge).'
    );
  }

  // The v4 assessments endpoint needs the UC location + bare id from the full v4 trace id.
  const parsed = parseV4TraceId(traceId);
  if (parsed === null) {
    return errored(
      `cannot resolve the trace location for ${JSON.stringify(traceId)} — the deployed labeling ` +
        'write needs a UC (v4) trace id (trace:/<catalog.schema.prefix>/<id>). Label this trace in ' +
        'the MLflow Traces UI instead.'
    );
  }

  const body: AssessmentWriteBody = {
    assessment_name: name,
    trace_id: traceId,
    source: { source_type: 'HUMAN', source_id: labeler },
    feedback: { value: cleanValue(value) },
    ...((input.rationale ?? '').trim() ? { rationale: (input.rationale ?? '').trim() } : {}),
  };

  let created: { assessment_id?: string };
  try {
    created = await client.createAssessment(parsed.location, parsed.traceId, body);
  } catch (err) {
    // Any write failure (auth, permission, trace not found) is an honest error — never a fake label.
    return errored(`${errText(err)}`);
  }
  if (!created || !created.assessment_id) {
    return errored(
      'the assessments write returned no assessment id — not confirming a label (never fabricating ' +
        'success). Verify in the MLflow Traces UI.'
    );
  }

  // Real write confirmed by the returned assessment_id. Return immediately — no
  // per-grade progress re-read: that would add a second (async) trace scan to every
  // rapid-fire grade, the exact per-write latency this transport exists to avoid. The
  // panel refetches `dimensions` after each label (LabelingPanel `reloadKey`), so the
  // progress cards update from a single scan there; the client renders the inline count
  // as a neutral `—` when the write result omits it (never a fabricated number).
  return {
    outcome: 'labeled',
    experiment_id,
    trace_id: traceId,
    name,
    value: cleanValue(value),
    labeler,
  };
}

// Adapt the real SDK WorkspaceClient to the narrow LabelingRestClient. All three calls
// go through the authenticated `apiClient.request` (the app's ambient SP creds), mirror
// how the approvals bridge builds + uses its WorkspaceClient. The 2-step async trace
// search (start → poll the operation to `done`) is hidden here so the bridge logic — and
// its tests — deal only in the resolved trace infos.
function adaptWorkspaceClient(
  ws: WorkspaceClient,
  cfg: { warehouseId: string; searchTimeoutMs: number; searchPollMs: number }
): LabelingRestClient {
  const jsonHeaders = (): Headers => new Headers({ 'Content-Type': 'application/json', Accept: 'application/json' });
  const getHeaders = (): Headers => new Headers({ Accept: 'application/json' });
  return {
    async listScorers(experimentId) {
      const resp = (await ws.apiClient.request({
        path: `/api/2.0/managed-evals/scheduled-scorers/${experimentId}`,
        method: 'GET',
        headers: getHeaders(),
        raw: false,
      })) as { scheduled_scorers?: { scorers?: RestScorer[] } };
      return resp.scheduled_scorers?.scorers ?? [];
    },
    async searchTraces(experimentId, maxResults) {
      const start = (await ws.apiClient.request({
        path: '/api/4.0/mlflow/traces/search-long-running',
        method: 'POST',
        headers: jsonHeaders(),
        raw: false,
        payload: {
          locations: [{ type: 'MLFLOW_EXPERIMENT', mlflow_experiment: { experiment_id: experimentId } }],
          max_results: maxResults,
          sql_warehouse_id: cfg.warehouseId,
          order_by: ['timestamp_ms DESC'],
        },
      })) as { name?: string };
      const opId = start.name;
      if (!opId) throw new Error('trace search returned no operation id');
      const deadline = Date.now() + cfg.searchTimeoutMs;
      for (;;) {
        const op = (await ws.apiClient.request({
          path: `/api/4.0/mlflow/traces/search/operations/${opId}`,
          method: 'GET',
          headers: getHeaders(),
          raw: false,
          query: { sql_warehouse_id: cfg.warehouseId },
        })) as { done?: boolean; response?: { trace_infos?: RestTraceInfo[] } };
        if (op.done) return op.response?.trace_infos ?? [];
        if (Date.now() >= deadline) {
          throw new Error(`trace search did not complete in ${cfg.searchTimeoutMs}ms`);
        }
        await sleep(cfg.searchPollMs);
      }
    },
    async createAssessment(location, traceId, body) {
      return (await ws.apiClient.request({
        path: `/api/4.0/mlflow/traces/${location}/${traceId}/assessments`,
        method: 'POST',
        headers: jsonHeaders(),
        raw: false,
        query: { sql_warehouse_id: cfg.warehouseId },
        payload: body,
      })) as { assessment_id?: string };
    },
  };
}

export function restLabelingBridge(options: RestBridgeOptions = {}): LabelingBridge {
  const warehouseId = options.warehouseId ?? process.env.DATABRICKS_WAREHOUSE_ID;
  const labelFloor = options.labelFloor ?? parseLabelFloor(process.env.AIL_LABEL_FLOOR);
  const scanLimit = options.scanLimit ?? DEFAULT_SCAN_LIMIT;
  const worklistLimit = options.worklistLimit ?? DEFAULT_WORKLIST_LIMIT;
  const searchTimeoutMs = options.searchTimeoutMs ?? (Number(process.env.AIL_LABELING_SEARCH_TIMEOUT_MS) || 60_000);
  const searchPollMs = options.searchPollMs ?? (Number(process.env.AIL_LABELING_SEARCH_POLL_MS) || 1_000);
  const clientFactory =
    options.clientFactory ??
    (() => {
      if (!warehouseId) {
        throw new Error(
          'DATABRICKS_WAREHOUSE_ID is not set — the deployed labeling transport needs a SQL warehouse ' +
            'for the MLflow tracing API.'
        );
      }
      return adaptWorkspaceClient(new WorkspaceClient({}), { warehouseId, searchTimeoutMs, searchPollMs });
    });

  return async (input: LabelingAction): Promise<LabelingResult> => {
    // Fail-closed dependency check: the v4 UC tracing endpoints require a warehouse id.
    // Without one the transport is unavailable — say so honestly, never a fake success.
    if (!warehouseId) {
      const guidance =
        'deployed labeling unavailable — no SQL warehouse configured (DATABRICKS_WAREHOUSE_ID). ' +
        'Use the MLflow Traces UI to label.';
      return input.action === 'label'
        ? {
            outcome: 'error',
            experiment_id: input.experiment_id,
            trace_id: input.trace_id ?? '',
            name: input.name ?? '',
            value: input.value,
            labeler: input.actor,
            error: guidance,
          }
        : { outcome: 'error', error: guidance };
    }

    let client: LabelingRestClient;
    try {
      client = clientFactory();
    } catch (err) {
      const guidance = `deployed labeling unavailable — ${errText(err)} Use the MLflow Traces UI to label.`;
      return input.action === 'label'
        ? { outcome: 'error', experiment_id: input.experiment_id, trace_id: input.trace_id ?? '', name: input.name ?? '', value: input.value, labeler: input.actor, error: guidance }
        : { outcome: 'error', error: guidance };
    }

    if (input.action === 'dimensions') {
      return runRestDimensions(client, input, { labelFloor, scanLimit, worklistLimit });
    }
    if (input.action === 'label') {
      return runRestLabel(client, input);
    }
    return { outcome: 'error', error: `unknown labeling action ${JSON.stringify(input.action)}` };
  };
}

// ---------------------------------------------------------------------------
// Transport selection — env-driven, so the SAME route + client work in both images.
// ---------------------------------------------------------------------------
//
// Env contract (mirrors approvals resolveApplyTransport/selectApplyBridge):
//   AIL_LABELING_TRANSPORT=rest  -> Node-native MLflow REST (the deployed transport)
//   otherwise                    -> local Python subprocess (default; local dev / self-hosted)
export type LabelingTransport = 'rest' | 'subprocess';

export function resolveLabelingTransport(env: NodeJS.ProcessEnv = process.env): LabelingTransport {
  return env.AIL_LABELING_TRANSPORT === 'rest' ? 'rest' : 'subprocess';
}

// Transport selection. Kept as a function (not a bare export) so the deployed Node-only
// REST transport is selected by env here without touching the route, exactly as the
// onboarding/approvals bridges do. Local/dev stays on the subprocess transport.
export function selectLabelingBridge(): LabelingBridge {
  return resolveLabelingTransport() === 'rest' ? restLabelingBridge() : spawnPythonLabelingBridge();
}
