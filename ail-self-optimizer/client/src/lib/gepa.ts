export const GEPA_JOB_KEY = 'gepa';
export const GEPA_SUITE_VERSION = 'phase2-mini';
export const GEPA_REFLECTION_LM = 'databricks:/databricks-claude-sonnet-4-6';
export const GEPA_POLL_INTERVAL_MS = 10_000;

export interface GepaDispatchInput {
  agentName: string;
  experimentId: string;
  maxMetricCalls: number;
  holdoutFraction: number;
  maxTrainTasks: number;
}

export interface GepaRunState {
  life_cycle_state?: string;
  result_state?: string;
  state_message?: string;
}

export interface GepaRun {
  run_id?: number;
  run_name?: string;
  run_page_url?: string;
  state?: GepaRunState;
  start_time?: number;
  end_time?: number;
}

export interface GepaCandidateResult {
  schema_version: 'ail.jobs.gepa_result/v1';
  agent_name: string;
  subject_experiment_id: string;
  reviewer_experiment_id: string;
  mlflow_run_id: string;
  mlflow_run_url: string | null;
  artifact_path: string;
  artifact_uri: string;
  optimizer: string;
  proposal_id: string | null;
  proposal_status: string | null;
  proposal_created: boolean;
  proposal_reason: string;
  candidate_changed: boolean;
  candidate_promoted: false;
  human_gate_required: true;
  suite_version: string;
  suite_content_hash: string;
  max_metric_calls: number;
  gepa_total_metric_calls: number | null;
  gepa_num_candidates: number | null;
  gepa_best_val_score: number | null;
  holdout_savings_delta_pct: number | null;
  holdout_evolved_savings_pct: number | null;
  holdout_seed_savings_pct: number | null;
}

export interface GepaOutputResponse {
  result: GepaCandidateResult | null;
  logs_truncated: boolean;
  task_error: string | null;
}

interface GepaStartResponse {
  runId: number;
}

type Fetcher = typeof fetch;

export function isGepaSupportedAgent(agentName: string): boolean {
  return agentName === 'claude_code';
}

export function isTerminalGepaRun(run: GepaRun | null): boolean {
  const state = run?.state?.life_cycle_state;
  return state === 'TERMINATED' || state === 'SKIPPED' || state === 'INTERNAL_ERROR';
}

export function isSuccessfulGepaRun(run: GepaRun | null): boolean {
  return run?.state?.life_cycle_state === 'TERMINATED' && run.state.result_state === 'SUCCESS';
}

export function gepaRunLabel(run: GepaRun | null): string {
  const life = run?.state?.life_cycle_state;
  const result = run?.state?.result_state;
  if (!life) return 'DISPATCHED';
  return result ? `${life} · ${result}` : life;
}

function idempotencyToken(agentName: string): string {
  const safeAgent = agentName.replace(/[^a-zA-Z0-9_-]/g, '-').slice(0, 20) || 'agent';
  const nonce = globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  return `gepa-${safeAgent}-${nonce}`.slice(0, 64);
}

async function jsonResponse<T>(response: Response): Promise<T> {
  const body: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const message =
      typeof body === 'object' && body !== null && 'error' in body && typeof body.error === 'string'
        ? body.error
        : `Request failed (${response.status})`;
    throw new Error(message);
  }
  return body as T;
}

export async function dispatchGepaRun(input: GepaDispatchInput, fetcher: Fetcher = fetch): Promise<number> {
  const response = await fetcher(`/api/jobs/${GEPA_JOB_KEY}/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      params: {
        job_parameters: {
          agent_name: input.agentName,
          experiment_id: input.experimentId,
          suite_version: GEPA_SUITE_VERSION,
          max_metric_calls: String(input.maxMetricCalls),
          holdout_fraction: String(input.holdoutFraction),
          max_train_tasks: String(input.maxTrainTasks),
          reflection_lm: GEPA_REFLECTION_LM,
          seed: '0',
          confirmed_costly_run: 'true',
        },
        idempotency_token: idempotencyToken(input.agentName),
      },
    }),
  });
  const body = await jsonResponse<GepaStartResponse>(response);
  if (!Number.isSafeInteger(body.runId) || body.runId <= 0) {
    throw new Error('GEPA dispatcher returned no valid run id');
  }
  return body.runId;
}

export async function fetchGepaRun(runId: number, fetcher: Fetcher = fetch): Promise<GepaRun> {
  const response = await fetcher(`/api/jobs/${GEPA_JOB_KEY}/runs/${runId}`);
  return jsonResponse<GepaRun>(response);
}

export async function fetchGepaOutput(runId: number, fetcher: Fetcher = fetch): Promise<GepaOutputResponse> {
  const response = await fetcher(`/api/gepa/runs/${runId}/output`);
  return jsonResponse<GepaOutputResponse>(response);
}
