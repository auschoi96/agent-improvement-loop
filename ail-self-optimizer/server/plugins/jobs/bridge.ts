import { WorkspaceClient } from '@databricks/sdk-experimental';

// The framework's registered Databricks jobs, discovered BY NAME (never by a
// workspace-specific numeric job id — job ids differ per workspace; discovering by
// name keeps the app reusable). These are the ONLY two things that run as tracked
// jobs today (resources/apply_service.job.yml, resources/l0_publish.job.yml;
// docs/LOOP_CONTROLLER.md): `ail-apply-service` (on-demand, one run per human
// approve/reject decision) and `ail-l0-publish-scheduled` (scheduled L0 publish).
// GEPA / RLM-HALO / MemAlign / asset-generation do NOT run as tracked jobs yet, so
// they are intentionally absent here and surfaced client-side as "not tracked".
export const REGISTERED_JOB_NAMES = [
  'ail-apply-service',
  'ail-l0-publish-scheduled',
  'ail-continuous-rlm-scheduled',
  'ail-judge-backfill',
  'ail-auto-align',
  'ail-advisory-memory-distiller',
  'ail-onboarding-service',
] as const;

// Default number of recent runs to list per job. Bounded below 25 because the Jobs
// listRuns API requires 0 < limit < 25.
export const DEFAULT_RUN_LIMIT = 10;
const MAX_RUN_LIMIT = 20;

// One run, exactly as the SDK returned it. Every field is optional because the SDK
// omits them depending on lifecycle (a RUNNING run has no end_time / run_duration /
// result_state). These values are rendered VERBATIM by the client — nothing here is
// reinterpreted, defaulted, or relabelled.
export interface JobRunView {
  run_id?: number;
  run_name?: string;
  run_page_url?: string;
  /** RunState.life_cycle_state verbatim: RUNNING / TERMINATED / SKIPPED / ... */
  life_cycle_state?: string;
  /** RunState.result_state verbatim: SUCCESS / FAILED / CANCELED / ... (undefined until terminal) */
  result_state?: string;
  state_message?: string;
  /** epoch milliseconds */
  start_time?: number;
  /** epoch milliseconds */
  end_time?: number;
  /** total run duration in milliseconds, as reported by the SDK */
  run_duration?: number;
}

// A discovered job's identity — the numeric id resolved AT RUNTIME from the name.
export interface DiscoveredJob {
  job_id: number;
  name?: string;
  description?: string;
}

// Per-job section of the Activity page. Discriminated and fail-closed:
//   - 'ok'        → the job exists and its recent runs were listed (may be empty).
//   - 'not_found' → no job by that name in the workspace (e.g. not deployed here).
//   - 'error'     → the SDK call for THIS job failed (timeout, or the app SP lacks
//                   run-view permission). An honest error — NEVER a fabricated run.
export type JobActivity =
  | { name: string; status: 'ok'; job_id: number; description?: string; runs: JobRunView[] }
  | { name: string; status: 'not_found' }
  | { name: string; status: 'error'; error: string };

// The whole Activity job-runs payload. `fatal_error` is set only when the SDK client
// could not be built at all (misconfigured / unauthenticated); in that case every
// job section is also marked 'error' so the client renders a uniform honest state.
export interface JobsActivityResult {
  jobs: JobActivity[];
  fatal_error?: string;
}

// The narrow seam the orchestrator calls — injectable so tests drive it with a fake
// client (no live workspace). The real SDK client is adapted to this via
// `adaptWorkspaceClient`.
export interface JobsClient {
  /** Resolve a job by its exact name; null when no such job exists. */
  discoverJobByName(name: string): Promise<DiscoveredJob | null>;
  /** List up to `limit` most-recent runs of a job (read-only; never triggers a run). */
  recentRuns(jobId: number, limit: number): Promise<JobRunView[]>;
}

export type JobsActivityBridge = (limit?: number) => Promise<JobsActivityResult>;

function clampLimit(limit: number): number {
  if (!Number.isFinite(limit) || limit < 1) return DEFAULT_RUN_LIMIT;
  return Math.min(Math.floor(limit), MAX_RUN_LIMIT);
}

// Discover each registered job by name and list its recent runs. Each job is isolated
// in its own try/catch, so one job's permission error (or absence) never suppresses
// another job's real data. Fail-closed throughout: a lookup failure yields an honest
// 'error' section and a missing job yields 'not_found' — never a fabricated row.
export async function fetchJobsActivity(
  client: JobsClient,
  names: readonly string[],
  limit: number
): Promise<JobsActivityResult> {
  const jobs: JobActivity[] = [];
  for (const name of names) {
    try {
      const job = await client.discoverJobByName(name);
      if (!job) {
        jobs.push({ name, status: 'not_found' });
        continue;
      }
      const runs = await client.recentRuns(job.job_id, limit);
      jobs.push({ name, status: 'ok', job_id: job.job_id, description: job.description, runs });
    } catch (err) {
      jobs.push({ name, status: 'error', error: err instanceof Error ? err.message : 'job run lookup failed' });
    }
  }
  return { jobs };
}

// Adapt the real SDK WorkspaceClient to the narrow JobsClient. The AsyncIterable →
// bounded-array conversion and the BaseJob/BaseRun → view mapping live here; this is
// the only code path that touches the live SDK (not unit-tested, exactly as the
// approvals `adaptWorkspaceClient` isn't). Read-only: list + listRuns only.
function adaptWorkspaceClient(ws: WorkspaceClient): JobsClient {
  return {
    async discoverJobByName(name: string): Promise<DiscoveredJob | null> {
      // The SDK `name` filter is an exact case-insensitive match; we still verify the
      // returned name matches so we never surface a job we didn't ask for.
      for await (const job of ws.jobs.list({ name, limit: 20, expand_tasks: false })) {
        if (job.job_id != null && (job.settings?.name ?? '').toLowerCase() === name.toLowerCase()) {
          return { job_id: job.job_id, name: job.settings?.name, description: job.settings?.description };
        }
      }
      return null;
    },
    async recentRuns(jobId: number, limit: number): Promise<JobRunView[]> {
      const runs: JobRunView[] = [];
      for await (const r of ws.jobs.listRuns({ job_id: jobId, limit, expand_tasks: false })) {
        runs.push({
          run_id: r.run_id,
          run_name: r.run_name,
          run_page_url: r.run_page_url,
          life_cycle_state: r.state?.life_cycle_state,
          result_state: r.state?.result_state,
          state_message: r.state?.state_message,
          start_time: r.start_time,
          end_time: r.end_time,
          run_duration: r.run_duration,
        });
        if (runs.length >= limit) break;
      }
      return runs;
    },
  };
}

export interface JobsActivityBridgeOptions {
  /** Injectable client factory (tests pass a fake; default builds a WorkspaceClient). */
  clientFactory?: () => JobsClient;
  /** Job names to surface (defaults to the framework's REGISTERED_JOB_NAMES). */
  names?: readonly string[];
}

// The read-only Activity bridge. Runs under the app's identity (the deployed
// framework service principal, or the dev profile) via the SDK default auth chain —
// mirrors how the approvals bridge builds its WorkspaceClient. If the client cannot
// be built at all, every section is marked 'error' with that reason (fail-closed) so
// the page renders an honest unavailable state rather than nothing.
export function jobsActivityBridge(options: JobsActivityBridgeOptions = {}): JobsActivityBridge {
  const names = options.names ?? REGISTERED_JOB_NAMES;
  const clientFactory = options.clientFactory ?? (() => adaptWorkspaceClient(new WorkspaceClient({})));
  return async (limit: number = DEFAULT_RUN_LIMIT): Promise<JobsActivityResult> => {
    let client: JobsClient;
    try {
      client = clientFactory();
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'could not initialize the Databricks client';
      return { jobs: names.map((name) => ({ name, status: 'error' as const, error: msg })), fatal_error: msg };
    }
    return fetchJobsActivity(client, names, clampLimit(limit));
  };
}

export function selectJobsActivityBridge(): JobsActivityBridge {
  return jobsActivityBridge();
}
