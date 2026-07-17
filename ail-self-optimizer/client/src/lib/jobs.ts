// Activity-page view logic, kept as pure functions so the run-state rendering,
// formatting, and outcome mapping are unit-testable without a DOM (mirrors
// lib/lineage.ts / lib/approvals.ts). The ActivityJobs component is a thin renderer
// over these. The load-bearing property of this page is HONESTY: every function here
// renders exactly what the SDK / the table returned and NEVER invents a run, a state,
// a duration, or an outcome.

// --- Server contract (mirrors server/plugins/jobs/bridge.ts) -----------------------
// Re-declared client-side (the client and server are separate tsconfig projects, so
// this mirrors how lib/approvals.ts re-declares the server's response shape).

export interface JobRunView {
  run_id?: number;
  run_name?: string;
  run_page_url?: string;
  /** RunState.life_cycle_state verbatim: RUNNING / TERMINATED / SKIPPED / ... */
  life_cycle_state?: string;
  /** RunState.result_state verbatim: SUCCESS / FAILED / CANCELED / ... (undefined until terminal) */
  result_state?: string;
  state_message?: string;
  start_time?: number;
  end_time?: number;
  run_duration?: number;
}

export type JobActivity =
  | { name: string; status: 'ok'; job_id: number; description?: string; runs: JobRunView[] }
  | { name: string; status: 'not_found' }
  | { name: string; status: 'error'; error: string };

export interface JobsActivityResult {
  jobs: JobActivity[];
  fatal_error?: string;
}

// One row of config/queries/recent_activity.sql. All columns are STRING in the
// source table; kept as strings and rendered verbatim (no recomputation).
export interface RecentActivityRow {
  proposal_id: string;
  agent_name: string;
  experiment_id: string;
  status: string;
  action_kind: string;
  risk_class: string;
  objective_metric: string;
  trigger_summary: string;
  created_at: string;
  generated_at: string;
}

// --- Run state (VERBATIM) ----------------------------------------------------------

// The run state as text, built ONLY from the two verbatim SDK fields. A terminal run
// shows "<life> · <result>" (e.g. "TERMINATED · SUCCESS"); a non-terminal run shows
// just its lifecycle (e.g. "RUNNING"). When the SDK returned no state at all we say
// "UNKNOWN" — we never guess a state.
export function runStateText(run: Pick<JobRunView, 'life_cycle_state' | 'result_state'>): string {
  const life = run.life_cycle_state?.trim();
  const result = run.result_state?.trim();
  if (life && result) return `${life} · ${result}`;
  if (life) return life;
  if (result) return result;
  return 'UNKNOWN';
}

export type RunTone = 'success' | 'error' | 'active' | 'neutral';

const NON_TERMINAL_LIFECYCLE = new Set(['PENDING', 'QUEUED', 'RUNNING', 'TERMINATING', 'BLOCKED', 'WAITING_FOR_RETRY']);

// A tone derived FAITHFULLY from the verbatim state — never a relabel. Only the two
// unambiguous terminal results are colored (SUCCESS → success, FAILED → error); a
// still-running lifecycle is 'active'; EVERYTHING else (an unrecognized or ambiguous
// result such as CANCELED / TIMEDOUT / SKIPPED / a state we don't enumerate) falls to
// 'neutral' so a state we can't be certain about is never dressed up as a success or
// a failure. The badge text always shows the state verbatim regardless of tone.
export function runTone(run: Pick<JobRunView, 'life_cycle_state' | 'result_state'>): RunTone {
  const result = run.result_state?.trim();
  if (result === 'SUCCESS') return 'success';
  if (result === 'FAILED') return 'error';
  const life = run.life_cycle_state?.trim();
  if (life && NON_TERMINAL_LIFECYCLE.has(life)) return 'active';
  return 'neutral';
}

// --- Formatting (honest, no fabrication) -------------------------------------------

// A UTC ISO timestamp from epoch milliseconds, or an em dash when the SDK didn't
// report the time (e.g. a run that hasn't ended). UTC is deterministic and avoids any
// locale ambiguity. Never fabricates a time.
export function fmtEpochMs(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms) || ms <= 0) return '—';
  return new Date(ms).toISOString().replace('T', ' ').replace('.000Z', 'Z');
}

// A human duration from the SDK's OWN run_duration field (milliseconds). We do not
// derive a duration from start/end — if the SDK didn't report a duration (e.g. a
// still-running run) we show an em dash rather than compute or guess one.
export function fmtDurationMs(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms) || ms <= 0) return '—';
  if (ms < 1_000) return `${Math.round(ms)} ms`;
  const seconds = ms / 1_000;
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  const mins = Math.floor(seconds / 60);
  const rem = Math.round(seconds - mins * 60);
  return `${mins}m ${rem}s`;
}

// --- Proposal/decision outcomes (recent_activity) ----------------------------------

// The proposal lifecycle states the controller/queue write (ail.loop.proposals
// ProposalStatus). Used only to pick a faithful tone; the status text is rendered
// verbatim from the table.
export type OutcomeTone = 'success' | 'active' | 'neutral';

export function outcomeTone(status: string | null | undefined): OutcomeTone {
  switch ((status ?? '').trim().toLowerCase()) {
    case 'applied':
      return 'success';
    case 'pending':
    case 'approved':
      return 'active';
    // rejected / superseded / anything unrecognized: a valid, non-success outcome —
    // shown neutrally, never as an error or a win.
    default:
      return 'neutral';
  }
}

// --- Un-instrumented optimizers (explicit "not tracked yet") ------------------------

// GEPA / RLM-HALO / MemAlign / asset-generation do NOT run as tracked Databricks jobs
// today — nothing records their runs — so the page states that honestly instead of
// showing a fake progress bar or a zero-filled row. This is the single source of that
// copy so the component stays a thin renderer.
export interface UntrackedOptimizer {
  key: string;
  name: string;
  detail: string;
}

export const UNTRACKED_OPTIMIZERS: readonly UntrackedOptimizer[] = [
  {
    key: 'gepa',
    name: 'GEPA prompt evolution',
    detail: 'Runs inside the controller, not as its own tracked job — no run ledger records it yet.',
  },
  {
    key: 'asset-gen',
    name: 'Asset generation',
    detail: 'Not wired to a tracked job — no run history to show.',
  },
];
