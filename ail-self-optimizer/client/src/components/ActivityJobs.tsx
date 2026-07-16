import { useEffect, useState } from 'react';
import { useMemo } from 'react';
import { sql } from '@databricks/appkit-ui/js';
import {
  Badge,
  Button,
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Separator,
  Skeleton,
  useAnalyticsQuery,
} from '@databricks/appkit-ui/react';
import {
  fmtDurationMs,
  fmtEpochMs,
  outcomeTone,
  runStateText,
  runTone,
  UNTRACKED_OPTIMIZERS,
  type JobActivity,
  type JobRunView,
  type JobsActivityResult,
  type RecentActivityRow,
  type RunTone,
  type OutcomeTone,
} from '../lib/jobs';

// The read-only Activity endpoint (server/plugins/jobs). Lists recent runs of the
// framework's registered jobs via the SDK. GET only — never triggers a run.
const ACTIVITY_ENDPOINT = '/api/jobactivity/activity';

type FetchStatus = 'loading' | 'loaded' | 'error';

const RUN_TONE_CLASS: Record<RunTone, string> = {
  success: 'border-emerald-500 text-emerald-700 dark:text-emerald-300',
  error: 'border-destructive text-destructive',
  active: 'border-primary text-primary',
  neutral: 'text-muted-foreground',
};

const OUTCOME_TONE_CLASS: Record<OutcomeTone, string> = {
  success: 'border-emerald-500 text-emerald-700 dark:text-emerald-300',
  active: 'border-primary text-primary',
  neutral: 'text-muted-foreground',
};

// The Activity / jobs-progress page. Three honest sections:
//   1. Registered-job run history — REAL runs of ail-apply-service and
//      ail-l0-publish-scheduled from the SDK (fail-closed per job).
//   2. Recent proposal/decision outcomes — a SELECT-only view of agent_proposed_actions.
//   3. Un-instrumented optimizers — an explicit "not tracked as jobs yet" state.
// Strictly read-only: nothing here mutates state or triggers a run.
export function ActivityJobs({ onClose, experimentId }: { onClose: () => void; experimentId: string }) {
  return (
    <Card className="shadow-sm border-primary/30">
      <CardHeader>
        <div className="flex items-start justify-between gap-4">
          <div>
            <CardTitle>Activity</CardTitle>
            <CardDescription>
              What the framework has actually been doing — real job runs and proposal outcomes. Read-only; every value
              is what the SDK / the table returned, never a fabricated status.
            </CardDescription>
          </div>
          <Button variant="ghost" onClick={onClose} aria-label="Close activity">
            Close
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-8">
        <RegisteredJobRuns />
        <Separator />
        <RecentOutcomes experimentId={experimentId} />
        <Separator />
        <UntrackedOptimizers />
      </CardContent>
    </Card>
  );
}

// --- Section 1: registered-job run history -----------------------------------------

function RegisteredJobRuns() {
  const [status, setStatus] = useState<FetchStatus>('loading');
  const [result, setResult] = useState<JobsActivityResult | null>(null);
  const [errorText, setErrorText] = useState<string | null>(null);

  // Fetch once, fail-closed: any non-ok / parse / network failure yields an honest
  // unavailable state rather than a fabricated run.
  useEffect(() => {
    let live = true;
    const controller = new AbortController();
    fetch(ACTIVITY_ENDPOINT, { signal: controller.signal })
      .then(async (res) => {
        const body = (await res.json().catch(() => null)) as JobsActivityResult | null;
        if (!live) return;
        if (!res.ok || !body || !Array.isArray(body.jobs)) {
          setErrorText(`Job activity is unavailable (request failed${res.ok ? '' : ` — ${res.status}`}).`);
          setStatus('error');
          return;
        }
        setResult(body);
        setStatus('loaded');
      })
      .catch((error: unknown) => {
        if (!live) return;
        if (error instanceof DOMException && error.name === 'AbortError') return;
        setErrorText('Job activity is unavailable (network error).');
        setStatus('error');
      });
    return () => {
      live = false;
      controller.abort();
    };
  }, []);

  return (
    <section className="space-y-3">
      <SectionHeading
        title="Registered-job run history"
        description="All deployed framework jobs: onboarding, L0 publishing, judge coverage, RLM, MemAlign, memory distillation, and approved-change application. Runs are discovered by name and shown exactly as returned."
      />

      {status === 'loading' && (
        <div className="space-y-3">
          {Array.from({ length: 2 }, (_, i) => (
            <Skeleton key={`job-skeleton-${i}`} className="h-28 w-full" />
          ))}
        </div>
      )}

      {status === 'error' && (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
          {errorText}
        </div>
      )}

      {status === 'loaded' && result && (
        <div className="space-y-4">
          {result.fatal_error && (
            <div className="rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-sm text-amber-700 dark:text-amber-300">
              Could not reach the Jobs API: {result.fatal_error}. Sections below show the per-job state.
            </div>
          )}
          {result.jobs.map((job) => (
            <JobCard key={job.name} job={job} />
          ))}
        </div>
      )}
    </section>
  );
}

function JobCard({ job }: { job: JobActivity }) {
  return (
    <Card className="shadow-sm">
      <CardContent className="p-4 space-y-3">
        <div className="flex flex-wrap items-center gap-2">
          <span className="font-mono text-sm font-semibold">{job.name}</span>
          {job.status === 'ok' && job.description && (
            <span className="text-xs text-muted-foreground">{job.description}</span>
          )}
        </div>

        {job.status === 'not_found' && (
          <p className="text-sm text-muted-foreground">
            Not found in this workspace — this registered job is not deployed here yet. Nothing to show (no fabricated
            run).
          </p>
        )}

        {job.status === 'error' && (
          <p className="text-sm text-destructive">
            Run history unavailable: {job.error}. If this is a permission error, the app service principal needs job-run
            VIEW on this job. Showing no runs rather than a fabricated one.
          </p>
        )}

        {job.status === 'ok' && job.runs.length === 0 && (
          <p className="text-sm text-muted-foreground">No runs recorded yet.</p>
        )}

        {job.status === 'ok' && job.runs.length > 0 && (
          <ul className="divide-y rounded-md border">
            {job.runs.map((run, i) => (
              <li key={run.run_id ?? `run-${i}`}>
                <RunRow run={run} />
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}

function RunRow({ run }: { run: JobRunView }) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 p-3 text-sm">
      <Badge variant="outline" className={RUN_TONE_CLASS[runTone(run)]}>
        {runStateText(run)}
      </Badge>
      {run.run_name && <span className="text-muted-foreground">{run.run_name}</span>}
      <span className="text-muted-foreground">
        started <span className="tabular-nums text-foreground">{fmtEpochMs(run.start_time)}</span>
      </span>
      <span className="text-muted-foreground">
        ended <span className="tabular-nums text-foreground">{fmtEpochMs(run.end_time)}</span>
      </span>
      <span className="text-muted-foreground">
        duration <span className="tabular-nums text-foreground">{fmtDurationMs(run.run_duration)}</span>
      </span>
      {run.state_message && <span className="w-full text-xs text-muted-foreground">{run.state_message}</span>}
      {run.run_page_url && (
        <a
          href={run.run_page_url}
          target="_blank"
          rel="noreferrer"
          className="text-xs text-primary underline underline-offset-2"
        >
          open run ↗
        </a>
      )}
    </div>
  );
}

// --- Section 2: recent proposal/decision outcomes ----------------------------------

function RecentOutcomes({ experimentId }: { experimentId: string }) {
  const params = useMemo(() => ({ experiment_id: sql.string(experimentId) }), [experimentId]);
  const { data, loading, error } = useAnalyticsQuery('recent_activity', params);
  const rows = (data ?? []) as RecentActivityRow[];

  return (
    <section className="space-y-3">
      <SectionHeading
        title="Recent proposal & decision outcomes"
        description="A read-only view of proposal outcomes scoped to the selected MLflow experiment. Switching experiments cannot leak proposals from the prior target."
      />

      {loading && (
        <div className="space-y-2">
          {Array.from({ length: 3 }, (_, i) => (
            <Skeleton key={`outcome-skeleton-${i}`} className="h-10 w-full" />
          ))}
        </div>
      )}

      {error && <div className="rounded-md bg-destructive/10 p-3 text-sm text-destructive">Error: {error}</div>}

      {!loading && !error && rows.length === 0 && (
        <div className="rounded-md border p-4 text-sm text-muted-foreground">
          No proposals recorded yet — the controller has not published any proposed action.
        </div>
      )}

      {!loading && !error && rows.length > 0 && (
        <ul className="divide-y rounded-md border">
          {rows.map((row) => (
            <li key={row.proposal_id} className="flex flex-wrap items-center gap-x-3 gap-y-1 p-3 text-sm">
              <Badge variant="outline" className={OUTCOME_TONE_CLASS[outcomeTone(row.status)]}>
                {row.status}
              </Badge>
              <span className="font-medium">{row.agent_name}</span>
              <span className="text-muted-foreground">{row.action_kind}</span>
              {row.objective_metric && <span className="text-xs text-muted-foreground">→ {row.objective_metric}</span>}
              {row.trigger_summary && (
                <span className="w-full text-xs text-muted-foreground">{row.trigger_summary}</span>
              )}
              <span className="text-xs text-muted-foreground tabular-nums">{row.created_at}</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

// --- Section 3: un-instrumented optimizers (explicit not-tracked) ------------------

function UntrackedOptimizers() {
  return (
    <section className="space-y-3">
      <SectionHeading
        title="Not tracked as jobs yet"
        description="These optimizers do not run as tracked Databricks jobs today, so nothing records their runs. Rather than show a fake progress bar or a zero-filled row, the page states this honestly. An optimizer-run ledger is the named follow-on that would light these up."
      />
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        {UNTRACKED_OPTIMIZERS.map((opt) => (
          <div key={opt.key} className="rounded-md border p-3 space-y-1">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-sm font-medium text-foreground">{opt.name}</span>
              <Badge variant="outline" className="text-muted-foreground">
                not tracked yet
              </Badge>
            </div>
            <p className="text-xs text-muted-foreground">{opt.detail}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

function SectionHeading({ title, description }: { title: string; description: string }) {
  return (
    <div>
      <h3 className="text-lg font-semibold text-foreground">{title}</h3>
      <p className="text-sm text-muted-foreground">{description}</p>
    </div>
  );
}
