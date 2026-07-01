import { describe, it, expect } from 'vitest';
import {
  fetchJobsActivity,
  jobsActivityBridge,
  REGISTERED_JOB_NAMES,
  type DiscoveredJob,
  type JobRunView,
  type JobsClient,
} from './bridge';

interface JobSpec {
  job?: DiscoveredJob | null;
  runs?: JobRunView[];
  discoverError?: string;
  runsError?: string;
}

// A fake JobsClient driven by a per-name spec — no live workspace, no SDK. Records
// its calls so tests can assert that a failing job never suppresses the others and
// that a run is never fetched for a job that wasn't found.
function fakeClient(spec: Record<string, JobSpec>) {
  const calls = { discover: [] as string[], runs: [] as number[] };
  const client: JobsClient = {
    discoverJobByName(name: string): Promise<DiscoveredJob | null> {
      calls.discover.push(name);
      const s = spec[name];
      if (s?.discoverError) return Promise.reject(new Error(s.discoverError));
      return Promise.resolve(s?.job ?? null);
    },
    recentRuns(jobId: number, limit: number): Promise<JobRunView[]> {
      calls.runs.push(jobId);
      const s = Object.values(spec).find((x) => x.job && x.job.job_id === jobId);
      if (s?.runsError) return Promise.reject(new Error(s.runsError));
      return Promise.resolve((s?.runs ?? []).slice(0, limit));
    },
  };
  return { client, calls };
}

// A real RUNNING run: no result_state, no end_time, no run_duration. The bridge must
// pass this through untouched — never inventing a SUCCESS or a duration.
const RUNNING_RUN: JobRunView = {
  run_id: 11,
  run_name: 'apply #11',
  run_page_url: 'https://example/run/11',
  life_cycle_state: 'RUNNING',
  start_time: 1_000,
};

const FAILED_RUN: JobRunView = {
  run_id: 10,
  run_name: 'apply #10',
  life_cycle_state: 'TERMINATED',
  result_state: 'FAILED',
  state_message: 'boom',
  start_time: 500,
  end_time: 900,
  run_duration: 400,
};

describe('fetchJobsActivity — real data, verbatim', () => {
  it('returns an ok section with the runs exactly as the SDK gave them', async () => {
    const { client, calls } = fakeClient({
      'ail-apply-service': {
        job: { job_id: 7, name: 'ail-apply-service', description: 'on-demand' },
        runs: [RUNNING_RUN, FAILED_RUN],
      },
    });

    const result = await fetchJobsActivity(client, ['ail-apply-service'], 10);

    expect(result.jobs).toHaveLength(1);
    const section = result.jobs[0];
    expect(section).toEqual({
      name: 'ail-apply-service',
      status: 'ok',
      job_id: 7,
      description: 'on-demand',
      runs: [RUNNING_RUN, FAILED_RUN],
    });
    // The RUNNING run is untouched: no fabricated result_state / duration.
    if (section.status === 'ok') {
      expect(section.runs[0].result_state).toBeUndefined();
      expect(section.runs[0].run_duration).toBeUndefined();
      expect(section.runs[1].result_state).toBe('FAILED');
    }
    expect(calls.runs).toEqual([7]);
  });

  it('a found job with zero runs is an honest empty list — not a fabricated row', async () => {
    const { client } = fakeClient({
      'ail-l0-publish-scheduled': { job: { job_id: 3, name: 'ail-l0-publish-scheduled' }, runs: [] },
    });

    const result = await fetchJobsActivity(client, ['ail-l0-publish-scheduled'], 10);

    expect(result.jobs[0]).toEqual({
      name: 'ail-l0-publish-scheduled',
      status: 'ok',
      job_id: 3,
      description: undefined,
      runs: [],
    });
  });
});

describe('fetchJobsActivity — fail-closed (never fabricates)', () => {
  it('a missing job yields not_found — never a fake row, and no runs are fetched for it', async () => {
    const { client, calls } = fakeClient({ 'ail-apply-service': { job: null } });

    const result = await fetchJobsActivity(client, ['ail-apply-service'], 10);

    expect(result.jobs[0]).toEqual({ name: 'ail-apply-service', status: 'not_found' });
    expect(calls.runs).toEqual([]); // never listed runs for a job we couldn't find
  });

  it('a discover permission error yields an error section — never a fabricated run', async () => {
    const { client, calls } = fakeClient({
      'ail-apply-service': { discoverError: 'PERMISSION_DENIED: cannot view job' },
    });

    const result = await fetchJobsActivity(client, ['ail-apply-service'], 10);

    expect(result.jobs[0]).toEqual({
      name: 'ail-apply-service',
      status: 'error',
      error: 'PERMISSION_DENIED: cannot view job',
    });
    expect(calls.runs).toEqual([]);
  });

  it('a listRuns failure yields an error section (never partial fabricated runs)', async () => {
    const { client } = fakeClient({
      'ail-apply-service': { job: { job_id: 7, name: 'ail-apply-service' }, runsError: 'RUN_VIEW denied' },
    });

    const result = await fetchJobsActivity(client, ['ail-apply-service'], 10);

    expect(result.jobs[0]).toEqual({ name: 'ail-apply-service', status: 'error', error: 'RUN_VIEW denied' });
  });

  it("one job's failure never suppresses another job's real data (per-job isolation)", async () => {
    const { client } = fakeClient({
      'ail-apply-service': { discoverError: 'PERMISSION_DENIED' },
      'ail-l0-publish-scheduled': { job: { job_id: 3, name: 'ail-l0-publish-scheduled' }, runs: [FAILED_RUN] },
    });

    const result = await fetchJobsActivity(client, REGISTERED_JOB_NAMES, 10);

    expect(result.jobs[0].status).toBe('error');
    expect(result.jobs[1]).toMatchObject({ name: 'ail-l0-publish-scheduled', status: 'ok', runs: [FAILED_RUN] });
  });

  it('honors the run limit', async () => {
    const runs = [FAILED_RUN, RUNNING_RUN, FAILED_RUN, RUNNING_RUN];
    const { client } = fakeClient({ 'ail-apply-service': { job: { job_id: 7 }, runs } });

    const result = await fetchJobsActivity(client, ['ail-apply-service'], 2);

    const section = result.jobs[0];
    expect(section.status).toBe('ok');
    if (section.status === 'ok') expect(section.runs).toHaveLength(2);
  });
});

describe('jobsActivityBridge — client-build failure is fail-closed', () => {
  it('marks every section error (+ fatal_error) when the SDK client cannot be built', async () => {
    const bridge = jobsActivityBridge({
      clientFactory: () => {
        throw new Error('no databricks credentials in this environment');
      },
    });

    const result = await bridge();

    expect(result.fatal_error).toMatch(/no databricks credentials/);
    expect(result.jobs).toHaveLength(REGISTERED_JOB_NAMES.length);
    for (const job of result.jobs) {
      expect(job.status).toBe('error');
    }
  });

  it('clamps a bogus limit to the default (never a zero / negative / oversized page)', async () => {
    let seenLimit = -1;
    const client: JobsClient = {
      discoverJobByName: () => Promise.resolve({ job_id: 1 }),
      recentRuns: (_jobId, limit) => {
        seenLimit = limit;
        return Promise.resolve([]);
      },
    };
    const bridge = jobsActivityBridge({ clientFactory: () => client, names: ['ail-apply-service'] });

    await bridge(0);
    expect(seenLimit).toBeGreaterThanOrEqual(1);
    expect(seenLimit).toBeLessThan(25);
  });
});
