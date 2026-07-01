import { Plugin, toPlugin, type IAppRouter, type PluginManifest } from '@databricks/appkit';
import manifest from './manifest.json';
import { selectJobsActivityBridge, type JobsActivityBridge } from './bridge';

// The minimal HTTP response shape the handler needs — Express's Response satisfies it
// structurally, so the same handler is driven by injectRoutes and by a fake res in
// tests (no server, no live SDK). Mirrors the approvals/onboarding plugins.
export interface ActivityHttpResponse {
  status(code: number): ActivityHttpResponse;
  json(body: unknown): void;
}

// The read-only Activity job-runs endpoint. There is NO write-path and NO auth gate
// beyond the platform proxy: the reads run under the app identity via the bridge and
// return the SDK's real run data verbatim. The bridge is fail-closed (per-job
// error/not_found sections, never a fabricated run); this handler adds one more layer
// of defense — if the bridge itself throws, it still returns an honest unavailable
// body rather than a 500 with no shape.
export async function handleActivity(res: ActivityHttpResponse, bridge: JobsActivityBridge): Promise<void> {
  try {
    const result = await bridge();
    res.status(200).json(result);
  } catch (err) {
    res.status(200).json({
      jobs: [],
      fatal_error: err instanceof Error ? err.message : 'job activity is unavailable',
    });
  }
}

// The custom AppKit plugin exposing the read-only Activity data source. Routes mount
// under /api/jobactivity/... (server plugin convention). Read-only by construction:
// the only route is a GET that lists job runs; it never triggers a run and never
// writes. The app's only write-path remains approvals.
export class JobActivityPlugin extends Plugin {
  static manifest = manifest as PluginManifest<'jobactivity'>;

  private readonly bridge: JobsActivityBridge = selectJobsActivityBridge();

  injectRoutes(router: IAppRouter): void {
    this.route(router, {
      name: 'activity',
      method: 'get',
      path: '/activity',
      handler: (_req, res) => handleActivity(res, this.bridge),
    });
  }
}

export const jobActivity = toPlugin(JobActivityPlugin);
