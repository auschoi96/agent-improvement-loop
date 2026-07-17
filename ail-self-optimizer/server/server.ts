import { createApp, analytics, jobs, server } from '@databricks/appkit';
import { z } from 'zod';
import { approvals } from './plugins/approvals';
import { onboarding } from './plugins/onboarding';
import { jobActivity } from './plugins/jobs';
import { labeling } from './plugins/labeling';
import { extractGepaResult } from './gepa-output';

const gepaDispatchSchema = z
  .object({
    job_parameters: z.record(z.string(), z.string()),
    idempotency_token: z.string().min(1).max(64),
  })
  .strict();

// analytics() + server() serve the two-tier SELECT-only reads; approvals() adds the
// app's approve/reject write-path (Phase C lane 3b, docs/LOOP_CONTROLLER.md);
// onboarding() adds the authenticated "Add an agent" wizard write-path (slice 1,
// docs/ONBOARDING_WIZARD.md) — fresh-experiment validate/create + agent registration
// behind the same fail-closed, identity-from-headers engine bridge. jobActivity()
// adds the READ-ONLY Activity page data source: recent runs of the framework's
// registered jobs via the SDK (fail-closed, never fabricated) — no write-path.
// labeling() adds the authenticated in-app labeling write-path (L4, docs/LABELING_UI.md)
// — record a HUMAN label named for a registered judge (reusing ail.judges.labeling),
// so L2's scheduled auto-align can pair the labels and align the judge.
createApp({
  plugins: [
    analytics(),
    server(),
    jobs({ jobs: { gepa: { params: gepaDispatchSchema } } }),
    approvals(),
    onboarding(),
    jobActivity(),
    labeling(),
  ],
  onPluginsReady(appkit) {
    appkit.server.extend((app) => {
      app.get('/api/gepa/runs/:runId/output', async (req, res) => {
        const runId = Number(req.params.runId);
        if (!Number.isSafeInteger(runId) || runId <= 0) {
          res.status(400).json({ error: 'runId must be a positive integer' });
          return;
        }

        const output = await appkit.jobs('gepa').getRunOutput(runId);
        if (!output.ok) {
          res.status(output.status).json({ error: output.message });
          return;
        }
        res.json({
          result: extractGepaResult(output.data),
          logs_truncated: output.data.logs_truncated ?? false,
          task_error: output.data.error ?? output.data.error_trace ?? null,
        });
      });
    });
  },
}).catch(console.error);
