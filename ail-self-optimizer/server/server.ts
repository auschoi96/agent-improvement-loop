import { createApp, analytics, server } from '@databricks/appkit';
import { approvals } from './plugins/approvals';
import { onboarding } from './plugins/onboarding';
import { jobActivity } from './plugins/jobs';
import { labeling } from './plugins/labeling';

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
  plugins: [analytics(), server(), approvals(), onboarding(), jobActivity(), labeling()],
}).catch(console.error);
