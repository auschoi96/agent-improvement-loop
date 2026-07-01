import { createApp, analytics, server } from '@databricks/appkit';
import { approvals } from './plugins/approvals';
import { onboarding } from './plugins/onboarding';

// analytics() + server() serve the two-tier SELECT-only reads; approvals() adds the
// app's approve/reject write-path (Phase C lane 3b, docs/LOOP_CONTROLLER.md);
// onboarding() adds the authenticated "Add an agent" wizard write-path (slice 1,
// docs/ONBOARDING_WIZARD.md) — fresh-experiment validate/create + agent registration
// behind the same fail-closed, identity-from-headers engine bridge.
createApp({
  plugins: [analytics(), server(), approvals(), onboarding()],
}).catch(console.error);
