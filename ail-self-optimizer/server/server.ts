import { createApp, analytics, server } from '@databricks/appkit';
import { approvals } from './plugins/approvals';

// analytics() + server() serve the two-tier SELECT-only reads; approvals() adds the
// app's first (and only) write-path: the authenticated approve/reject route that
// triggers the framework's gated apply engine (Phase C lane 3b, docs/LOOP_CONTROLLER.md).
createApp({
  plugins: [analytics(), server(), approvals()],
}).catch(console.error);
