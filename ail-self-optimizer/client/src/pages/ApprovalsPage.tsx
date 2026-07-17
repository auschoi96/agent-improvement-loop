import { PageHeader } from '../shell/PageHeader';
import { RequireAgent } from '../shell/RequireAgent';
import { PanelBoundary } from '../shell/PanelBoundary';
import { ApprovalQueue } from '../components/ApprovalQueue';

// Approvals — the human control plane and the app's only write-path. ApprovalQueue is
// re-homed unchanged: the authenticated approve/reject POST (identity from headers,
// engine re-checks the gate) is untouched.
export function ApprovalsPage() {
  return (
    <div className="space-y-6">
      <PageHeader />
      <RequireAgent>
        {(agent) => (
          <PanelBoundary title="Approval queue failed to load">
            <ApprovalQueue agentName={agent.agent_name} experimentId={agent.experiment_id} />
          </PanelBoundary>
        )}
      </RequireAgent>
    </div>
  );
}
