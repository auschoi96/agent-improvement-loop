import { PageHeader } from '../shell/PageHeader';
import { RequireAgent } from '../shell/RequireAgent';
import { PanelBoundary } from '../shell/PanelBoundary';
import { VersionComparison } from '../components/VersionComparison';

// Compare — baseline vs a newer version. VersionComparison is re-homed unchanged: its
// amber "controlled proof · collecting" verdict and red "regressed" verdict are
// decided in Python and rendered here exactly as before.
export function ComparePage() {
  return (
    <div className="space-y-6">
      <PageHeader />
      <RequireAgent>
        {(agent) => (
          <PanelBoundary title="Version comparison failed to load">
            <VersionComparison agentName={agent.agent_name} experimentId={agent.experiment_id} />
          </PanelBoundary>
        )}
      </RequireAgent>
    </div>
  );
}
