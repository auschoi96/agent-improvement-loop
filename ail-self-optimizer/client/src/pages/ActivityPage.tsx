import { useNavigate } from 'react-router';
import { PageHeader } from '../shell/PageHeader';
import { PanelBoundary } from '../shell/PanelBoundary';
import { ActivityJobs } from '../components/ActivityJobs';
import { agentSearch } from '../lib/navigation';
import { useAgent } from '../context/agent-context';
import { RequireAgent } from '../shell/RequireAgent';

// Activity combines registry-driven shared job runs with proposal outcomes scoped to
// the selected experiment. RequireAgent keeps the experiment boundary explicit.
export function ActivityPage() {
  const navigate = useNavigate();
  const { selected } = useAgent();
  return (
    <div className="space-y-6">
      <PageHeader />
      <RequireAgent>
        {(agent) => (
          <PanelBoundary title="Activity failed to load">
            <ActivityJobs
              experimentId={agent.experiment_id}
              onClose={() => void navigate(`/overview${agentSearch(selected?.agent_name)}`)}
            />
          </PanelBoundary>
        )}
      </RequireAgent>
    </div>
  );
}
