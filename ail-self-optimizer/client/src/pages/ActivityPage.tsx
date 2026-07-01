import { useNavigate } from 'react-router';
import { PageHeader } from '../shell/PageHeader';
import { PanelBoundary } from '../shell/PanelBoundary';
import { ActivityJobs } from '../components/ActivityJobs';
import { agentSearch } from '../lib/navigation';
import { useAgent } from '../context/agent-context';

// Activity — workspace-wide (not agent-scoped), so no RequireAgent guard. ActivityJobs
// is re-homed unchanged: its fail-closed states (error / not_found / "not tracked as
// jobs yet") render exactly as before. Its own Card header is the page hero, so
// PageHeader shows the breadcrumb only. The header's Close returns to Overview.
export function ActivityPage() {
  const navigate = useNavigate();
  const { selected } = useAgent();
  return (
    <div className="space-y-6">
      <PageHeader />
      <PanelBoundary title="Activity failed to load">
        <ActivityJobs onClose={() => void navigate(`/overview${agentSearch(selected?.agent_name)}`)} />
      </PanelBoundary>
    </div>
  );
}
