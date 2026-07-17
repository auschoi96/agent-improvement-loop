import { GepaDispatcher } from '../components/GepaDispatcher';
import { PageHeader } from '../shell/PageHeader';
import { PanelBoundary } from '../shell/PanelBoundary';
import { RequireAgent } from '../shell/RequireAgent';

export function OptimizePage() {
  return (
    <div className="space-y-6">
      <PageHeader />
      <RequireAgent>
        {(agent) => (
          <PanelBoundary title="GEPA dispatcher failed to load">
            <GepaDispatcher key={`${agent.agent_name}:${agent.experiment_id}`} agent={agent} />
          </PanelBoundary>
        )}
      </RequireAgent>
    </div>
  );
}
