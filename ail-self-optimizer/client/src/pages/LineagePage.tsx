import { PageHeader } from '../shell/PageHeader';
import { RequireAgent } from '../shell/RequireAgent';
import { PanelBoundary } from '../shell/PanelBoundary';
import { LineageTimeline } from '../components/LineageTimeline';

// Lineage & audit — the prompt-registry timeline. LineageTimeline is re-homed
// unchanged: a force-registered non-improving version keeps its amber warning and is
// never styled as a win.
export function LineagePage() {
  return (
    <div className="space-y-6">
      <PageHeader />
      <RequireAgent>
        {(agent) => (
          <PanelBoundary title="Lineage timeline failed to load">
            <LineageTimeline agentName={agent.agent_name} experimentId={agent.experiment_id} />
          </PanelBoundary>
        )}
      </RequireAgent>
    </div>
  );
}
