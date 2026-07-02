import { PageHeader } from '../shell/PageHeader';
import { RequireAgent } from '../shell/RequireAgent';
import { PanelBoundary } from '../shell/PanelBoundary';
import { LabelingPanel } from '../components/LabelingPanel';

// Labeling (L4) — the human-facing surface that PRODUCES the labels the loop needs.
// A signed-in user grades traces along the agent's REGISTERED judged dimensions; each
// label is written (server-side) as a HUMAN assessment named for the judge, which is
// what L2's scheduled auto-align pairs to align the judge. Agent-scoped: the panel
// reads the selected agent's MLflow experiment.
export function LabelingPage() {
  return (
    <div className="space-y-6">
      <PageHeader />
      <RequireAgent>
        {(agent) => (
          <PanelBoundary title="Labeling failed to load">
            <LabelingPanel agentName={agent.agent_name} experimentId={agent.experiment_id} />
          </PanelBoundary>
        )}
      </RequireAgent>
    </div>
  );
}
