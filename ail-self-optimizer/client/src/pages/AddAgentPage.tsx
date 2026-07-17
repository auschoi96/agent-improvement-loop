import { useRef, useState } from 'react';
import { useNavigate } from 'react-router';
import { PageHeader } from '../shell/PageHeader';
import { PanelBoundary } from '../shell/PanelBoundary';
import { OnboardingWizard } from '../components/OnboardingWizard';
import { agentSearch } from '../lib/navigation';
import { useAgent } from '../context/agent-context';
import { QuickConnectPanel } from '../components/QuickConnectPanel';

// Add agent — the distinct onboarding flow. OnboardingWizard is re-homed unchanged:
// every gate/scorer/floor fact still comes from the Python engine; nothing is
// re-derived in TS. On registration we refresh the agent registry (so the new agent
// appears in the switcher) and remember its name; Done/Close then lands on the new
// agent's Overview.
export function AddAgentPage() {
  const navigate = useNavigate();
  const { selected, reloadAgents } = useAgent();
  const registeredName = useRef<string | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const goOverview = () =>
    void navigate(`/overview${agentSearch(registeredName.current ?? selected?.agent_name ?? null)}`);

  return (
    <div className="space-y-6">
      <PageHeader />
      {!showAdvanced && (
        <QuickConnectPanel
          onAdvanced={() => setShowAdvanced(true)}
          onRegistered={(name) => {
            registeredName.current = name;
            reloadAgents();
          }}
        />
      )}
      {!showAdvanced && (
        <p className="text-sm text-muted-foreground">
          Already have MLflow or OTEL traces? Use advanced setup to connect the existing experiment and include its
          history.
        </p>
      )}
      {showAdvanced && (
        <PanelBoundary title="Onboarding wizard failed to load">
          <OnboardingWizard
            onRegistered={(name) => {
              registeredName.current = name;
              reloadAgents();
            }}
            onClose={goOverview}
          />
        </PanelBoundary>
      )}
    </div>
  );
}
