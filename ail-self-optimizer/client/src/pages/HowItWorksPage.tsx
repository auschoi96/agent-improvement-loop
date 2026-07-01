import { useNavigate } from 'react-router';
import { PageHeader } from '../shell/PageHeader';
import { PanelBoundary } from '../shell/PanelBoundary';
import { TutorialGuide } from '../components/TutorialGuide';
import { agentSearch } from '../lib/navigation';
import { useAgent } from '../context/agent-context';

// How it works — the guided tour. TutorialGuide is re-homed unchanged: it still fetches
// the readiness floors live from the Python engine and shows a neutral placeholder
// (never a fabricated number) when they're unavailable. Its own Card header is the
// hero. Done/Close returns to Overview.
export function HowItWorksPage() {
  const navigate = useNavigate();
  const { selected } = useAgent();
  return (
    <div className="space-y-6">
      <PageHeader />
      <PanelBoundary title="Tutorial failed to load">
        <TutorialGuide onClose={() => void navigate(`/overview${agentSearch(selected?.agent_name)}`)} />
      </PanelBoundary>
    </div>
  );
}
