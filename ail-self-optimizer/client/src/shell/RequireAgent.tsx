import { type ReactNode } from 'react';
import { Link } from 'react-router';
import { Alert, AlertDescription, AlertTitle, Button, Skeleton } from '@databricks/appkit-ui/react';
import { AlertTriangle, Bot, Plus } from 'lucide-react';
import { useAgent, type AgentRow } from '../context/agent-context';
import { EmptyState } from './EmptyState';

// Guard for agent-scoped pages. Renders honest states while the agent registry loads
// / errors / is empty, and only invokes `children` once a concrete agent is selected —
// passing it down so the panel never has to null-check the selection.
export function RequireAgent({ children }: { children: (agent: AgentRow) => ReactNode }) {
  const { loading, error, agents, selected } = useAgent();

  if (loading && agents.length === 0) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-48" />
        <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
          {Array.from({ length: 4 }, (_, i) => (
            <Skeleton key={`require-agent-skeleton-${i}`} className="h-24 w-full" />
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertTriangle className="h-4 w-4" />
        <AlertTitle>Couldn’t load the agent registry</AlertTitle>
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    );
  }

  if (agents.length === 0) {
    return (
      <EmptyState
        icon={Bot}
        title="No agents registered yet"
        description="Register your first agent from a fresh MLflow experiment to start tracking its L0 metrics."
        action={
          <Button asChild>
            <Link to="/add-agent">
              <Plus className="h-4 w-4" /> Add an agent
            </Link>
          </Button>
        }
      />
    );
  }

  if (!selected) {
    return (
      <EmptyState
        icon={Bot}
        title="Select an agent"
        description="Choose an agent from the switcher in the top bar to view its metrics."
      />
    );
  }

  return <>{children(selected)}</>;
}
