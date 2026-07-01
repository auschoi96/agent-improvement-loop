import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue, Skeleton } from '@databricks/appkit-ui/react';
import { useAgent } from '../context/agent-context';

// The agent/experiment switcher — the shell's "project switcher", living in the top
// bar. One MLflow experiment per agent (docs/OBSERVABILITY_APP.md). It is now a thin,
// controlled Select bound to AgentContext (which owns the `agents` query and the
// URL-based selection), so selecting an agent re-points every page and is reflected in
// the URL. Loading / error / empty stay honest.
export function AgentSwitcher() {
  const { agents, loading, error, selected, selectAgent } = useAgent();

  if (loading && agents.length === 0) return <Skeleton className="h-9 w-64" />;
  if (error) {
    return <div className="rounded-md bg-destructive/10 p-2 text-sm text-destructive">Error: {error}</div>;
  }
  if (agents.length === 0) {
    return <div className="text-sm text-muted-foreground">No agents registered.</div>;
  }
  if (!selected) return null;

  return (
    <Select value={selected.agent_name} onValueChange={selectAgent}>
      <SelectTrigger className="w-56 md:w-64" aria-label="Select agent">
        <SelectValue placeholder="Select an agent">{selected.agent_name}</SelectValue>
      </SelectTrigger>
      <SelectContent>
        {agents.map((a) => (
          <SelectItem key={a.agent_name} value={a.agent_name}>
            <span className="flex flex-col">
              <span>{a.agent_name}</span>
              <span className="text-xs text-muted-foreground">experiment {a.experiment_id}</span>
            </span>
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
