import { useEffect } from 'react';
import {
  useAnalyticsQuery,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  Skeleton,
} from '@databricks/appkit-ui/react';

export interface AgentRow {
  agent_name: string;
  experiment_id: string;
  description: string;
}

// Stable empty-params reference (the agents query takes none).
const EMPTY_PARAMS = {};

// The agent switcher: one MLflow experiment per agent (docs/OBSERVABILITY_APP.md).
// Selecting an agent re-points the L0 leaderboard and the version comparison at
// that agent's experiment. Auto-selects the first registered agent on load.
export function AgentSwitcher({ value, onChange }: { value: string | null; onChange: (agent: AgentRow) => void }) {
  const { data, loading, error } = useAnalyticsQuery('agents', EMPTY_PARAMS);

  useEffect(() => {
    if (!value && data && data.length > 0) {
      onChange(data[0]);
    }
  }, [value, data, onChange]);

  if (loading) return <Skeleton className="h-9 w-64" />;
  if (error) {
    return <div className="text-destructive bg-destructive/10 p-2 rounded-md text-sm">Error: {error}</div>;
  }
  if (!data?.length) {
    return <div className="text-muted-foreground text-sm">No agents registered.</div>;
  }

  const selected = data.find((a) => a.agent_name === value) ?? data[0];

  return (
    <div className="flex flex-col gap-1">
      <Select
        value={selected.agent_name}
        onValueChange={(name) => {
          const next = data.find((a) => a.agent_name === name);
          if (next) onChange(next);
        }}
      >
        <SelectTrigger className="w-72">
          <SelectValue placeholder="Select an agent">{selected.agent_name}</SelectValue>
        </SelectTrigger>
        <SelectContent>
          {data.map((a) => (
            <SelectItem key={a.agent_name} value={a.agent_name}>
              <span className="flex flex-col">
                <span>{a.agent_name}</span>
                <span className="text-xs text-muted-foreground">experiment {a.experiment_id}</span>
              </span>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <span className="text-xs text-muted-foreground">
        {selected.description} · experiment {selected.experiment_id}
      </span>
    </div>
  );
}
