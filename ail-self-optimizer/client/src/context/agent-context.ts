import { createContext, useContext } from 'react';

// One MLflow experiment per agent (docs/OBSERVABILITY_APP.md). The registry row the
// `agents` query returns.
export interface AgentRow {
  agent_name: string;
  experiment_id: string;
  reviewer_experiment_id?: string;
  description: string;
}

export interface AgentContextValue {
  agents: AgentRow[];
  loading: boolean;
  error: string | null;
  /** The active agent — the URL-selected one, or the first registered as a fallback. */
  selected: AgentRow | null;
  /** Select an agent by name; writes it to the URL so the view is shareable/refresh-safe. */
  selectAgent: (agentName: string) => void;
  /** Force the agents query to re-run (e.g. after onboarding registers a new agent). */
  reloadAgents: () => void;
}

// The context object + hook live here (not in AgentContext.tsx) so the provider file
// exports only a component — keeping React Fast Refresh boundaries clean.
export const AgentContext = createContext<AgentContextValue | null>(null);

export function useAgent(): AgentContextValue {
  const ctx = useContext(AgentContext);
  if (!ctx) throw new Error('useAgent must be used within an AgentProvider');
  return ctx;
}
