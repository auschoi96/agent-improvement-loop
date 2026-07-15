import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react';
import { useSearchParams } from 'react-router';
import { useAnalyticsQuery } from '@databricks/appkit-ui/react';
import { AgentContext, type AgentContextValue, type AgentRow } from './agent-context';

// The selection lives in the URL (?agent=<name>) so every routed view is deep-linkable
// and a refresh/share restores both the section AND the agent.
const AGENT_PARAM = 'agent';
// Stable empty-params reference (the agents query takes none) — identical to the
// pre-shell AgentSwitcher, so the data logic is unchanged; it is just lifted to a
// provider so the sidebar badge, the top-bar switcher, and every page read one source.
const EMPTY_PARAMS = {};

interface AgentsState {
  agents: AgentRow[];
  loading: boolean;
  error: string | null;
}

// Isolated subscriber: owns the `agents` analytics query and reports its result up. It
// renders nothing and sits as a sibling of the app tree, so remounting it via `key`
// (reloadAgents) forces a fresh subscription/refetch WITHOUT remounting the app —
// mirroring how the pre-shell App remounted the switcher to pick up a new agent.
function AgentsSubscriber({ onResult }: { onResult: (state: AgentsState) => void }) {
  const { data, loading, error } = useAnalyticsQuery('agents', EMPTY_PARAMS);
  useEffect(() => {
    onResult({ agents: (data ?? []) as AgentRow[], loading, error });
  }, [data, loading, error, onResult]);
  return null;
}

export function AgentProvider({ children }: { children: ReactNode }) {
  const [reloadKey, setReloadKey] = useState(0);
  const [state, setState] = useState<AgentsState>({ agents: [], loading: true, error: null });
  const [searchParams, setSearchParams] = useSearchParams();

  const reloadAgents = useCallback(() => {
    setState((s) => ({ ...s, loading: true }));
    setReloadKey((k) => k + 1);
  }, []);

  useEffect(() => {
    const timer = window.setInterval(reloadAgents, 30_000);
    return () => window.clearInterval(timer);
  }, [reloadAgents]);

  const agents = state.agents;
  const requested = searchParams.get(AGENT_PARAM);

  const selected = useMemo<AgentRow | null>(() => {
    if (agents.length === 0) return null;
    return agents.find((a) => a.agent_name === requested) ?? agents[0];
  }, [agents, requested]);

  // Auto-select the first registered agent when NONE is requested (preserves the
  // pre-shell behavior), writing it to the URL so the selection is shareable/refresh-
  // safe. Only fires when the param is absent — a present-but-not-yet-loaded name (e.g.
  // an agent just registered, mid-refetch) is left intact rather than clobbered.
  useEffect(() => {
    if (agents.length === 0 || requested !== null || !selected) return;
    setSearchParams(
      (prev) => {
        const next = new URLSearchParams(prev);
        next.set(AGENT_PARAM, selected.agent_name);
        return next;
      },
      { replace: true }
    );
  }, [agents, requested, selected, setSearchParams]);

  const selectAgent = useCallback(
    (agentName: string) => {
      setSearchParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          next.set(AGENT_PARAM, agentName);
          return next;
        },
        { replace: false }
      );
    },
    [setSearchParams]
  );

  const value = useMemo<AgentContextValue>(
    () => ({ agents, loading: state.loading, error: state.error, selected, selectAgent, reloadAgents }),
    [agents, state.loading, state.error, selected, selectAgent, reloadAgents]
  );

  return (
    <AgentContext.Provider value={value}>
      <AgentsSubscriber key={reloadKey} onResult={setState} />
      {children}
    </AgentContext.Provider>
  );
}
