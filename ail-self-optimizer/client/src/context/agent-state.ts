import type { AgentRow } from './agent-context';

export interface AgentsState {
  agents: AgentRow[];
  loading: boolean;
  error: string | null;
  hasLoaded: boolean;
}

export interface AgentsQueryResult {
  data: AgentRow[] | null;
  loading: boolean;
  error: string | null;
}

/**
 * Keep the last completed registry result while a background subscription starts.
 * AppKit clears `data` before every query; publishing that transient null as an
 * empty registry makes RequireAgent remove and then recreate the active page.
 */
export function reconcileAgentsState(previous: AgentsState, result: AgentsQueryResult): AgentsState {
  if (!result.loading && !result.error && result.data !== null) {
    return {
      agents: result.data,
      loading: false,
      error: null,
      hasLoaded: true,
    };
  }

  // A completed registry is authoritative until another query completes. Loading
  // and refresh failures must not replace a usable app with a route-level fallback.
  if (previous.hasLoaded) return previous;

  if (result.error) {
    return { ...previous, loading: false, error: result.error };
  }

  if (result.loading && (!previous.loading || previous.error)) {
    return { ...previous, loading: true, error: null };
  }

  return previous;
}
