import { describe, expect, it } from 'vitest';
import { reconcileAgentsState, type AgentsState } from './agent-state';

const AGENTS = [
  {
    agent_name: 'claude_code',
    experiment_id: 'exp-subject',
    description: 'Claude Code',
  },
];

const loadedState = (): AgentsState => ({
  agents: AGENTS,
  loading: false,
  error: null,
  hasLoaded: true,
});

describe('reconcileAgentsState', () => {
  it('preserves the completed registry while a background query clears its data', () => {
    const previous = loadedState();

    expect(reconcileAgentsState(previous, { data: null, loading: false, error: null })).toBe(previous);
    expect(reconcileAgentsState(previous, { data: null, loading: true, error: null })).toBe(previous);
  });

  it('preserves the completed registry when a background refresh fails', () => {
    const previous = loadedState();

    expect(reconcileAgentsState(previous, { data: null, loading: false, error: 'temporary failure' })).toBe(previous);
  });

  it('accepts a completed empty result as an authoritative empty registry', () => {
    expect(reconcileAgentsState(loadedState(), { data: [], loading: false, error: null })).toEqual({
      agents: [],
      loading: false,
      error: null,
      hasLoaded: true,
    });
  });

  it('does not publish partial data while a refresh is still loading', () => {
    const previous = loadedState();

    expect(reconcileAgentsState(previous, { data: [], loading: true, error: null })).toBe(previous);
  });

  it('publishes a completed refresh without an intermediate empty state', () => {
    const previous = loadedState();
    const refreshed = [{ ...AGENTS[0], experiment_id: 'exp-new' }];

    const duringRefresh = reconcileAgentsState(previous, { data: null, loading: true, error: null });
    expect(duringRefresh).toBe(previous);
    expect(reconcileAgentsState(duringRefresh, { data: refreshed, loading: false, error: null })).toEqual({
      agents: refreshed,
      loading: false,
      error: null,
      hasLoaded: true,
    });
  });

  it('still exposes an initial-load failure when no registry has ever loaded', () => {
    const initial: AgentsState = { agents: [], loading: true, error: null, hasLoaded: false };

    expect(reconcileAgentsState(initial, { data: null, loading: false, error: 'not authorized' })).toEqual({
      agents: [],
      loading: false,
      error: 'not authorized',
      hasLoaded: false,
    });
  });
});
