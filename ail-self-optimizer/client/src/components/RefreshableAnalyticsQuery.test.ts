import { describe, expect, it } from 'vitest';
import { reconcileRefreshState, type RefreshableQueryState } from '../lib/refresh-state';

const loaded: RefreshableQueryState<number[]> = {
  data: [1, 2, 3],
  loading: false,
  refreshing: false,
  error: null,
};

describe('reconcileRefreshState', () => {
  it('retains visible data while a replacement query is loading', () => {
    expect(reconcileRefreshState(loaded, { data: null, loading: true, error: null })).toEqual({
      ...loaded,
      refreshing: true,
    });
  });

  it('retains visible data and reports a background refresh error', () => {
    expect(reconcileRefreshState(loaded, { data: null, loading: false, error: 'warehouse unavailable' })).toEqual({
      ...loaded,
      error: 'warehouse unavailable',
    });
  });

  it('atomically replaces retained data after a successful refresh', () => {
    expect(reconcileRefreshState(loaded, { data: [4], loading: false, error: null })).toEqual({
      data: [4],
      loading: false,
      refreshing: false,
      error: null,
    });
  });
});
