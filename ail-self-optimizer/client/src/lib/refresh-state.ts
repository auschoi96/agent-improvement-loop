export interface RefreshableQueryState<T> {
  data: T | null;
  loading: boolean;
  refreshing: boolean;
  error: string | null;
}

export interface QueryResult<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}

export function reconcileRefreshState<T>(
  previous: RefreshableQueryState<T>,
  result: QueryResult<T>
): RefreshableQueryState<T> {
  if (!result.loading && !result.error && result.data !== null) {
    return { data: result.data, loading: false, refreshing: false, error: null };
  }
  if (previous.data !== null) {
    return {
      ...previous,
      loading: false,
      refreshing: result.loading,
      error: result.error,
    };
  }
  return { data: null, loading: result.loading, refreshing: false, error: result.error };
}
