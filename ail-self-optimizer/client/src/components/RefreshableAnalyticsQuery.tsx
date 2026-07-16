import { useCallback, useEffect, useState, type ReactNode } from 'react';
import { useAnalyticsQuery, type QueryRegistry } from '@databricks/appkit-ui/react';
import { useLiveRefreshRevision } from '../shell/live-refresh-context';
import { reconcileRefreshState, type QueryResult, type RefreshableQueryState } from '../lib/refresh-state';

export type { RefreshableQueryState } from '../lib/refresh-state';

export type AppQueryKey = keyof {
  [K in keyof QueryRegistry as string extends K ? never : K]: QueryRegistry[K];
};

function QuerySubscriber<K extends AppQueryKey>({
  queryKey,
  parameters,
  onResult,
}: {
  queryKey: K;
  parameters: QueryRegistry[K]['parameters'];
  onResult: (result: QueryResult<QueryRegistry[K]['result']>) => void;
}) {
  // QueryRegistry[K] already ties the key to its exact parameter type. AppKit's
  // exported hook loses that relationship for a generic K because its internal
  // literal-key helper is not exported, so narrow only at this boundary.
  const { data, loading, error } = useAnalyticsQuery<QueryRegistry[K]['result'], K>(queryKey, parameters as never);
  useEffect(() => {
    onResult({ data, loading, error });
  }, [data, loading, error, onResult]);
  return null;
}

/**
 * Remount only the AppKit query subscriber on a refresh revision while retaining
 * the last successful result in the parent. Consumers never flash back to an
 * empty skeleton merely because a background replacement query started.
 */
function RefreshableQueryInstance<K extends AppQueryKey>({
  queryKey,
  parameters,
  children,
}: {
  queryKey: K;
  parameters: QueryRegistry[K]['parameters'];
  children: (state: RefreshableQueryState<QueryRegistry[K]['result']>) => ReactNode;
}) {
  const revision = useLiveRefreshRevision();
  const [state, setState] = useState<RefreshableQueryState<QueryRegistry[K]['result']>>({
    data: null,
    loading: true,
    refreshing: false,
    error: null,
  });

  const receive = useCallback((result: QueryResult<QueryRegistry[K]['result']>) => {
    setState((previous) => reconcileRefreshState(previous, result));
  }, []);

  return (
    <>
      <QuerySubscriber key={`${queryKey}-${revision}`} queryKey={queryKey} parameters={parameters} onResult={receive} />
      {children(state)}
    </>
  );
}

export function RefreshableAnalyticsQuery<K extends AppQueryKey>({
  queryKey,
  parameters,
  children,
}: {
  queryKey: K;
  parameters: QueryRegistry[K]['parameters'];
  children: (state: RefreshableQueryState<QueryRegistry[K]['result']>) => ReactNode;
}) {
  // A parameter change means a different data scope (usually a different agent).
  // Key the retained-state owner by the serialized AppKit markers so stale rows
  // cannot render under the new scope, while refresh revisions still retain data.
  const scopeKey = `${String(queryKey)}:${JSON.stringify(parameters)}`;
  return (
    <RefreshableQueryInstance key={scopeKey} queryKey={queryKey} parameters={parameters}>
      {children}
    </RefreshableQueryInstance>
  );
}
