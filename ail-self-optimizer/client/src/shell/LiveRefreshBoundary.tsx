import { useEffect, useState, type ReactNode } from 'react';
import { LiveRefreshContext } from './live-refresh-context';

const DEFAULT_INTERVAL_MS = 30_000;
const INITIAL_REFRESH_DELAY_MS = 60_000;
export function LiveRefreshBoundary({
  children,
  intervalMs = DEFAULT_INTERVAL_MS,
}: {
  children: ReactNode;
  intervalMs?: number;
}) {
  const [revision, setRevision] = useState(0);

  useEffect(() => {
    let nextRefresh = Date.now() + Math.max(intervalMs, INITIAL_REFRESH_DELAY_MS);
    const timer = window.setInterval(
      () => {
        if (document.visibilityState !== 'visible' || Date.now() < nextRefresh) return;
        nextRefresh = Date.now() + intervalMs;
        setRevision((value) => value + 1);
      },
      Math.min(intervalMs, 1_000)
    );
    return () => window.clearInterval(timer);
  }, [intervalMs]);

  return <LiveRefreshContext.Provider value={revision}>{children}</LiveRefreshContext.Provider>;
}
