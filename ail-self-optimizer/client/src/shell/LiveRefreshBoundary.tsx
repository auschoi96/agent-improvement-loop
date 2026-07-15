import { Fragment, useEffect, useRef, useState, type ReactNode } from 'react';

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
  const interacting = useRef(false);
  const nextRefresh = useRef(0);

  useEffect(() => {
    nextRefresh.current = Date.now() + Math.max(intervalMs, INITIAL_REFRESH_DELAY_MS);
    const timer = window.setInterval(() => {
      if (document.visibilityState !== 'visible' || interacting.current) return;
      if (Date.now() < nextRefresh.current) return;
      nextRefresh.current = Date.now() + intervalMs;
      setRevision((value) => value + 1);
    }, Math.min(intervalMs, 1_000));
    return () => window.clearInterval(timer);
  }, [intervalMs]);

  return (
    <div
      onFocusCapture={() => {
        interacting.current = true;
      }}
      onBlurCapture={() => {
        interacting.current = false;
        nextRefresh.current = Date.now() + intervalMs;
      }}
    >
      <Fragment key={revision}>{children}</Fragment>
    </div>
  );
}
