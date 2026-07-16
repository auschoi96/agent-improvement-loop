const FORCE_EXIT_AFTER_MS = 8_000;

interface ShutdownProcess {
  exit(code?: number): unknown;
  prependOnceListener(event: 'SIGTERM' | 'SIGINT', listener: () => void): unknown;
}

/**
 * AppKit stops accepting traffic and closes its HTTP server on SIGTERM. Keep a
 * shorter platform-safe backstop so a lingering SSE/keep-alive connection cannot
 * push the process past Databricks Apps' 15-second SIGKILL deadline.
 */
export function installGracefulShutdownBackstop(
  target: ShutdownProcess = process,
  forceExitAfterMs = FORCE_EXIT_AFTER_MS
): void {
  let installed = false;
  const onSignal = () => {
    if (installed) return;
    installed = true;
    const timer = setTimeout(() => target.exit(0), forceExitAfterMs);
    timer.unref();
  };

  target.prependOnceListener('SIGTERM', onSignal);
  target.prependOnceListener('SIGINT', onSignal);
}
