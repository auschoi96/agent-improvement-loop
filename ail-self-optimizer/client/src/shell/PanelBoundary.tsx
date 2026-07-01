import { type ReactNode } from 'react';
import { Alert, AlertDescription, AlertTitle } from '@databricks/appkit-ui/react';
import { AlertTriangle } from 'lucide-react';
import { ErrorBoundary } from '../ErrorBoundary';

// A per-panel error boundary. Wrapping each page's panel means one panel throwing
// renders a compact inline Alert in that panel's place instead of blanking the whole
// app (which the top-level ErrorBoundary would otherwise do). Reuses ErrorBoundary's
// renderFallback hook so there is a single boundary implementation.
export function PanelBoundary({
  title = 'This panel failed to render',
  children,
}: {
  title?: string;
  children: ReactNode;
}) {
  return (
    <ErrorBoundary
      renderFallback={(error) => (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>{title}</AlertTitle>
          <AlertDescription>
            {error?.message ?? 'An unexpected error occurred.'} Other sections are unaffected — try reloading this view.
          </AlertDescription>
        </Alert>
      )}
    >
      {children}
    </ErrorBoundary>
  );
}
