// Labeling-page view logic (L4), kept as pure functions so the request/response
// mapping, progress rendering, and message mapping are unit-testable without a DOM
// (mirrors lib/approvals.ts, lib/onboarding.ts). The LabelingPanel component is a thin
// renderer over these.
//
// Two-tier discipline (docs/OBSERVABILITY_APP.md; the trap caught in the onboarding
// wizard + tutorial): the label FLOOR is NOT modeled here. It is computed by the
// Python engine (ail.readiness.ReadinessThresholds.quality_min_labels) and arrives on
// every dimension as `label_floor`; this file renders whatever number Python sent and
// never hardcodes, invents, or branches on its magnitude. Likewise the registered
// judged dimensions come from the engine (ail.judges.registration) — never invented in
// TS. The one honest fallback is a neutral placeholder when a number is missing.

// A number we could not read from the engine renders as this — never a fabricated digit.
export const MISSING = '—';

// ---------------------------------------------------------------------------
// Server response shapes (the typed JSON ail.labeling.service prints)
// ---------------------------------------------------------------------------

export interface LabelInput {
  kind: string; // 'numeric' | 'pass_fail' | 'free'
  min?: number | null;
  max?: number | null;
  positive?: string | null;
  negative?: string | null;
}

export interface DimensionProgress {
  name: string;
  labels_so_far: number;
  label_floor: number;
  remaining: number;
  complete: boolean;
  input?: LabelInput | null;
  summary: string;
}

export interface TraceTarget {
  trace_id: string;
  request_time?: string | null;
  preview?: string | null;
  labeled: Record<string, boolean>;
}

export interface DimensionsResponse {
  outcome: string;
  experiment_id?: string;
  label_floor?: number;
  dimensions?: DimensionProgress[];
  traces?: TraceTarget[];
  scanned?: number;
  scan_capped?: boolean;
  summary?: string;
  error?: string | null;
  refused_reason?: string | null;
}

export interface LabelResponse {
  outcome: string; // 'labeled' | 'refused' | 'error'
  name?: string;
  value?: unknown;
  labeler?: string;
  labels_so_far?: number | null;
  label_floor?: number | null;
  remaining?: number | null;
  complete?: boolean | null;
  refused_reason?: string | null;
  error?: string | null;
}

export const DIMENSIONS_ENDPOINT = '/api/labeling/dimensions';
export const LABEL_ENDPOINT = '/api/labeling/label';

// ---------------------------------------------------------------------------
// Progress rendering — numbers come from Python, rendered verbatim
// ---------------------------------------------------------------------------

// The "N / FLOOR" fraction for a dimension, rendered VERBATIM from the engine's
// numbers. A missing/non-finite value renders the neutral placeholder — never a
// hardcoded or invented floor. (The two-tier guard the reviewer enforces.)
export function progressLabel(dim: Pick<DimensionProgress, 'labels_so_far' | 'label_floor'>): string {
  const have = Number.isFinite(dim.labels_so_far) ? dim.labels_so_far : null;
  const floor = Number.isFinite(dim.label_floor) ? dim.label_floor : null;
  return `${have ?? MISSING} / ${floor ?? MISSING}`;
}

// A 0..1 completion ratio for a progress bar, clamped. Returns null when the numbers
// are missing (the bar renders empty rather than a fabricated fill). Never scales a
// magnitude the engine did not send.
export function progressRatio(dim: Pick<DimensionProgress, 'labels_so_far' | 'label_floor'>): number | null {
  const { labels_so_far: have, label_floor: floor } = dim;
  if (!Number.isFinite(have) || !Number.isFinite(floor) || floor <= 0) return null;
  return Math.max(0, Math.min(1, have / floor));
}

// ---------------------------------------------------------------------------
// Label request — the labeler is NOT included (resolved server-side)
// ---------------------------------------------------------------------------

export interface LabelRequest {
  experiment_id: string;
  trace_id: string;
  name: string;
  value: unknown;
  reason?: string;
}

// Build the POST body for the authenticated label route. The labeler is NOT included:
// it is resolved server-side from the platform identity headers (never trusted from
// the browser). Throws (fail-closed) when the value is missing so the UI cannot submit
// an empty label. `rationale` is optional context (the same one-line evidence the
// judge is asked to produce).
export function buildLabelRequest(
  target: { experiment_id: string; trace_id: string },
  name: string,
  value: unknown,
  rationale?: string
): LabelRequest {
  if (!name.trim()) throw new Error('Pick a dimension to label.');
  if (value === undefined || value === null || (typeof value === 'string' && !value.trim())) {
    throw new Error('Enter a value before submitting.');
  }
  return {
    experiment_id: target.experiment_id,
    trace_id: target.trace_id,
    name: name.trim(),
    value: typeof value === 'string' ? value.trim() : value,
    ...(rationale && rationale.trim() ? { reason: rationale.trim() } : {}),
  };
}

export type LabelTone = 'success' | 'error' | 'warning';

// Map a label outcome to an honest message + tone. A refusal surfaces WHY (e.g. an
// unknown judge name); an error surfaces the failure. Only a real `labeled` outcome is
// a success — a fabricated success is never shown.
export function labelMessage(resp: LabelResponse): { tone: LabelTone; text: string } {
  switch (resp.outcome) {
    case 'labeled': {
      const progress =
        resp.labels_so_far != null && resp.label_floor != null ? ` (${resp.labels_so_far} / ${resp.label_floor})` : '';
      return { tone: 'success', text: `Labeled ${resp.name ?? ''}${progress}.`.replace('  ', ' ') };
    }
    case 'refused':
      return {
        tone: 'error',
        text: `Refused — ${resp.refused_reason ?? 'the label was rejected'}. Nothing was written.`,
      };
    default:
      return { tone: 'error', text: `Error — ${resp.error ?? 'the label could not be written'}.` };
  }
}

// The message for a non-ok HTTP status with no engine body (auth/transport failures).
export function httpErrorMessage(status: number): string {
  return status === 401 ? 'Not authenticated — sign in to label traces.' : `Request failed (${status}).`;
}

// The dimensions a trace still needs, given the registered dimension order. Pure so the
// row rendering is testable — reads the trace's labeled map, never recomputes coverage.
export function missingDimensions(target: TraceTarget, order: readonly string[]): string[] {
  return order.filter((name) => !target.labeled[name]);
}
