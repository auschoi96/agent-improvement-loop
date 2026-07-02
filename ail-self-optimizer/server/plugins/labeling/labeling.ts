import { Plugin, toPlugin, type IAppRouter, type PluginManifest } from '@databricks/appkit';
import manifest from './manifest.json';
import { selectLabelingBridge, type LabelingAction, type LabelingBridge } from './bridge';

// The minimal HTTP shapes the handlers need — Express's Request/Response satisfy
// these structurally, so the same handlers are used by injectRoutes and driven by a
// fake req/res in tests (no server, no subprocess). Mirrors the onboarding plugin.
export interface LabelingHttpRequest {
  headers: Record<string, string | string[] | undefined>;
  body: unknown;
}
export interface LabelingHttpResponse {
  status(code: number): LabelingHttpResponse;
  json(body: unknown): void;
}

// The authenticated app user, resolved from the platform-injected identity headers
// (mirrors the approvals/onboarding write-paths): the OBO email (preferred —
// human-meaningful) then the user id. Returns null when neither is present — the
// request is unauthenticated and MUST be refused (fail-closed). This identity is the
// label's HUMAN source; a `labeler` in the request body is never read or trusted.
export function readLabeler(req: LabelingHttpRequest): string | null {
  const header = (name: string): string | null => {
    const v = req.headers[name];
    const value = Array.isArray(v) ? v[0] : v;
    return value && value.trim() ? value.trim() : null;
  };
  return header('x-forwarded-email') ?? header('x-forwarded-user');
}

function unauthorized(res: LabelingHttpResponse): void {
  res.status(401).json({
    outcome: 'refused',
    refused_reason: 'unauthenticated — no forwarded user identity; sign in to label traces',
  });
}

function badRequest(res: LabelingHttpResponse, error: string): void {
  res.status(400).json({ outcome: 'error', error });
}

// Run one authenticated action through the engine bridge. A bridge failure (the
// subprocess itself failed) is surfaced as an honest error (502) — never a fabricated
// success, exactly as the approvals/onboarding routes treat an engine failure. A
// decision-level result (including the engine's own `refused`/`error`) comes back at
// 200 and is returned verbatim.
async function dispatch(res: LabelingHttpResponse, bridge: LabelingBridge, action: LabelingAction): Promise<void> {
  try {
    const result = await bridge(action);
    res.status(200).json(result);
  } catch (err) {
    res.status(502).json({
      outcome: 'error',
      error: err instanceof Error ? err.message : 'labeling engine bridge failed',
    });
  }
}

function stringField(body: Record<string, unknown>, key: string): string {
  const v = body[key];
  return typeof v === 'string' ? v.trim() : '';
}

// Read side: the registered judged dimensions for an experiment + each one's label
// progress toward the floor + the traces still needing a label. The registered-judge
// set and the floor come from the Python engine (ail.labeling.service → ail.judges /
// ail.readiness), so the app never invents a dimension or hardcodes the floor in TS.
export async function handleDimensions(
  req: LabelingHttpRequest,
  res: LabelingHttpResponse,
  bridge: LabelingBridge
): Promise<void> {
  const labeler = readLabeler(req);
  if (!labeler) return unauthorized(res);
  const body = (req.body ?? {}) as Record<string, unknown>;
  const experiment_id = stringField(body, 'experiment_id');
  if (!experiment_id) return badRequest(res, 'experiment_id is required');
  await dispatch(res, bridge, { action: 'dimensions', actor: labeler, experiment_id });
}

// Write side: record one HUMAN label on a trace, named for a registered judge. The
// labeler is the AUTHENTICATED identity; a spoofed `labeler` in the body is ignored
// (the action carries `actor`, resolved here). The engine reuses record_label and
// refuses a name that is not a registered judge — fail-closed, never a fake label.
export async function handleLabel(
  req: LabelingHttpRequest,
  res: LabelingHttpResponse,
  bridge: LabelingBridge
): Promise<void> {
  const labeler = readLabeler(req);
  if (!labeler) return unauthorized(res);
  const body = (req.body ?? {}) as Record<string, unknown>;
  const experiment_id = stringField(body, 'experiment_id');
  const trace_id = stringField(body, 'trace_id');
  const name = stringField(body, 'name');
  const value = body.value;
  if (!experiment_id || !trace_id || !name) {
    return badRequest(res, 'experiment_id, trace_id and name are required');
  }
  if (value === undefined || value === null || (typeof value === 'string' && !value.trim())) {
    return badRequest(res, 'a label value is required');
  }
  const rationale = stringField(body, 'rationale');
  await dispatch(res, bridge, {
    action: 'label',
    actor: labeler,
    experiment_id,
    trace_id,
    name,
    value,
    ...(rationale ? { rationale } : {}),
  });
}

// The custom AppKit plugin exposing the labeling read + write paths. Routes mount
// under /api/labeling/... (server plugin convention). Reads of L0/L2 stay two-tier
// SELECT-only via the analytics plugin; these authenticated routes read/write the
// permission-sensitive MLflow experiment behind the same fail-closed engine bridge.
export class LabelingPlugin extends Plugin {
  static manifest = manifest as PluginManifest<'labeling'>;

  private readonly bridge: LabelingBridge = selectLabelingBridge();

  injectRoutes(router: IAppRouter): void {
    this.route(router, {
      name: 'dimensions',
      method: 'post',
      path: '/dimensions',
      handler: (req, res) => handleDimensions(req, res, this.bridge),
    });
    this.route(router, {
      name: 'label',
      method: 'post',
      path: '/label',
      handler: (req, res) => handleLabel(req, res, this.bridge),
    });
  }
}

export const labeling = toPlugin(LabelingPlugin);
