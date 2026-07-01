import { Plugin, toPlugin, type IAppRouter, type PluginManifest } from '@databricks/appkit';
import manifest from './manifest.json';
import { selectApplyBridge, type ApplyBridge, type DecisionInput } from './bridge';

// The minimal HTTP shapes the handler needs — Express's Request/Response satisfy
// these structurally, so the same handler is used by injectRoutes and driven by a
// fake req/res in tests (no server, no subprocess).
export interface DecisionHttpRequest {
  headers: Record<string, string | string[] | undefined>;
  body: unknown;
}
export interface DecisionHttpResponse {
  status(code: number): DecisionHttpResponse;
  json(body: unknown): void;
}

// The authenticated app user, resolved from the platform-injected identity headers
// (docs execution-context): the OBO email (preferred — human-meaningful) then the
// user id. Returns null when neither is present — the request is unauthenticated and
// MUST be refused (fail-closed). Never trusts an approver from the request body.
export function readApprover(req: DecisionHttpRequest): string | null {
  const header = (name: string): string | null => {
    const v = req.headers[name];
    const value = Array.isArray(v) ? v[0] : v;
    return value && value.trim() ? value.trim() : null;
  };
  return header('x-forwarded-email') ?? header('x-forwarded-user');
}

interface DecisionBody {
  proposal_id?: unknown;
  agent_name?: unknown;
  decision?: unknown;
  reason?: unknown;
}

// The authenticated approve/reject write-path. Fail-closed: an unauthenticated
// request is refused (401) before anything runs; a malformed request (missing ids,
// unknown decision, a reject without a reason) is refused (400). Only a well-formed,
// authenticated decision reaches the engine bridge, and the approver is the
// AUTHENTICATED identity (never the request body). The engine itself re-checks the
// proof + gate and performs the gated apply; this route only triggers it.
export async function handleDecision(
  req: DecisionHttpRequest,
  res: DecisionHttpResponse,
  bridge: ApplyBridge
): Promise<void> {
  const approver = readApprover(req);
  if (!approver) {
    res.status(401).json({
      outcome: 'refused',
      refused_reason: 'unauthenticated — no forwarded user identity; refusing to record an anonymous decision',
    });
    return;
  }

  const body = (req.body ?? {}) as DecisionBody;
  const proposal_id = typeof body.proposal_id === 'string' ? body.proposal_id.trim() : '';
  const agent_name = typeof body.agent_name === 'string' ? body.agent_name.trim() : '';
  const decision = body.decision;
  const reason = typeof body.reason === 'string' ? body.reason.trim() : '';

  if (!proposal_id || !agent_name) {
    res.status(400).json({ outcome: 'error', error: 'proposal_id and agent_name are required' });
    return;
  }
  if (decision !== 'approve' && decision !== 'reject') {
    res.status(400).json({ outcome: 'error', error: "decision must be 'approve' or 'reject'" });
    return;
  }
  if (decision === 'reject' && !reason) {
    res.status(400).json({ outcome: 'error', error: 'a reject requires a non-empty reason' });
    return;
  }

  const input: DecisionInput = {
    proposal_id,
    agent_name,
    decision,
    approver, // authenticated identity — the recorded approver
    decided_at: new Date().toISOString(),
    ...(reason ? { reason } : {}),
  };

  try {
    const result = await bridge(input);
    res.status(200).json(result);
  } catch (err) {
    // The bridge (subprocess / job) itself failed — an honest ERROR, never a fake apply.
    res.status(502).json({
      outcome: 'error',
      error: err instanceof Error ? err.message : 'apply engine bridge failed',
    });
  }
}

// The custom AppKit plugin exposing the authenticated write-path. Routes mount under
// /api/approvals/... (server plugin convention); this is the app's FIRST write-path.
// Reads stay two-tier SELECT-only via the analytics plugin; only this route writes.
export class ApprovalsPlugin extends Plugin {
  static manifest = manifest as PluginManifest<'approvals'>;

  // Transport chosen by environment (bridge.ts): the Databricks Job trigger on the
  // deployed Node-only image (AIL_APPLY_TRANSPORT=job or AIL_APPLY_JOB_ID set), else
  // the local subprocess. The route stays bridge-injectable and unchanged.
  private readonly bridge: ApplyBridge = selectApplyBridge();

  injectRoutes(router: IAppRouter): void {
    this.route(router, {
      name: 'decision',
      method: 'post',
      path: '/decision',
      handler: (req, res) => handleDecision(req, res, this.bridge),
    });
  }
}

export const approvals = toPlugin(ApprovalsPlugin);
