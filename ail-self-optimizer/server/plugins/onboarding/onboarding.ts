import { Plugin, toPlugin, type IAppRouter, type PluginManifest } from '@databricks/appkit';
import manifest from './manifest.json';
import { selectOnboardingBridge, type OnboardingAction, type OnboardingBridge } from './bridge';

// The minimal HTTP shapes the handlers need — Express's Request/Response satisfy
// these structurally, so the same handlers are used by injectRoutes and driven by a
// fake req/res in tests (no server, no subprocess). Mirrors the approvals plugin.
export interface OnboardingHttpRequest {
  headers: Record<string, string | string[] | undefined>;
  body: unknown;
}
export interface OnboardingHttpResponse {
  status(code: number): OnboardingHttpResponse;
  json(body: unknown): void;
}

// The authenticated app user, resolved from the platform-injected identity headers
// (docs execution-context): the OBO email (preferred — human-meaningful) then the
// user id. Returns null when neither is present — the request is unauthenticated and
// MUST be refused (fail-closed). Never trusts an actor from the request body.
export function readActor(req: OnboardingHttpRequest): string | null {
  const header = (name: string): string | null => {
    const v = req.headers[name];
    const value = Array.isArray(v) ? v[0] : v;
    return value && value.trim() ? value.trim() : null;
  };
  return header('x-forwarded-email') ?? header('x-forwarded-user');
}

function unauthorized(res: OnboardingHttpResponse): void {
  res.status(401).json({
    outcome: 'refused',
    refused_reason: 'unauthenticated — no forwarded user identity; sign in to onboard an agent',
  });
}

function badRequest(res: OnboardingHttpResponse, error: string): void {
  res.status(400).json({ outcome: 'error', error });
}

// Run one authenticated action through the engine bridge. A bridge failure (the
// subprocess / job itself failed) is surfaced as an honest error (502) — never a
// fabricated success, exactly as the approvals route treats an engine failure.
async function dispatch(
  res: OnboardingHttpResponse,
  bridge: OnboardingBridge,
  action: OnboardingAction
): Promise<void> {
  try {
    const result = await bridge(action);
    res.status(200).json(result);
  } catch (err) {
    res.status(502).json({
      outcome: 'error',
      error: err instanceof Error ? err.message : 'onboarding engine bridge failed',
    });
  }
}

function stringField(body: Record<string, unknown>, key: string): string {
  const v = body[key];
  return typeof v === 'string' ? v.trim() : '';
}

function goalsField(body: Record<string, unknown>): string[] {
  const v = body.goals;
  return Array.isArray(v) ? v.filter((g): g is string => typeof g === 'string') : [];
}

// The human's explicit objective target (a signed relative fraction). Only a finite
// number is forwarded; anything else is omitted so the engine refuses honestly
// ("set/acknowledge the target first") rather than being handed a fabricated value.
function numberField(body: Record<string, unknown>, key: string): number | undefined {
  const v = body[key];
  return typeof v === 'number' && Number.isFinite(v) ? v : undefined;
}

// Page 2/3 source: the fixed goal catalog + the data gates a selection needs. The
// gate/floor facts come from the Python engine (ail.onboarding.goals), so the app
// never re-derives readiness thresholds in TS.
export async function handleRequirements(
  req: OnboardingHttpRequest,
  res: OnboardingHttpResponse,
  bridge: OnboardingBridge
): Promise<void> {
  const actor = readActor(req);
  if (!actor) return unauthorized(res);
  const body = (req.body ?? {}) as Record<string, unknown>;
  await dispatch(res, bridge, { action: 'requirements', actor, goals: goalsField(body) });
}

// Page 1: validate an experiment is fresh (empty of prior AIL state). Fail-closed —
// the engine never reports "fresh" it could not verify.
export async function handleValidateExperiment(
  req: OnboardingHttpRequest,
  res: OnboardingHttpResponse,
  bridge: OnboardingBridge
): Promise<void> {
  const actor = readActor(req);
  if (!actor) return unauthorized(res);
  const body = (req.body ?? {}) as Record<string, unknown>;
  const experiment_id = stringField(body, 'experiment_id');
  if (!experiment_id) return badRequest(res, 'experiment_id is required');
  await dispatch(res, bridge, { action: 'validate_experiment', actor, experiment_id });
}

// Page 1: create a fresh MLflow experiment (a write). Fail-closed — a create the SP
// is not authorized for returns an honest error + the documented prerequisite.
export async function handleCreateExperiment(
  req: OnboardingHttpRequest,
  res: OnboardingHttpResponse,
  bridge: OnboardingBridge
): Promise<void> {
  const actor = readActor(req);
  if (!actor) return unauthorized(res);
  const body = (req.body ?? {}) as Record<string, unknown>;
  const name = stringField(body, 'name');
  if (!name) return badRequest(res, 'an experiment name is required');
  await dispatch(res, bridge, { action: 'create_experiment', actor, name });
}

// Page 4: register the agent by reusing ail.publish_versions (server-side). The
// actor is the AUTHENTICATED identity; a spoofed actor in the body is ignored.
export async function handleRegisterAgent(
  req: OnboardingHttpRequest,
  res: OnboardingHttpResponse,
  bridge: OnboardingBridge
): Promise<void> {
  const actor = readActor(req);
  if (!actor) return unauthorized(res);
  const body = (req.body ?? {}) as Record<string, unknown>;
  const agent_name = stringField(body, 'agent_name');
  const experiment_id = stringField(body, 'experiment_id');
  const goals = goalsField(body);
  if (!agent_name || !experiment_id) {
    return badRequest(res, 'agent_name and experiment_id are required');
  }
  if (goals.length === 0) return badRequest(res, 'select at least one goal');
  await dispatch(res, bridge, { action: 'register_agent', actor, agent_name, experiment_id, goals });
}

// Free-form requirements PREVIEW: hand the raw requirements blob to the engine,
// which extracts + routes + composes and returns the plan for human review. A pure
// proposal — authors nothing, persists nothing. The engine owns every routing /
// threshold / target fact (two-tier); the client renders the response verbatim. The
// actor is the AUTHENTICATED identity; a body actor is never trusted.
export async function handlePreviewRequirements(
  req: OnboardingHttpRequest,
  res: OnboardingHttpResponse,
  bridge: OnboardingBridge
): Promise<void> {
  const actor = readActor(req);
  if (!actor) return unauthorized(res);
  const body = (req.body ?? {}) as Record<string, unknown>;
  const requirements_text = stringField(body, 'requirements_text');
  if (!requirements_text) return badRequest(res, 'requirements_text is required');
  const agent_name = stringField(body, 'agent_name');
  const cohort = stringField(body, 'cohort');
  await dispatch(res, bridge, { action: 'preview_requirements', actor, requirements_text, agent_name, cohort });
}

// Free-form requirements CONFIRM (a write): re-derive the plan, apply the human's
// explicit objective target, author the judges + persist the goal. Fail-closed in
// the engine (nothing authored/persisted unless the plan is confirmed AND the goal
// is human_confirmed; a missing target is refused). The actor is server-set.
export async function handleConfirmRequirements(
  req: OnboardingHttpRequest,
  res: OnboardingHttpResponse,
  bridge: OnboardingBridge
): Promise<void> {
  const actor = readActor(req);
  if (!actor) return unauthorized(res);
  const body = (req.body ?? {}) as Record<string, unknown>;
  const requirements_text = stringField(body, 'requirements_text');
  const agent_name = stringField(body, 'agent_name');
  const experiment_id = stringField(body, 'experiment_id');
  if (!requirements_text || !agent_name || !experiment_id) {
    return badRequest(res, 'requirements_text, agent_name, and experiment_id are required');
  }
  const objective_target = numberField(body, 'objective_target');
  const cohort = stringField(body, 'cohort');
  await dispatch(res, bridge, {
    action: 'confirm_requirements',
    actor,
    requirements_text,
    agent_name,
    experiment_id,
    objective_target,
    cohort,
  });
}

// The custom AppKit plugin exposing the onboarding write-path. Routes mount under
// /api/onboarding/... (server plugin convention). Reads stay two-tier SELECT-only
// via the analytics plugin; only these authenticated routes write / read the
// permission-sensitive workspace, all behind the same fail-closed engine.
export class OnboardingPlugin extends Plugin {
  static manifest = manifest as PluginManifest<'onboarding'>;

  private readonly bridge: OnboardingBridge = selectOnboardingBridge();

  injectRoutes(router: IAppRouter): void {
    this.route(router, {
      name: 'requirements',
      method: 'post',
      path: '/requirements',
      handler: (req, res) => handleRequirements(req, res, this.bridge),
    });
    this.route(router, {
      name: 'validate-experiment',
      method: 'post',
      path: '/experiment/validate',
      handler: (req, res) => handleValidateExperiment(req, res, this.bridge),
    });
    this.route(router, {
      name: 'create-experiment',
      method: 'post',
      path: '/experiment/create',
      handler: (req, res) => handleCreateExperiment(req, res, this.bridge),
    });
    this.route(router, {
      name: 'register-agent',
      method: 'post',
      path: '/register',
      handler: (req, res) => handleRegisterAgent(req, res, this.bridge),
    });
    this.route(router, {
      name: 'preview-requirements',
      method: 'post',
      path: '/requirements/preview',
      handler: (req, res) => handlePreviewRequirements(req, res, this.bridge),
    });
    this.route(router, {
      name: 'confirm-requirements',
      method: 'post',
      path: '/requirements/confirm',
      handler: (req, res) => handleConfirmRequirements(req, res, this.bridge),
    });
  }
}

export const onboarding = toPlugin(OnboardingPlugin);
