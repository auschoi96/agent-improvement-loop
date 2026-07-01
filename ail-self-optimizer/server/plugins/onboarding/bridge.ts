import { spawn } from 'node:child_process';
import path from 'node:path';

// The JSON action the authenticated onboarding route hands the (Python) engine.
// `actor` is set SERVER-SIDE by the route (from the authenticated request) — it is
// never trusted from the browser (mirrors the approvals write-path's `approver`).
export interface OnboardingAction {
  action: 'requirements' | 'validate_experiment' | 'create_experiment' | 'register_agent';
  actor: string;
  goals?: string[];
  experiment_id?: string;
  name?: string;
  agent_name?: string;
}

// The JSON ail.onboarding.service prints — a typed onboarding result. Kept open
// (outcome + passthrough fields) so the route returns it verbatim and the client
// wizard renders it (never re-deriving the goal/gate/registry facts in TS).
export interface OnboardingResult {
  outcome: string;
  [key: string]: unknown;
}

// The seam the route calls to run the (Python) onboarding engine server-side.
// Injectable so the route is unit-testable with a fake bridge (no subprocess, no
// live write) — exactly as ail.loop.apply_service's ApplyBridge seam is.
export type OnboardingBridge = (input: OnboardingAction) => Promise<OnboardingResult>;

interface SpawnBridgeOptions {
  /** Python interpreter that has the `ail` package importable. */
  pythonBin?: string;
  /** Extra PYTHONPATH entry so `python -m ail.onboarding.service` resolves in dev. */
  srcPath?: string;
  /** Hard timeout for the subprocess (ms). */
  timeoutMs?: number;
}

// The default bridge: run `python -m ail.onboarding.service`, write the JSON action
// on stdin, read the typed result JSON from stdout. The engine runs under the app's
// identity (the framework service principal / the dev profile); it performs the
// permission-sensitive read/writes and returns an HONEST result — an experiment is
// only "created" when MLflow created it, an agent only "registered" when the
// registry write succeeded (see src/ail/onboarding/service.py).
//
// Deployment note (mirrors docs/LOOP_CONTROLLER.md for approvals): the deployed
// Databricks App image is Node-only — the `ail` wheel runs as serverless Jobs — so
// this subprocess bridge is the local-dev / self-hosted transport. A Databricks
// Job-trigger transport for the deployed image (the analogue of the approvals
// `jobTriggerApplyBridge`) is a documented FOLLOW-ON: it needs a wheel-task entry
// point + a job resource, which are out of scope for this slice (new plugin +
// client only). The `OnboardingBridge` seam is unchanged when that lands.
export function spawnPythonOnboardingBridge(options: SpawnBridgeOptions = {}): OnboardingBridge {
  const pythonBin =
    options.pythonBin ?? process.env.AIL_ONBOARDING_PYTHON_BIN ?? process.env.AIL_APPLY_PYTHON_BIN ?? 'python3';
  // The app runs from ail-self-optimizer/; the ail package source is ../src.
  const srcPath =
    options.srcPath ??
    process.env.AIL_ONBOARDING_SRC_PATH ??
    process.env.AIL_APPLY_SRC_PATH ??
    path.resolve(process.cwd(), '..', 'src');
  const timeoutMs = options.timeoutMs ?? 120_000;

  return (input: OnboardingAction) =>
    new Promise<OnboardingResult>((resolve, reject) => {
      const existing = process.env.PYTHONPATH;
      const child = spawn(pythonBin, ['-m', 'ail.onboarding.service'], {
        env: { ...process.env, PYTHONPATH: existing ? `${srcPath}${path.delimiter}${existing}` : srcPath },
      });

      let stdout = '';
      let stderr = '';
      const timer = setTimeout(() => {
        child.kill('SIGKILL');
        reject(new Error(`onboarding-service timed out after ${timeoutMs}ms`));
      }, timeoutMs);

      child.stdout.on('data', (d: Buffer) => (stdout += d.toString()));
      child.stderr.on('data', (d: Buffer) => (stderr += d.toString()));
      child.on('error', (err) => {
        clearTimeout(timer);
        reject(err);
      });
      child.on('close', (code) => {
        clearTimeout(timer);
        if (code !== 0) {
          reject(new Error(`onboarding-service exited ${code}: ${stderr.trim() || stdout.trim()}`));
          return;
        }
        try {
          resolve(JSON.parse(stdout) as OnboardingResult);
        } catch {
          reject(new Error(`onboarding-service returned unparseable output: ${stdout.slice(0, 500)}`));
        }
      });

      child.stdin.write(JSON.stringify(input));
      child.stdin.end();
    });
}

// Transport selection — the subprocess transport for this slice. Kept as a function
// (not a bare export) so a future Node-only Job transport can be selected by env
// here without touching the route, exactly as approvals' selectApplyBridge does.
export function selectOnboardingBridge(): OnboardingBridge {
  return spawnPythonOnboardingBridge();
}
