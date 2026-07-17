import { spawn } from 'node:child_process';
import path from 'node:path';

// The opt-in Tier-2 "verify on my suite" request the authenticated route hands the
// companion. `requested_by` and `requested_at` are set SERVER-SIDE by the route (from
// the authenticated request) — never trusted from the browser, exactly like the
// approve/reject DecisionInput.
export interface VerifyInput {
  proposal_id: string;
  agent_name: string;
  requested_by: string;
  requested_at: string;
}

// The JSON `ail.loop.verify_service` prints — a VerifyRequestResult. Kept open (outcome
// + passthrough fields) so the route returns it verbatim and the client renders it.
export interface VerifyBridgeResult {
  outcome: string;
  [key: string]: unknown;
}

// The seam the route calls to run the (Python) verify-request engine server-side.
// Injectable so the route is unit-testable with a fake bridge (no subprocess, no live
// write) — the same pattern as ApplyBridge.
export type VerifyBridge = (input: VerifyInput) => Promise<VerifyBridgeResult>;

interface SpawnVerifyBridgeOptions {
  /** Python interpreter that has the `ail` package importable. */
  pythonBin?: string;
  /** Extra PYTHONPATH entry so `python -m ail.loop.verify_service` resolves in dev. */
  srcPath?: string;
  /** Hard timeout for the subprocess (ms). */
  timeoutMs?: number;
}

// The default bridge: run `python -m ail.loop.verify_service`, write the request as
// JSON on stdin, read the VerifyRequestResult JSON from stdout. This engine does NOT
// prove anything and applies NOTHING — it only flags the pending proposal as
// verify-requested in UC (the deployer's companion poll loop later runs the frozen-
// suite prover and writes the result back). This mirrors `spawnPythonApplyBridge`: it
// is the local-dev / self-hosted transport where the `ail` wheel is importable.
//
// Deployment note (docs/LOOP_CONTROLLER.md): the deployed Databricks App image is
// Node-only, so a subprocess bridge cannot run there. The deployed (Job-trigger)
// transport mirrors `jobTriggerApplyBridge`; it is deferred (see `selectVerifyBridge`),
// which fails CLOSED with an honest error rather than silently dropping the request.
export function spawnPythonVerifyBridge(options: SpawnVerifyBridgeOptions = {}): VerifyBridge {
  const pythonBin = options.pythonBin ?? process.env.AIL_APPLY_PYTHON_BIN ?? 'python3';
  // The app runs from ail-self-optimizer/; the ail package source is ../src.
  const srcPath = options.srcPath ?? process.env.AIL_APPLY_SRC_PATH ?? path.resolve(process.cwd(), '..', 'src');
  const timeoutMs = options.timeoutMs ?? 120_000;

  return (input: VerifyInput) =>
    new Promise<VerifyBridgeResult>((resolve, reject) => {
      const existing = process.env.PYTHONPATH;
      const child = spawn(pythonBin, ['-m', 'ail.loop.verify_service'], {
        env: {
          ...process.env,
          PYTHONPATH: existing ? `${srcPath}${path.delimiter}${existing}` : srcPath,
        },
      });

      let stdout = '';
      let stderr = '';
      const timer = setTimeout(() => {
        child.kill('SIGKILL');
        reject(new Error(`verify-service timed out after ${timeoutMs}ms`));
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
          reject(new Error(`verify-service exited ${code}: ${stderr.trim() || stdout.trim()}`));
          return;
        }
        try {
          resolve(JSON.parse(stdout) as VerifyBridgeResult);
        } catch {
          reject(new Error(`verify-service returned unparseable output: ${stdout.slice(0, 500)}`));
        }
      });

      child.stdin.write(JSON.stringify(input));
      child.stdin.end();
    });
}

// ---------------------------------------------------------------------------
// Transport selection — env-driven, mirroring resolveApplyTransport.
// ---------------------------------------------------------------------------
//
// Env contract:
//   AIL_VERIFY_TRANSPORT=job (or) AIL_VERIFY_JOB_ID set -> Databricks Job trigger
//   otherwise                                            -> local subprocess (default)
export type VerifyTransport = 'job' | 'subprocess';

export function resolveVerifyTransport(env: NodeJS.ProcessEnv = process.env): VerifyTransport {
  if (env.AIL_VERIFY_TRANSPORT === 'job' || (env.AIL_VERIFY_JOB_ID && env.AIL_VERIFY_JOB_ID.trim())) {
    return 'job';
  }
  return 'subprocess';
}

// The deployed (Node-only) Job-trigger transport is not yet wired for verify. Rather
// than silently drop the request or fabricate a "requested", this bridge fails CLOSED:
// it throws an honest error the route surfaces as outcome:"error". The template is
// `jobTriggerApplyBridge` in bridge.ts (trigger a pre-deployed wheel job that runs the
// SAME `ail.loop.verify_service`); wiring the job + bundle is the deploy lane's follow-up.
export function deferredJobVerifyBridge(): VerifyBridge {
  return () =>
    Promise.reject(
      new Error(
        'verify job transport not yet wired for the deployed (Node-only) image — run the ' +
          'companion locally (subprocess transport) or wire the ail-verify-job (mirrors ' +
          'jobTriggerApplyBridge). Failing closed: no verify was requested.'
      )
    );
}

export function selectVerifyBridge(env: NodeJS.ProcessEnv = process.env): VerifyBridge {
  return resolveVerifyTransport(env) === 'job' ? deferredJobVerifyBridge() : spawnPythonVerifyBridge();
}
