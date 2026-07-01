import { spawn } from 'node:child_process';
import path from 'node:path';

// The decision the authenticated route hands the apply engine. `approver` and
// `decided_at` are set SERVER-SIDE by the route (from the authenticated request) —
// never trusted from the browser.
export interface DecisionInput {
  proposal_id: string;
  agent_name: string;
  decision: 'approve' | 'reject';
  reason?: string;
  approver: string;
  decided_at: string;
}

// The JSON ail.loop.apply_service prints — an ApplyServiceResult. Kept open (outcome
// + passthrough fields) so the route returns it verbatim and the client renders it.
export interface BridgeResult {
  outcome: string;
  [key: string]: unknown;
}

// The seam the route calls to run the (Python) apply engine server-side. Injectable
// so the route is unit-testable with a fake bridge (no subprocess, no live write).
export type ApplyBridge = (input: DecisionInput) => Promise<BridgeResult>;

interface SpawnBridgeOptions {
  /** Python interpreter that has the `ail` package importable. */
  pythonBin?: string;
  /** Extra PYTHONPATH entry so `python -m ail.loop.apply_service` resolves in dev. */
  srcPath?: string;
  /** Hard timeout for the subprocess (ms). */
  timeoutMs?: number;
}

// The default bridge: run `python -m ail.loop.apply_service`, write the decision as
// JSON on stdin, read the ApplyServiceResult JSON from stdout. The engine runs under
// the framework service principal (the deployed app's identity / the dev profile);
// it re-checks the proof + gate and performs the gated apply. The UI only triggers.
//
// Deployment note (docs/LOOP_CONTROLLER.md): the deployed Databricks App image is
// Node-only — the `ail` wheel runs as serverless Jobs — so this subprocess bridge is
// the local-dev/self-hosted transport. In a Node-only image, swap this single seam
// for a Databricks Job trigger (the AppKit `jobs` plugin) that runs the *same*
// ail.loop.apply_service entry; the engine + wiring are unchanged.
export function spawnPythonApplyBridge(options: SpawnBridgeOptions = {}): ApplyBridge {
  const pythonBin = options.pythonBin ?? process.env.AIL_APPLY_PYTHON_BIN ?? 'python3';
  // The app runs from ail-self-optimizer/; the ail package source is ../src.
  const srcPath = options.srcPath ?? process.env.AIL_APPLY_SRC_PATH ?? path.resolve(process.cwd(), '..', 'src');
  const timeoutMs = options.timeoutMs ?? 120_000;

  return (input: DecisionInput) =>
    new Promise<BridgeResult>((resolve, reject) => {
      const existing = process.env.PYTHONPATH;
      const child = spawn(pythonBin, ['-m', 'ail.loop.apply_service'], {
        env: { ...process.env, PYTHONPATH: existing ? `${srcPath}${path.delimiter}${existing}` : srcPath },
      });

      let stdout = '';
      let stderr = '';
      const timer = setTimeout(() => {
        child.kill('SIGKILL');
        reject(new Error(`apply-service timed out after ${timeoutMs}ms`));
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
          reject(new Error(`apply-service exited ${code}: ${stderr.trim() || stdout.trim()}`));
          return;
        }
        try {
          resolve(JSON.parse(stdout) as BridgeResult);
        } catch {
          reject(new Error(`apply-service returned unparseable output: ${stdout.slice(0, 500)}`));
        }
      });

      child.stdin.write(JSON.stringify(input));
      child.stdin.end();
    });
}
