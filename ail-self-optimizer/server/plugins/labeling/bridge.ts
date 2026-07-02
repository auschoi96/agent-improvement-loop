import { spawn } from 'node:child_process';
import path from 'node:path';

// The JSON action the authenticated labeling route hands the (Python) engine.
// `actor` is set SERVER-SIDE by the route (from the authenticated request) — it is
// never trusted from the browser (mirrors the approvals write-path's `approver` and
// the onboarding write-path's `actor`). The engine uses it as the label's HUMAN
// source; any `labeler` in the browser payload is ignored.
export interface LabelingAction {
  action: 'dimensions' | 'label';
  actor: string;
  experiment_id: string;
  // `label` only:
  trace_id?: string;
  name?: string; // the dimension == the registered judge name (the alignment key)
  value?: unknown; // 1–5 / 'pass'|'fail' / free-form — the human's grade
  rationale?: string;
}

// The JSON ail.labeling.service prints — a typed labeling result. Kept open
// (outcome + passthrough fields) so the route returns it verbatim and the client
// renders it (never re-deriving the registered judges or the label floor in TS).
export interface LabelingResult {
  outcome: string;
  [key: string]: unknown;
}

// The seam the route calls to run the (Python) labeling engine server-side.
// Injectable so the route is unit-testable with a fake bridge (no subprocess, no
// live write) — exactly as the onboarding/apply bridges are.
export type LabelingBridge = (input: LabelingAction) => Promise<LabelingResult>;

interface SpawnBridgeOptions {
  /** Python interpreter that has the `ail` package importable. */
  pythonBin?: string;
  /** Extra PYTHONPATH entry so `python -m ail.labeling.service` resolves in dev. */
  srcPath?: string;
  /** Hard timeout for the subprocess (ms). */
  timeoutMs?: number;
}

// The default bridge: run `python -m ail.labeling.service`, write the JSON action on
// stdin, read the typed result JSON from stdout. The engine runs under the app's
// identity (the framework service principal / the dev profile); it performs the
// permission-sensitive MLflow read/write and returns an HONEST result — a trace is
// only "labeled" when mlflow.log_feedback succeeded (see src/ail/labeling/service.py).
// This is grounded, not guessed: it is the SAME transport the onboarding plugin uses
// for its MLflow writes (a Python subprocess over the MLflow SDK), reusing the L1/L2
// helpers so the sacred name-match write is never reimplemented in TypeScript.
//
// Deployment note (mirrors the onboarding/approvals bridges): the deployed Databricks
// App image is Node-only — the `ail` wheel runs as serverless Jobs — so this
// subprocess bridge is the local-dev / self-hosted transport. A Databricks
// Job-trigger transport (the analogue of the approvals `jobTriggerApplyBridge`) is a
// documented FOLLOW-ON; the `LabelingBridge` seam is unchanged when it lands.
export function spawnPythonLabelingBridge(options: SpawnBridgeOptions = {}): LabelingBridge {
  const pythonBin =
    options.pythonBin ??
    process.env.AIL_LABELING_PYTHON_BIN ??
    process.env.AIL_ONBOARDING_PYTHON_BIN ??
    process.env.AIL_APPLY_PYTHON_BIN ??
    'python3';
  // The app runs from ail-self-optimizer/; the ail package source is ../src.
  const srcPath =
    options.srcPath ??
    process.env.AIL_LABELING_SRC_PATH ??
    process.env.AIL_ONBOARDING_SRC_PATH ??
    process.env.AIL_APPLY_SRC_PATH ??
    path.resolve(process.cwd(), '..', 'src');
  const timeoutMs = options.timeoutMs ?? 120_000;

  return (input: LabelingAction) =>
    new Promise<LabelingResult>((resolve, reject) => {
      const existing = process.env.PYTHONPATH;
      const child = spawn(pythonBin, ['-m', 'ail.labeling.service'], {
        env: { ...process.env, PYTHONPATH: existing ? `${srcPath}${path.delimiter}${existing}` : srcPath },
      });

      let stdout = '';
      let stderr = '';
      const timer = setTimeout(() => {
        child.kill('SIGKILL');
        reject(new Error(`labeling-service timed out after ${timeoutMs}ms`));
      }, timeoutMs);

      // Handle stdin stream errors: a child that exits WITHOUT reading its stdin makes
      // the write below emit EPIPE on this stream. Without a handler that EPIPE
      // surfaces as an UNHANDLED exception AFTER the tests finish — vitest catches it
      // as an uncaught error and fails the gate even though every test passed (a timing
      // race that passes locally, fails in CI). The child's exit code + stdout are the
      // real signal (via 'close'/'error'), so a broken input pipe is safely swallowed.
      // (Same guard the onboarding subprocess bridge needs for its stub-interpreter tests.)
      child.stdin.on('error', () => {});
      child.stdout.on('data', (d: Buffer) => (stdout += d.toString()));
      child.stderr.on('data', (d: Buffer) => (stderr += d.toString()));
      child.on('error', (err) => {
        clearTimeout(timer);
        reject(err);
      });
      child.on('close', (code) => {
        clearTimeout(timer);
        if (code !== 0) {
          reject(new Error(`labeling-service exited ${code}: ${stderr.trim() || stdout.trim()}`));
          return;
        }
        try {
          resolve(JSON.parse(stdout) as LabelingResult);
        } catch {
          reject(new Error(`labeling-service returned unparseable output: ${stdout.slice(0, 500)}`));
        }
      });

      // Guard the write itself: if the pipe is already torn down the synchronous call
      // can throw. Swallow it — the 'close'/'error' handlers settle the promise from the
      // child's real exit code / output, never a write-after-close.
      try {
        child.stdin.write(JSON.stringify(input));
        child.stdin.end();
      } catch {
        // stdin already closed; the outcome comes from the child's exit + stdout.
      }
    });
}

// Transport selection — the subprocess transport for this slice. Kept as a function
// (not a bare export) so a future Node-only Job transport can be selected by env here
// without touching the route, exactly as the onboarding/approvals bridges do.
export function selectLabelingBridge(): LabelingBridge {
  return spawnPythonLabelingBridge();
}
