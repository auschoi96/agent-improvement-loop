import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { mkdtempSync, rmSync, writeFileSync, chmodSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { spawnPythonLabelingBridge, selectLabelingBridge, type LabelingAction } from './bridge';

// Hermetic subprocess test: the bridge hardcodes `-m ail.labeling.service`, so we
// point `pythonBin` at tiny stub interpreters that ignore those args and stand in for
// the Python engine — proving the bridge's happy path (stdin action -> parsed stdout
// result) and its fail-closed handling (non-zero exit / unparseable output / spawn
// failure) without needing the `ail` package or a live workspace.

let dir: string;
const stub = (name: string, script: string): string => {
  const p = path.join(dir, name);
  writeFileSync(p, `#!/bin/sh\n${script}\n`);
  chmodSync(p, 0o755);
  return p;
};

let echoStub: string;
let failStub: string;
let garbageStub: string;

const INPUT: LabelingAction = {
  action: 'label',
  actor: 'a@b.com',
  experiment_id: 'exp-1',
  trace_id: 't1',
  name: 'correctness',
  value: 'pass',
};

beforeAll(() => {
  dir = mkdtempSync(path.join(tmpdir(), 'ail-labeling-bridge-'));
  echoStub = stub('echo.sh', 'cat'); // echo stdin JSON back as the "result"
  failStub = stub('fail.sh', 'echo "boom" 1>&2\nexit 1');
  garbageStub = stub('garbage.sh', "printf 'not json {'");
});

afterAll(() => {
  rmSync(dir, { recursive: true, force: true });
});

describe('spawnPythonLabelingBridge — subprocess transport', () => {
  it('writes the action on stdin and parses the JSON result from stdout', async () => {
    const bridge = spawnPythonLabelingBridge({ pythonBin: echoStub, srcPath: dir });
    const result = await bridge(INPUT);
    // the echo stub returns the action verbatim -> the bridge parsed it as the result
    expect(result).toMatchObject({ action: 'label', actor: 'a@b.com', name: 'correctness' });
  });

  it('rejects a non-zero exit (never a fabricated result)', async () => {
    const bridge = spawnPythonLabelingBridge({ pythonBin: failStub, srcPath: dir });
    await expect(bridge(INPUT)).rejects.toThrow(/exited 1/);
  });

  it('rejects unparseable stdout (never a fabricated result)', async () => {
    const bridge = spawnPythonLabelingBridge({ pythonBin: garbageStub, srcPath: dir });
    await expect(bridge(INPUT)).rejects.toThrow(/unparseable/);
  });

  it('rejects when the interpreter cannot be spawned', async () => {
    const bridge = spawnPythonLabelingBridge({
      pythonBin: path.join(dir, 'does-not-exist'),
      srcPath: dir,
    });
    await expect(bridge(INPUT)).rejects.toBeInstanceOf(Error);
  });
});

describe('selectLabelingBridge', () => {
  it('returns the subprocess transport (a callable bridge)', () => {
    expect(typeof selectLabelingBridge()).toBe('function');
  });
});
