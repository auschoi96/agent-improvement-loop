import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { mkdtempSync, rmSync, writeFileSync, chmodSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { spawnPythonOnboardingBridge, selectOnboardingBridge, type OnboardingAction } from './bridge';

// Hermetic subprocess test: the bridge hardcodes `-m ail.onboarding.service`, so we
// point `pythonBin` at tiny stub interpreters that ignore those args and stand in
// for the Python engine — proving the bridge's happy path (stdin action -> parsed
// stdout result) and its fail-closed handling (non-zero exit / unparseable output)
// without needing the `ail` package or a live workspace.

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

const INPUT: OnboardingAction = { action: 'requirements', actor: 'a@b.com', goals: ['cost'] };

beforeAll(() => {
  dir = mkdtempSync(path.join(tmpdir(), 'ail-onboarding-bridge-'));
  echoStub = stub('echo.sh', 'cat'); // echo stdin JSON back as the "result"
  failStub = stub('fail.sh', 'echo "boom" 1>&2\nexit 1');
  garbageStub = stub('garbage.sh', "printf 'not json {'");
});

afterAll(() => {
  rmSync(dir, { recursive: true, force: true });
});

describe('spawnPythonOnboardingBridge — subprocess transport', () => {
  it('writes the action on stdin and parses the JSON result from stdout', async () => {
    const bridge = spawnPythonOnboardingBridge({ pythonBin: echoStub, srcPath: dir });
    const result = await bridge(INPUT);
    // the echo stub returns the action verbatim -> the bridge parsed it as the result
    expect(result).toMatchObject({ action: 'requirements', actor: 'a@b.com' });
  });

  it('rejects a non-zero exit (never a fabricated result)', async () => {
    const bridge = spawnPythonOnboardingBridge({ pythonBin: failStub, srcPath: dir });
    await expect(bridge(INPUT)).rejects.toThrow(/exited 1/);
  });

  it('rejects unparseable stdout (never a fabricated result)', async () => {
    const bridge = spawnPythonOnboardingBridge({ pythonBin: garbageStub, srcPath: dir });
    await expect(bridge(INPUT)).rejects.toThrow(/unparseable/);
  });

  it('rejects when the interpreter cannot be spawned', async () => {
    const bridge = spawnPythonOnboardingBridge({
      pythonBin: path.join(dir, 'does-not-exist'),
      srcPath: dir,
    });
    await expect(bridge(INPUT)).rejects.toBeInstanceOf(Error);
  });
});

describe('selectOnboardingBridge', () => {
  it('returns the subprocess transport (a callable bridge)', () => {
    expect(typeof selectOnboardingBridge()).toBe('function');
  });
});
