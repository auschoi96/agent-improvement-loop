# Phase-2 Live Harness — per-arm isolation + tamper-proof verification

The Phase-2 comparison (`ail.compare.compare_candidate` →
`ail.optimize.phase2.run_phase2_comparison`) decides PROMOTE/BLOCK for a
token-efficiency lever by running each frozen Task-Suite task twice — a
**baseline** arm (no intervention) and a **candidate** arm (intervention applied)
— and gating on a deterministic **L1** check.

This document is the contract for running that comparison on **real,
file-mutating coding tasks honestly**. Two properties make it honest:

1. **Per-arm isolation** — the two arms edit *separate copies* of the starting
   repo state, so the candidate's edits can never land on top of the baseline's
   (no cross-arm contamination).
2. **Tamper-proof, arm-aware verification** — after each arm's run, the pristine
   L1 check is *restored* into that arm's workspace (overwriting any agent edits
   to the test) and run there, so the baseline verdict reflects the baseline's
   edits and the candidate verdict the candidate's — and an agent cannot game the
   check by editing or deleting the test.

It extends the existing seam; it does **not** fork a parallel runner. Mock /
trace-only tasks (no fixture) keep running on the legacy arm-blind path
unchanged.

---

## Fixture layout

A live task is backed by a fixture directory:

```
eval/phase2_fixtures/<task_id>/
├── seed/      # the starting repo state the agent edits
│   └── ...    # source files, package layout, etc.
└── verify/    # the pristine, deterministic L1 check (e.g. a pytest test)
    └── ...
```

- **`seed/`** is copied into a fresh, per-arm workspace before each run. It is the
  only thing the agent sees and edits.
- **`verify/`** is the L1 check. It is **not** copied at run time; it is
  *restored* into each arm's workspace **after** that arm's run (see below). It is
  authored to be deterministic — a pytest test, a script that exits `0`/`1`, a
  build invocation — anything whose exit code is a trustworthy pass/fail.

> Fixtures are authored by a **separate lane**. This harness ships exactly one
> tiny throwaway fixture, `eval/phase2_fixtures/example-token-task/`, used only by
> `tests/test_phase2_live_isolation.py`.

A task with **no** fixture directory falls back to the legacy arm-blind path
(sound only for mock / trace-only tasks — it provides no file isolation).

---

## Per-arm execution (no contamination)

For each task with a fixture, the runner (`ail.optimize.phase2`, via
`ail.optimize.fixtures.isolated_arm_workspaces`):

1. creates a fresh per-task temp directory with two subdirectories,
   `baseline/` and `candidate/`, and copies `seed/` into **each** — separate
   directories, never shared;
2. calls `compare_candidate(..., workspace=ArmWorkspaces(baseline_cwd=…,
   candidate_cwd=…, verify=…))`. The harness sets each arm's
   `AgentTask.cwd` to its own workspace (`baseline_cwd` for the baseline arm,
   `candidate_cwd` for the candidate arm), so the agent edits an isolated copy.

Because the workspaces are distinct directories, the candidate's edits are never
visible to the baseline and vice-versa.

## Tamper-proof, arm-aware verification

After both arms have run, the harness verifies **each arm in its own workspace**
via the `ArmVerifier` the runner supplies (`ArmVerifier = Callable[[str,
AgentRunResult], ProgrammaticSignal]`). For an arm with workspace `ws`, the
verifier:

1. **restores** the pristine `verify/` tree into `ws/verify` — removing any
   agent-created `verify/` first and copying the fixture's copy back
   (`ail.optimize.fixtures.restore_verify`), so a test the agent edited, added to,
   or deleted is replaced by the original;
2. runs the verify command with `cwd = ws`.

The two per-arm signals feed the **unchanged** programmatic guardrail
(`ail.compare.harness._programmatic_guardrail`), so all fail-closed semantics are
identical to before — the only change is that the signals are produced per-arm
instead of from a single fixed-cwd run.

## Run plan (`--run-plan` YAML/JSON)

The run plan maps `task_id → L1 verification command`. It is passed to
`scripts/run_phase2_comparison.py` (the live CLI driver); the library function is
the tested seam.

```yaml
# run_plan.yaml
example-token-task:
  name: example-check
  command: ["python", "verify/check.py"]   # relative to the arm workspace
  timeout_seconds: 600

ts-017:
  name: pytest-suite
  command: ["python", "-m", "pytest", "verify", "-q"]
```

Fields:

| field             | meaning                                                        |
| ----------------- | -------------------------------------------------------------- |
| `command`         | argv list (or string with `shell: true`) — **required**        |
| `name`            | label recorded on the signal (default `verify-<task_id>`)      |
| `shell`           | run via the shell (default `false`)                            |
| `timeout_seconds` | no-verdict timeout → fails closed (default `600`)              |
| `cwd`             | **ignored for fixture-backed tasks** (see below)               |

**`cwd` is set BY the harness, not the plan.** For a fixture-backed task the
command runs with `cwd` = that arm's isolated workspace (where `verify/` has been
restored and the agent's edits live), so a `cwd` in the plan is ignored. Write
commands **relative to the workspace root** (e.g. `verify/check.py`,
`python -m pytest verify`). `cwd` is honored only for legacy non-fixture (mock /
trace-only) tasks.

A task with **no** run-plan entry has no L1 correctness signal and fails closed
(`BLOCK`), even with a fixture present.

### Running it

```bash
python scripts/run_phase2_comparison.py \
    --suite-version v1 \
    --experiment /Shared/dais-demo-agent-improvement \
    --profile dais-demo \
    --run-plan run_plan.yaml \
    --fixtures-root .            # default: repo discovery of eval/phase2_fixtures
    --output artifacts/phase2_token_lever.json
```

`--fixtures-root` overrides where `eval/phase2_fixtures` is discovered (default:
walk up from the package, matching `AIL_PHASE2_FIXTURES_ROOT`). **This runs the
real, costly comparison — do not run it in CI.** The library functions
(`compare_candidate`, `run_phase2_comparison`) are the unit-tested seam; tests use
a fake adapter, never a live agent.

---

## Fail-closed guardrails (unchanged)

Per-arm isolation and arm-aware verification change *how the L1 signal is
produced*, not the gate. Every one of these still BLOCKS, exactly as before:

- a **crashed/failed** agent run (execution guardrail — its near-zero "token
  reduction" is never a win);
- a **failed baseline** (not a valid anchor — a reduction measured against it
  proves nothing);
- a **missing** verifier (no run-plan entry → no correctness signal);
- an **unrunnable / errored** verifier (cannot launch, times out, crashes →
  `errored` → no verdict);
- a **correctness regression** (the check passed at baseline and fails for the
  candidate → `REGRESSED`);
- a **sub-threshold** token reduction (`--min-reduction-pct`).

Realized token savings are still summed over **PROMOTE** tasks only — a blocked or
crashed task's token delta is never counted.

## Cleanup

The per-arm workspaces are torn down after each task — **including on
error/exception** (the temp tree is removed in the `isolated_arm_workspaces`
context manager's `finally`), so a crashed run never leaks a workspace.

---

## Code map

| concern                                   | location                                                       |
| ----------------------------------------- | -------------------------------------------------------------- |
| per-arm cwd + arm-aware verify seam       | `ArmWorkspaces`, `ArmVerifier`, `compare_candidate` in `src/ail/compare/harness.py` |
| fixture loader + workspace lifecycle      | `src/ail/optimize/fixtures.py` (`load_fixture`, `restore_verify`, `isolated_arm_workspaces`) |
| runner wiring (dual path) + verify command| `src/ail/optimize/phase2.py` (`_compare_isolated`, `_make_arm_verifier`, `run_phase2_comparison`) |
| live CLI driver                           | `scripts/run_phase2_comparison.py`                             |
| tests (fake adapter, no live agent)       | `tests/test_phase2_live_isolation.py`                          |
| example throwaway fixture                 | `eval/phase2_fixtures/example-token-task/`                     |
