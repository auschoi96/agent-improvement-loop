# `example-token-task` — throwaway test fixture

This is the **single example fixture** that ships with the live Phase-2 harness.
It exists only to exercise the loader and the per-arm isolation / tamper-proof
verification machinery in `tests/test_phase2_live_isolation.py`. It is **not** a
real benchmark task — real fixtures are authored by a separate lane.

Layout (the live-fixture contract — see `docs/PHASE2_LIVE_HARNESS.md`):

- `seed/` — the starting repo state the agent edits. Copied into a fresh, per-arm
  workspace before each run.
- `verify/` — the pristine, deterministic L1 check. Restored into each arm's
  workspace **after** that arm's run (overwriting any agent edits), then run with
  `cwd` = that arm's workspace.

`verify/check.py` exits `0` iff `seed/solution.py` (as the agent left it) contains
the implementation `return a + b`.
