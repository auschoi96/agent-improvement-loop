# `ts-fix-01` — fix two bugs in the `shapes/` package

**Prompt.** The tests in `verify/` are failing. Fix the bug(s) in the `shapes/`
package so all tests pass. Do not modify the tests.

**Layout** (the live-fixture contract — see `docs/PHASE2_LIVE_HARNESS.md`):

- `seed/` — the `shapes/` package the agent edits (`area.py`, `registry.py`).
- `verify/` — the pristine pytest check, restored into the arm workspace after
  the run. Verify command: `python -m pytest -q verify/`.

**The gap (two bugs, two files).** Both seeded defects must be fixed:

1. `shapes/area.py` — `triangle_area` returns `base * height` instead of a
   triangle's `0.5 * base * height`.
2. `shapes/registry.py` — the `"triangle"` entry is mis-wired to
   `rectangle_area` instead of `triangle_area`.

`test_triangle_area` fails on the seed (it gets `rectangle_area(6, 4) == 24`,
not `12`); the circle and rectangle tests pass and act as regression guards. A
correct fix repairs the formula **and** the registry wiring.

Stdlib + pytest only; no third-party packages, no network.
