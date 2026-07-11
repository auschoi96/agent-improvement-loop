# Phase-2 Runnable Mini-Suite — Fixture Spec (5 tasks)

Purpose: a small, **runnable + deterministically verifiable** task suite so the
Phase-2 live comparison can prove the token-efficiency skill (candidate) reduces
tokens **with correctness held** vs. the no-skill baseline. Each task is run in
its own per-arm isolated workspace (see `docs/PHASE2_LIVE_HARNESS.md`), edited by
the Claude Code agent, and graded by a tamper-proof pytest check.

## Design rules (all 5 tasks)
- **Self-contained**: each fixture is a tiny Python project; **stdlib + pytest only**, no network, no external deps. Runs anywhere.
- **File-editing + read-heavy**: each requires reading 2–3 files and editing 1–2 — this is what exercises the redundant-read / boilerplate token waste the skill targets.
- **Baseline-solvable**: a clean (no-skill) Claude Code run should reliably PASS. This is essential — a failed baseline BLOCKs as "not a valid anchor", so an unsolvable task yields no signal.
- **Tamper-proof verify**: the `verify/` test is restored from pristine before grading; the prompt says "Do not modify the tests." The agent cannot game the check.
- **Layout** (per the harness contract): `eval/phase2_fixtures/<task_id>/seed/` (starting state the agent edits) + `eval/phase2_fixtures/<task_id>/verify/` (the pytest check).
- **Verify command** (uniform), run by the harness with `cwd` = the arm's workspace: `["python", "-m", "pytest", "-q", "verify/"]`, timeout 600s.
- **Difficulty**: medium — real multi-file work, but unambiguous and solvable. Avoid trickiness that makes the baseline flaky (that would muddy the comparison).

## Token-waste shapes (mapped to real corpus patterns)
Multi-file tracing, cross-module API reading, re-reading the same files during a
refactor, repeated `cd`/setup boilerplate, and route+model reading — these mirror
the dominant waste the RLM batch + L0 found (repeated Read of the same path,
27× `cd` boilerplate, the `epl.ts` route edits).

---

## ts-fix-01 — Multi-file bug fix
**Prompt:** "The tests in `verify/` are failing. Find and fix the bug(s) in the `shapes/` package so all tests pass. Do not modify the tests."
**seed/** `shapes/__init__.py`, `shapes/area.py` (area functions: `circle_area`, `rectangle_area`, `triangle_area` — **triangle has the wrong formula**, e.g. `base * height` instead of `0.5 * base * height`), `shapes/registry.py` (maps shape name → area fn — **one entry mis-wired**, e.g. "triangle" → rectangle_area).
**verify/** `test_shapes.py`: asserts correct areas for circle, rectangle, triangle via the registry; two bugs across two files.
**Waste exercised:** re-reading `area.py` + `registry.py` repeatedly while tracing the failure.

## ts-impl-02 — Implement-to-spec across modules
**Prompt:** "Implement `evaluate()` in `calc/evaluate.py` so the tests in `verify/` pass, using the existing helpers in `calc/ops.py` and `calc/parser.py`. Do not modify the tests."
**seed/** `calc/ops.py` (`add`, `sub`, `mul` with correct precedence helpers), `calc/parser.py` (`tokenize(expr)` → tokens), `calc/evaluate.py` (`def evaluate(expr: str) -> float: raise NotImplementedError`).
**verify/** `test_evaluate.py`: `evaluate("2 + 3 * 4") == 14`, `evaluate("10 - 2 - 3") == 5`, etc.
**Waste exercised:** reading `ops.py` + `parser.py` multiple times to learn the API before implementing.

## ts-refactor-03 — Behavior-preserving refactor
**Prompt:** "Extract the duplicated currency-formatting logic into a single shared helper in `common.py` and use it in all three report modules (`report_a.py`, `report_b.py`, `report_c.py`). The tests in `verify/` must still pass unchanged. Do not modify the tests."
**seed/** three modules each containing an identical `_format_currency(amount)` plus a `render()` that uses it; no `common.py` yet.
**verify/** `test_reports.py`: asserts each report's rendered output (unchanged behavior).
**Waste exercised:** re-reading the three near-identical files during the extraction.

## ts-config-04 — Config + code change with validation
**Prompt:** "Add a `max_retries` setting (integer, default 3) to `app/config.yaml` and enforce it in `app/settings.py` (load it, validate it's a positive int) so the tests in `verify/` pass. Do not modify the tests."
**seed/** `app/config.yaml` (existing settings, **no** `max_retries`), `app/settings.py` (`load()` parses + validates known keys; raises on a missing required key).
**verify/** `test_settings.py`: asserts `load().max_retries == 3`, asserts a `ValueError` when `max_retries` is missing/invalid.
**Waste exercised:** repeated `cd`/`cat`/re-read of config + settings while iterating (the boilerplate-re-run pattern).

## ts-route-05 — Add an API route using an existing model
**Prompt:** "Implement and register a `get_user` handler in the `api/` package so the tests in `verify/` pass, using the existing `User` model and in-memory store. Route `GET /users/<id>` to it; return the user, or a 404-style result for a missing id. Do not modify the tests."
**seed/** `api/models.py` (`User` dataclass + `STORE` dict + `get(user_id)`), `api/router.py` (a tiny dict-dispatch router: `register(method, path_pattern, handler)` + `dispatch(method, path)`), `api/handlers.py` (one existing handler as a pattern to follow), `api/__init__.py`.
**verify/** `test_api.py`: `dispatch("GET", "/users/1")` returns the right user; `/users/999` returns the not-found result.
**Waste exercised:** reading `models.py` + `router.py` + `handlers.py` to learn the wiring (the real `epl.ts` route pattern).

---

## run_plan.yaml (authored by orchestrator; cwd set by harness per arm)
```yaml
ts-fix-01:      {name: verify, command: ["python","-m","pytest","-q","verify/"], timeout_seconds: 600}
ts-impl-02:     {name: verify, command: ["python","-m","pytest","-q","verify/"], timeout_seconds: 600}
ts-refactor-03: {name: verify, command: ["python","-m","pytest","-q","verify/"], timeout_seconds: 600}
ts-config-04:   {name: verify, command: ["python","-m","pytest","-q","verify/"], timeout_seconds: 600}
ts-route-05:    {name: verify, command: ["python","-m","pytest","-q","verify/"], timeout_seconds: 600}
```

## Honest measurement plan
- Run each task **N≥3 times** per arm (token counts vary run-to-run); report mean/min/max per task and the distribution, not a single number.
- Headline = token reduction summed over **PROMOTE** tasks only (correctness held), per the harness. A task where the baseline fails is excluded (not a valid anchor) and flagged.
- Expect **smaller, noisier** deltas than the 500K-token sessions — the claim is "directionally fewer tokens with correctness held across 5 tasks", not a guaranteed 50%.
