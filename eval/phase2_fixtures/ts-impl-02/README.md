# `ts-impl-02` — implement `evaluate()` in the `calc/` package

**Prompt.** Implement `evaluate()` in `calc/evaluate.py` so `verify/` passes,
using the existing helpers in `calc/ops.py` and `calc/parser.py`. Do not modify
the tests.

**Layout** (the live-fixture contract — see `docs/PHASE2_LIVE_HARNESS.md`):

- `seed/` — the `calc/` package: `ops.py` (`add`/`sub`/`mul`, `precedence`,
  `apply`), `parser.py` (`tokenize`), and `evaluate.py` (a stub raising
  `NotImplementedError`).
- `verify/` — the pristine pytest check. Verify command:
  `python -m pytest -q verify/`.

**The gap.** `evaluate()` is unimplemented. A correct implementation lexes the
expression with `tokenize`, applies `*` before `+`/`-` (precedence) and treats
equal-precedence operators left-associatively, using `precedence`/`apply`:

- `evaluate("2 + 3 * 4") == 14`
- `evaluate("10 - 2 - 3") == 5`

Stdlib + pytest only; no third-party packages, no network.
