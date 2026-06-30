# `ts-refactor-03` — extract duplicated currency formatting into `common.py`

**Prompt.** Extract the duplicated currency-formatting logic into a shared
helper `common.py` and use it in all three report modules. `verify/` must still
pass unchanged. Do not modify the tests.

**Layout** (the live-fixture contract — see `docs/PHASE2_LIVE_HARNESS.md`):

- `seed/` — `report_a.py`, `report_b.py`, `report_c.py`, each with an identical
  private `_format_currency` and a `render()`. There is no `common.py`.
- `verify/` — the pristine pytest check. Verify command:
  `python -m pytest -q verify/`.

**The gap (structural).** Unlike the other fixtures, the seed's *behavior* is
already correct: `test_reports_render_expected_output` passes as-is. What fails is
the extraction: there is no `common.py`, so `test_currency_logic_extracted_to_common`
and `test_all_reports_use_the_shared_helper` fail. A correct refactor:

1. creates `common.py` exposing `format_currency`;
2. has each report module import it (`from common import format_currency`) and
   call it from `render` — so `report_x.format_currency is common.format_currency`;
3. keeps every rendered string byte-for-byte the same.

Stdlib + pytest only; no third-party packages, no network.
