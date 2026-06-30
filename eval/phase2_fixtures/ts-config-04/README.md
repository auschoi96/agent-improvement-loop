# `ts-config-04` — add and enforce a `max_retries` setting

**Prompt.** Add a `max_retries` setting (int, default 3) to `app/config.yaml`
and enforce it in `app/settings.py` so `verify/` passes. Do not modify the tests.

**Layout** (the live-fixture contract — see `docs/PHASE2_LIVE_HARNESS.md`):

- `seed/` — the `app/` package: `config.yaml` (flat `key: value`, no
  `max_retries`) and `settings.py` (`load()` + `validate()`, which raises on a
  missing required key). The config is parsed with the stdlib only — no PyYAML.
- `verify/` — the pristine pytest check. Verify command:
  `python -m pytest -q verify/`.

**The gap.** Neither the config nor `Settings` knows about `max_retries`, so
`load().max_retries` raises `AttributeError`. A correct change:

1. adds `max_retries: 3` to `config.yaml`;
2. gives `Settings` a `max_retries: int = 3` field;
3. enforces it in `validate`: default `3` when absent, and raise `ValueError`
   when present-but-invalid (non-int, or not a positive integer).

`load().max_retries == 3`; `validate({...})` defaults to `3`, honors an explicit
value, and raises `ValueError` for `0`/`"lots"` and for a missing required key.

Stdlib + pytest only; no third-party packages, no network.
