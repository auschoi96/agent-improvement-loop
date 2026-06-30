# `ts-route-05` — implement and register a `get_user` handler

**Prompt.** Implement and register a `get_user` handler in `api/` so `verify/`
passes, routing `GET /users/<id>` to it using the existing `User` model + store;
return the user or a not-found result. Do not modify the tests.

**Layout** (the live-fixture contract — see `docs/PHASE2_LIVE_HARNESS.md`):

- `seed/` — the `api/` package: `models.py` (`User`, `NotFound`, `STORE`,
  `get()`), `router.py` (a `register`/`dispatch` router with `<id>` capture),
  `handlers.py` (one example `index` handler), and `__init__.py`
  (`build_router()` that registers only `GET /health`).
- `verify/` — the pristine pytest check. Verify command:
  `python -m pytest -q verify/`.

**The gap.** No `get_user` handler exists and `GET /users/<id>` is not routed, so
`build_router().dispatch("GET", "/users/1")` falls through to `NotFound` and
`test_existing_user_is_returned` fails. A correct change:

1. adds a `get_user` handler that looks up the user via `models.get` and returns
   the `User` or a `NotFound`;
2. registers it for `GET /users/<id>` in `build_router`.

`dispatch("GET", "/users/1")` returns `User(id=1, name="Ada")`;
`dispatch("GET", "/users/999")` returns a `NotFound`.

Stdlib + pytest only; no third-party packages, no network.
