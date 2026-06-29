# Contributing & Merge Safety

This repo ships two independently-gated components: the Python package `ail/`
(`src/` + `tests/`) and the AppKit TypeScript app under `ail-self-optimizer/`.
Every change reaches `main` through a pull request that clears the automated CI
gates **and** human review. Direct pushes to `main` should be avoided — they
bypass both, and `main` is the branch everything else is cut from.

## Merge policy

A PR may be merged only when all three hold:

1. **Green in CI.** Every job in [`.github/workflows/ci.yml`](.github/workflows/ci.yml)
   must pass — the Python matrix (3.11 / 3.12 / 3.13: ruff lint, ruff format,
   mypy, pytest) and the AppKit job (typecheck, lint, format, build, vitest).
   A red or skipped required check is a blocker, not a footnote.

2. **Up to date with `main` before merge.** Merge (or rebase) the latest `main`
   into the branch and **re-run the suite** before merging. CI green on a stale
   branch only proves the branch passed against an old `main`; an unrelated
   change that landed in the meantime can still break the integration. Bringing
   `main` in and re-running is what makes the merge *provably* safe rather than
   safe-as-of-yesterday.

3. **Independent cross-vendor review.** At least one reviewer who did **not**
   author the change must approve it, and the review must come from an
   independent source (a different person/vendor than the implementer — not a
   self-approval or a rubber stamp from the same toolchain that wrote it).
   Reviewers check correctness, scope, and that no gate was weakened to go green.

## Reproducing the gates locally

Run these before opening or updating a PR — they mirror CI exactly.

### Python (`ail/`)

```bash
pip install -e ".[dev]"
ruff check src tests
ruff format --check src tests
mypy src
pytest                 # 81 passed, 2 skipped (the 2 live MLflow tests self-skip
                       # without AIL_LIVE_MLFLOW=1; CI deselects them via -m "not live")
```

No Databricks credentials are required: the two MLflow tests carry the registered
`live` marker (deselected in CI) and additionally self-skip via the
`AIL_LIVE_MLFLOW` guard, so the suite is green offline.

### AppKit app (`ail-self-optimizer/`)

```bash
cd ail-self-optimizer
npm ci
npm run typecheck
npm run lint
npm run format        # prettier --check
npm run build
npx vitest run        # headless unit tests
```

CI runs `npx vitest run` rather than `npm test`, because `npm test` also runs the
Playwright **smoke test**, which needs a running app, a browser, and a live
Databricks SQL warehouse. The smoke test belongs in a deploy-time / e2e check
against a real app, not in the per-PR offline gate.

## Don't weaken a gate to make it pass

If a gate is red, fix the cause. Do not disable a check, add `--no-verify`,
loosen a lint rule, or `skip` a test to turn red green. Generated artifacts that
a formatter/linter shouldn't own (e.g. AppKit's `appkit.plugins.json` and
`shared/appkit-types/`) are excluded via `.prettierignore` — that's scoping the
gate to hand-authored code, not weakening it.
