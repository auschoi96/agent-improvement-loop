# Task Suite — `phase2-mini-v1`

A small, **runnable** Task Suite for the Phase-2 live token-efficiency
comparison. Unlike `v1-seed` (the held-out benchmark abstracted from the real L0
diagnosis), every task here is backed by a live fixture under
`eval/phase2_fixtures/<task_id>/` so `ail.optimize.phase2.run_phase2_comparison`
can run baseline-vs-candidate on **real, file-mutating, deterministically
verifiable** coding tasks.

- **Artifact:** [`tasks.yaml`](./tasks.yaml) — load with
  `ail.task_suite.load_task_suite("phase2-mini")`.
- **Source:** `src/ail/task_suite/phase2_mini.py` (`build_phase2_mini_suite()`).
  `tasks.yaml` is the *frozen serialization* of that builder; a test pins that
  they agree, so the artifact cannot drift from its source.
- **Content version:** `phase2-mini-v1` · **schema version:** `ail.task_suite/v1`.
- **Content hash:** `b0fb7c3230ad3c896e1d86fddc6c1013f7c0b83fd95f7f7d672d01b0273e4a45`.

## The five tasks

| task_id | fixture | gap the agent must close |
|---|---|---|
| `ts-fix-01` | `shapes/` | two bugs (wrong triangle formula + mis-wired registry entry) |
| `ts-impl-02` | `calc/` | implement `evaluate()` over the existing helpers |
| `ts-refactor-03` | `report_a/b/c.py` | extract duplicated currency formatting into `common.py` |
| `ts-config-04` | `app/` | add + enforce a `max_retries` setting |
| `ts-route-05` | `api/` | implement + register a `get_user` handler |

Each task id matches its fixture directory so `load_fixture(task_id)` resolves.
The tasks are authored, self-contained (stdlib + pytest only, no network)
problems — **not** L0 trace reconstructions — so `source_trace_id` records
synthetic provenance (`phase2-fixture:<task_id>`) and `category` is a coarse
profile label. Each `seed/` ships the gap (so `verify/` fails as-is); a correct
change makes `python -m pytest -q verify/` pass.

## Running the comparison

`run_plan.yaml` (repo root) maps each task id to its verify command. See
`docs/PHASE2_LIVE_HARNESS.md` and `scripts/run_phase2_comparison.py`:

```bash
python scripts/run_phase2_comparison.py \
    --suite-version phase2-mini \
    --run-plan run_plan.yaml \
    --experiment <experiment> --profile <profile> \
    --output artifacts/phase2_mini_token_lever.json
```

## Freeze / immutability

`tasks.yaml` is **frozen** with a `content_hash` over its tasks; `load_task_suite`
recomputes the hash and fails closed (`TaskSuiteIntegrityError`) on any drift, and
rejects an unfrozen artifact outright — identical to the `v1` contract (see
`../v1/README.md` and `tests/test_task_suite.py`).
