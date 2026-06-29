# Example 1 — Token-Waste Diagnosis (L0 baseline)

Deterministic L0 metrics over experiment `660599403165942` (90 traces), generated `2026-06-29T08:30:42.456472+00:00`. Reproduces the token-waste scenario in `docs/ARCHITECTURE.md` §8. Every number below is mechanical (token counts, timestamps, tool spans) — no model in the loop.

## Corpus

- **Traces:** 90
- **Status:** {'OK': 90}
- **Producers:** claude_code=89, <unknown-producer>=1
- **Models:** claude-opus-4-8=84, claude-sonnet-4-6=4, <unknown-model>=2

## Token distribution (bimodal)

- **Total tokens:** 7,639,587
- **Median:** 18,828.5 · **Mean:** 84,884.3 · **p90:** 242,673.9 · **Max:** 942,929
- A low median with a heavy tail: most sessions are small, a few are enormous. That tail is where the token spend lives.

## High-token sessions (Example 1)

Sessions at or above 500,000 total tokens:

| trace | model | total tokens | input | output | tools | duration (s) | est. cost |
|---|---|---|---|---|---|---|---|
| `a9b23c23c1efb90c576cd012f0a70522` | claude-opus-4-8 | 942,929 | 868,255 | 74,674 | 60 | 33,095.1 | $6.21 |
| `14da0deab1fee461c836a529d9f1e5ae` | claude-opus-4-8 | 549,300 | 479,341 | 69,959 | 57 | 1,365.7 | $4.15 |
| `680444e7d8a9679489700fb3d2958dc6` | claude-opus-4-8 | 543,910 | 391,029 | 152,881 | 156 | 3,324.8 | $5.78 |

## Tool-call redundancy

- **Strict redundancy rate** (byte-identical repeated calls): **0.007** (8 of 1149 calls). Low — exact-duplicate calls are rare; the waste is in repeated *targets*, below.

### Re-run shell-setup boilerplate (per trace)

Recurrences of the same normalized shell prologue (`cd <dir>`/env setup, per-session scratch UUIDs collapsed) — the agent re-establishing the same working directory on call after call:

| trace | repeats | prologue |
|---|---|---|
| `a9b23c23c1efb90c576cd012f0a70522` | 27× | `cd /Users/austin.choi/PycharmProjects2/omniagent/agent-framework` |
| `680444e7d8a9679489700fb3d2958dc6` | 15× | `cd /Users/austin.choi/PycharmProjects2/emerging_epl/emerging-epl` |
| `bdb3b11e597555cda869ed7ab5b123dd` | 13× | `cd /Users/austin.choi/PycharmProjects2/emerging_epl/emerging-epl` |
| `0d810f8f550f2c71206b64ba7e174a7c` | 12× | `cd /Users/austin.choi/PycharmProjects2/omniagent/agent-framework` |
| `f9eb702f32e1f531944ecad247a4deea` | 12× | `cd /Users/austin.choi/PycharmProjects2/emerging_epl/emerging-epl` |
| `acb3925baed7d48deb0ce7441f8cb0de` | 8× | `cd /Users/austin.choi/PycharmProjects2/emerging_epl/emerging-epl` |
| `14da0deab1fee461c836a529d9f1e5ae` | 7× | `cd /Users/austin.choi/PycharmProjects2/emerging_epl/emerging-epl` |
| `e10ee82da2412883ea522b81092b472a` | 5× | `export SCRATCH="/private/tmp/claude-502/-Users-austin-choi/<id>/scratchpad"` |

### Repeated file access (per trace)

Same file path targeted repeatedly by the same tool:

| trace | tool | repeats | path |
|---|---|---|---|
| `bdb3b11e597555cda869ed7ab5b123dd` | Edit | 8× | `sync_from_logfood.mjs` |
| `14da0deab1fee461c836a529d9f1e5ae` | Edit | 6× | `epl.ts` |
| `acb3925baed7d48deb0ce7441f8cb0de` | Edit | 6× | `schema.ts` |
| `680444e7d8a9679489700fb3d2958dc6` | Edit | 4× | `epl.ts` |
| `44e3e992122283e59710d40aba014134` | Edit | 3× | `claude_code.py` |
| `8b53ae884b3458d81c3676c7da00aff4` | Read | 3× | `cli.py` |
| `e10ee82da2412883ea522b81092b472a` | Edit | 3× | `repoint_cc_tracing.py` |
| `f9eb702f32e1f531944ecad247a4deea` | Edit | 3× | `sync_from_logfood.mjs` |

## Estimated cost

- **Total (priced traces):** $61.36 across 88 priced trace(s); 2 unpriced.
- **Pricing caveats:**
  - Unpriced models (tokens counted, cost omitted): (no model recorded)
  - Base input/output prices: claude-api skill model pricing table (cached 2026-06-04). Verify against live pricing before using dollar figures for billing decisions.

## Reconciliation with `docs/ARCHITECTURE.md` §8 (77-trace snapshot)

| signal | live | doc snapshot | verdict |
|---|---|---|---|
| high-token sessions | ['942,929', '549,300', '543,910'] | ~[549000, 943000] | match |
| median tokens | 18,828.5 | ~18,500 | match |
| shell boilerplate re-runs | up to 27×/trace | 13–21× | reproduced (re-run shell setup prologue is present and in/above range) |
| re-read same path | 4× (Read), 8× (any file tool) | 34× | NOT reproduced on the live corpus — the 34x re-read trace is not present (corpus grew 77->90 and rotates); strongest current re-read of one path is 4x |

> **Flagged:** the doc's *34× re-read of the same path* is **not** present in the live 90-trace corpus. The corpus is explicitly live and growing (the doc snapshot was 77 traces); that high-redundancy trace has rotated out. The token-waste shape (bimodal distribution, huge tail sessions, re-run shell boilerplate) reproduces; the specific 34× figure does not, and is not asserted.

