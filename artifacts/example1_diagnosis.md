# Example 1 — Token-Waste Diagnosis (L0 baseline)

Deterministic L0 metrics over experiment `660599403165942` (91 traces), generated `2026-06-29T08:54:10.927851+00:00`. Reproduces the token-waste scenario in `docs/ARCHITECTURE.md` §8. Every number below is mechanical (token counts, timestamps, tool spans) — no model in the loop.

## Corpus

- **Traces:** 91
- **Status:** {'OK': 91}
- **Producers:** claude_code=90, <unknown-producer>=1
- **Models:** claude-opus-4-8=85, claude-sonnet-4-6=4, <unknown-model>=2

## Token distribution (bimodal)

- **Total tokens:** 8,252,407
- **Median:** 19,149 · **Mean:** 90,685.8 · **p90:** 249,891 · **Max:** 942,929
- A low median with a heavy tail: most sessions are small, a few are enormous. That tail is where the token spend lives.

## High-token sessions (Example 1)

Sessions at or above 500,000 total tokens:

| trace | model | total tokens | input | output | tools | duration (s) | est. cost |
|---|---|---|---|---|---|---|---|
| `a9b23c23c1efb90c576cd012f0a70522` | claude-opus-4-8 | 942,929 | 868,255 | 74,674 | 60 | 33,095.1 | $6.21 |
| `201506dc57618323b322df3a4c328f0f` | claude-opus-4-8 | 612,820 | 501,987 | 110,833 | 90 | 2,109.9 | $5.28 |
| `14da0deab1fee461c836a529d9f1e5ae` | claude-opus-4-8 | 549,300 | 479,341 | 69,959 | 57 | 1,365.7 | $4.15 |
| `680444e7d8a9679489700fb3d2958dc6` | claude-opus-4-8 | 543,910 | 391,029 | 152,881 | 156 | 3,324.8 | $5.78 |

## Tool-call redundancy

- **Strict redundancy rate** (byte-identical repeated calls): **0.006** (8 of 1239 calls). Low — exact-duplicate calls are rare; the waste is in repeated *targets*, below.

### Re-run shell-setup boilerplate (per trace)

Recurrences of the same normalized shell prologue (`cd <dir>`/env setup, per-session scratch UUIDs collapsed) — the agent re-establishing the same working directory on call after call:

| trace | repeats | prologue |
|---|---|---|
| `a9b23c23c1efb90c576cd012f0a70522` | 27× | `cd /Users/austin.choi/PycharmProjects2/omniagent/agent-framework` |
| `201506dc57618323b322df3a4c328f0f` | 20× | `cd /Users/austin.choi/PycharmProjects2/ail-worktrees/wave0b-l0-metrics` |
| `680444e7d8a9679489700fb3d2958dc6` | 15× | `cd /Users/austin.choi/PycharmProjects2/emerging_epl/emerging-epl` |
| `bdb3b11e597555cda869ed7ab5b123dd` | 13× | `cd /Users/austin.choi/PycharmProjects2/emerging_epl/emerging-epl` |
| `0d810f8f550f2c71206b64ba7e174a7c` | 12× | `cd /Users/austin.choi/PycharmProjects2/omniagent/agent-framework` |
| `f9eb702f32e1f531944ecad247a4deea` | 12× | `cd /Users/austin.choi/PycharmProjects2/emerging_epl/emerging-epl` |
| `acb3925baed7d48deb0ce7441f8cb0de` | 8× | `cd /Users/austin.choi/PycharmProjects2/emerging_epl/emerging-epl` |
| `14da0deab1fee461c836a529d9f1e5ae` | 7× | `cd /Users/austin.choi/PycharmProjects2/emerging_epl/emerging-epl` |

### Repeated file access (per trace)

Same file path targeted repeatedly by the same tool:

| trace | tool | repeats | path |
|---|---|---|---|
| `bdb3b11e597555cda869ed7ab5b123dd` | Edit | 8× | `sync_from_logfood.mjs` |
| `14da0deab1fee461c836a529d9f1e5ae` | Edit | 6× | `epl.ts` |
| `acb3925baed7d48deb0ce7441f8cb0de` | Edit | 6× | `schema.ts` |
| `680444e7d8a9679489700fb3d2958dc6` | Edit | 4× | `epl.ts` |
| `201506dc57618323b322df3a4c328f0f` | Edit | 3× | `l0_deterministic.py` |
| `44e3e992122283e59710d40aba014134` | Edit | 3× | `claude_code.py` |
| `8b53ae884b3458d81c3676c7da00aff4` | Read | 3× | `cli.py` |
| `e10ee82da2412883ea522b81092b472a` | Edit | 3× | `repoint_cc_tracing.py` |

## Estimated cost

- **Total (priced traces):** $66.64 across 89 priced trace(s); 2 unpriced.
- **Pricing caveats:**
  - Unpriced models (tokens counted, cost omitted): (no model recorded)
  - Base input/output prices: claude-api skill model pricing table (cached 2026-06-04). Verify against live pricing before using dollar figures for billing decisions.

## Reconciliation with `docs/ARCHITECTURE.md` §8 (77-trace snapshot)

| signal | live | doc snapshot | verdict |
|---|---|---|---|
| high-token sessions | ['942,929', '612,820', '549,300', '543,910'] | ~[549000, 943000] | match |
| median tokens | 19,149 | ~18,500 | match |
| shell boilerplate re-runs | up to 27×/trace | 13–21× | reproduced (max 27×/trace ≥ documented floor 13×) |
| re-read same path | 4× (Read), 8× (any file tool) | 34× | NOT reproduced — strongest live re-read of one path is 4× (any file tool 8×) vs documented 34×; corpus grew 77->91 and rotates, so the 34× trace is no longer present |

> **Flagged:** the documented *34× re-read of the same path* is **not** present in the live 91-trace corpus (strongest current re-read 4× by one tool, 8× across file tools). The corpus is explicitly live and growing (the doc snapshot was 77 traces); that high-redundancy trace has rotated out. The token-waste shape (bimodal distribution, huge tail sessions, re-run shell boilerplate) reproduces; the specific 34× figure does not, and is not asserted.

