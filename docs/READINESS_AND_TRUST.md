# Readiness & Trust

This document defines **when the optimizer is allowed to claim an improvement**
and **how it refuses to lie**. It is the product spine for a self-improving
loop: without it, the system manufactures false confidence — scores climb while
real quality stalls or regresses.

Read alongside `docs/ARCHITECTURE.md` (the frozen-eval-wall design).

---

## 1. "Connect a trace stream", not "upload an agent"

The intuitive aspiration is *"upload my agent and it gets optimized."* That is
**not achievable** and we should not imply it. The unit of input is not an agent
binary — it is a **stream of observed behavior (traces)**. The system optimizes
what it can *measure*, and it can only measure what has been *traced*.

So the honest UX is: **connect your agent's trace source** (point it at an
MLflow experiment, or install the adapter that logs its sessions), then let the
layer observe, evaluate, and propose changes behind the scenes. The
`AgentAdapter` / `TraceSource` seam is that connection point — a connection, not
an upload.

**Corollary the user must see up front:** a freshly connected agent with **0
traces produces 0 trustworthy optimization**. There is a warm-up period, and the
system must say so plainly rather than show an empty dashboard that implies
readiness.

---

## 2. Tiered readiness — different goals need different data

"How many traces do I need?" has no single answer. Each goal has its own gate.
The system must compute readiness **per goal** and **refuse to claim improvement
until the gate is met**, showing the user exactly which gate they are behind.

| Goal | Can baseline at | Can *prove* improvement at | Why |
|---|---|---|---|
| **Token / cost** (L0, deterministic) | ~10–20 traces | ~50+ diverse traces **and** ≥N candidate runs | Token usage is heavy-tailed (median ~18K, tail ~940K here). High variance ⇒ a "50% reduction" on a handful of traces is noise, not signal. |
| **Coding quality** (L2, judged) | needs a frozen Task Suite | ~30–50 human-labeled traces **and** a calibrated judge **and** a frozen suite | A judge with no human labels is uncalibrated; a quality claim from an uncalibrated judge is unfounded. Labels are the hard gate. |
| **Deep failure-mode discovery** (L3, recursive) | large/complex traces only | n/a (diagnostic, not a leaderboard metric) | Recursive review is wasted on trivial traces; it informs *what to fix*, it does not score the leaderboard. |

### Readiness ladder (heuristics, not laws — surfaced, never silent)

- **0 traces** → status: *Collecting*. No baseline, no claims. Tell the user
  what to connect.
- **~10–20 traces** → L0 baseline + waste diagnosis become meaningful. **Still
  cannot prove improvement** (no candidate comparison, no statistical power on a
  heavy tail).
- **~30–50 traces + frozen Task Suite + ≥~20 human labels** → first
  *trustworthy* quality signal; judge alignment (MemAlign) can run.
- **~50+ diverse traces** → statistical power for improvement claims; pass@k /
  pass^k become meaningful (consistent with the ~40–50-input guidance for
  significance).
- **Ongoing collection** → feeds judge re-alignment and overfit detection on
  their own cadence.

These thresholds are defaults and must be **visible and adjustable**, not buried
constants. The product surface is a **Readiness panel**: for each stated goal,
red/yellow/green plus *"you need N more traces / M more labels / a frozen
baseline."*

---

## 3. Risk register — the failure modes that destroy trust

A self-improving loop has adversarial dynamics with itself. These are the
concrete ways it produces fake improvement, and the guardrail for each.

| # | Failure mode | Guardrail (the invariant) |
|---|---|---|
| 1 | **Silent eval failure** — a judge/RLM errors or returns a default that reads as "fine" (e.g. HALO parser returning score 0 = "perfect" on a broken run; an unmeasured judge reading as "trusted"). | **FAIL LOUD, FAIL CLOSED.** An un-run or errored evaluation is **never** a pass. Distinguish *evaluated-and-passed* from *did-not-evaluate*. (Already enforced in the L3 parser and the judge-agreement `distrusted` flag — must be a system-wide invariant.) |
| 2 | **Coverage gap masquerading as health** — scorers *registered* but not *running* (no SQL warehouse), or sampling 10% — dashboard shows "judges configured" while ~0 traces are actually scored. | Surface **scored-coverage** (% of traces with real verdicts) and **judge-run success rate**, not just "judges exist." Low coverage ⇒ no claim. |
| 3 | **Agent games the judge (Goodhart / reward hacking)** — agent learns the judge's surface cues instead of doing good work (e.g. terse-but-wrong to look "token-efficient"). | L0 metrics are judged-independent (counted from spans). Token efficiency is scored **conditioned on task success**. The **frozen Task Suite the optimizer never trains on** is the anti-gaming spine. |
| 4 | **Judge–agent co-adaptation** — aligning the judge on feedback from the same loop the agent optimizes in ⇒ both drift together, scores climb, reality doesn't. | Align the judge on a **separate cadence** against **fresh human labels** on a **disjoint slice**. Three disjoint pools (Task Suite / Alignment Set / Human Anchor) never mixed. |
| 5 | **Overfitting to a small static suite** — GEPA overfits the exact suite tasks. | Rotate/refresh held-out tasks; track a never-seen rotation; alarm if suite score climbs but rotation doesn't. |
| 6 | **Baseline drift / moving goalposts** — live corpus grows; naive re-baselining hides regressions. | Versioned, **frozen** baselines per comparison. One comparison = one fixed baseline. |
| 7 | **Attribution failure** — multiple changes ship; can't tell which helped. | One lever at a time through the comparison harness. |
| 8 | **Opaque numbers** — "you improved 30%" with no provenance. | Every claimed number carries provenance: which traces, which judge version, which baseline, sample size, confidence. No black-box improvement. |

### The trust alarm

The first-class metric that catches most of the above is **judge-vs-human
agreement**, tracked over time against the Human Anchor with a **floor**. If
agreement drops below the floor, the judge is **distrusted** and its verdicts
stop counting toward improvement claims until re-aligned. A judge that hasn't
been measured against humans is distrusted **by default** (never silently
trusted).

---

## 4. What this means for the build

- **Readiness gating module** — compute per-goal readiness from {trace count,
  label count, frozen-suite present, judge agreement, scored coverage}; gate
  every improvement claim on it.
- **Eval-health / coverage metric** — scored-coverage %, judge-run success rate,
  count of distrusted judges — surfaced in the app next to every number.
- **System-wide fail-closed invariant** — no un-run/errored evaluation ever
  counts as a pass, anywhere (parsers, judges, harness guardrails).
- **Comparison harness** consults readiness + coverage before it is allowed to
  emit `PROMOTE`; a guardrail judge that did not actually run ⇒ `BLOCK`.

---

## 5. Current honest status (example of the panel in words)

As of this writing, for experiment `660599403165942`:

- **L0 (token/cost):** baseline exists (~91 traces). Diagnosis is real. Improvement
  *provable in principle*, pending candidate runs on a frozen suite.
- **Quality (L2):** judges are **registered but NOT running** (no SQL warehouse
  attached) → **scored-coverage ≈ 0%**. And there are **0 human labels** → judges
  are **uncalibrated / distrusted by default**. Therefore: **no quality
  improvement claim is permitted yet.**
- **L3:** reviewer built; runs on demand on large traces; diagnostic only.

The correct thing for the system to display today is **"not ready for quality
optimization — collect labels + attach a scoring warehouse,"** not a green
dashboard. That refusal *is* the feature.
