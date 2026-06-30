# Connect your agent — "uploading" = start tracing to its experiment

> **The mental-model shift (repeated because it matters):** you do **not** upload
> an agent to this framework. There is no model file, no binary, no checkpoint to
> hand over. The optimizer improves **what it can measure**, and it can only
> measure **what has been traced**. So "onboarding an agent" means one thing:
> **point that agent's tracing at an MLflow experiment**, and let traces
> accumulate. *Connecting the trace stream is the upload.*

This guide is the generic, framework-agnostic onboarding path. For the two
first-class adapters see [the Claude Code path in GETTING_STARTED](GETTING_STARTED.md)
and [`CONNECT_CODEX.md`](CONNECT_CODEX.md). Everything below applies to **any**
agent that can emit traces — including custom agents, LangChain/LangGraph,
OpenAI/Anthropic SDK apps, and a full multi-agent / supervisor system.

---

## Step 0 — Give the agent its own experiment (one agent = one experiment)

This framework uses **one MLflow experiment per agent** (see
[`OBSERVABILITY_APP.md`](OBSERVABILITY_APP.md) for why). Each experiment gets its
own judges, its own scorer schedule, and its own baselines — which is exactly
what lets the app distinguish agents and a supervisor/MAS from one another.

Register the agent so the framework knows its experiment:

```python
from ail.registry import Agent, AgentRegistry

# Register "my_agent" -> a dedicated experiment id.
# A multi-agent / supervisor system is just another agent with its own experiment.
registry = AgentRegistry.load()              # reads config/agents.yaml
registry.add(Agent(name="my_agent", experiment_id="<your_experiment_id>"))
```

Mirror it in `config/agents.yaml` (a test enforces the YAML and the in-code seed
stay in sync). The app's **agent switcher** then lists every registered agent.

---

## Step 1 — "Upload" the agent: start tracing into that experiment

You have two ways to get traces into the experiment. **Either** works — the
framework's ingestion is deliberately **producer-agnostic**: `ail.ingest`
reads *any* `mlflow.entities.Trace` in the experiment via `mlflow.search_traces`
and detects the producer/model **best-effort** (never required). A trace that
lands in the experiment is a trace the optimizer can use, no matter how it got
there.

### Path A — native MLflow autolog (simplest, if your stack has it)

Point tracking at your workspace + experiment, then enable autolog for your
framework. These are standard MLflow calls — use the one for your stack:

```python
import mlflow
mlflow.set_tracking_uri("databricks")          # Databricks-managed MLflow
mlflow.set_experiment(experiment_id="<your_experiment_id>")

mlflow.langchain.autolog()      # LangChain / LangGraph
# mlflow.openai.autolog()       # OpenAI SDK
# mlflow.anthropic.autolog()    # Anthropic SDK
# mlflow.dspy.autolog()         # DSPy
# Claude Code:  `mlflow autolog claude`   (see GETTING_STARTED.md)
# Codex:        @mlflow/codex notify hook  (see CONNECT_CODEX.md)
```

Now run your agent normally. Every session is traced into the experiment.

### Path B — OpenTelemetry (OTEL), for anything else

If your agent already emits **OpenTelemetry** spans, you do **not** need to
re-instrument it. MLflow can ingest OTLP, so the play is:

1. **Use OTEL going forward** — configure your agent's existing OTLP exporter to
   export to the MLflow experiment's tracing endpoint, **or**
2. **Import a backlog of OTEL traces** you already have into the experiment.

> **Honest note on exact wiring:** the precise OTLP endpoint, headers, and
> exporter snippet depend on your **MLflow version** and whether you're on
> Databricks-managed MLflow or self-hosted. Get the current, version-correct
> exporter config from **MLflow's own Tracing → OpenTelemetry docs** rather than
> from a snippet pasted here — an endpoint copied from a doc that has since
> changed is worse than no snippet. What this framework guarantees is the
> **read** side: once OTEL spans land in the experiment as MLflow traces,
> `ail.ingest` consumes them with zero extra code (the producer simply shows up
> as best-effort / unknown, which is fine — L0 token/cost/redundancy metrics
> come straight off the span data).

### Verify the connection worked

The honest "did my upload take?" check is just: **are traces landing?**

```python
import mlflow
mlflow.set_tracking_uri("databricks")
traces = mlflow.search_traces(experiment_ids=["<your_experiment_id>"], max_results=5)
print(len(traces), "traces visible")   # > 0 means the framework can see your agent
```

(Optionally tag them per agent — `ail.agent = my_agent` — so cohorts/filters
isolate this agent cleanly; see Stage 1b in [GETTING_STARTED](GETTING_STARTED.md).)

---

## Step 2 — Collect enough traces (the part only data unblocks)

This is the requirement to be blunt about up front: **with zero traces, expect
zero improvement claims.** Each capability has a real, code-enforced data gate
(`ReadinessThresholds` in `src/ail/readiness/compute.py` — these are the actual
defaults, not aspirational numbers):

| You have… | What unlocks | Threshold (default) |
|---|---|---|
| **≥ 10 traces** | L0 token/cost/redundancy **baseline** + **RLM/HALO** deep review of the big traces (failure-mode diagnosis) | `baseline_min_traces = 10` |
| **≥ 20 human labels** | A **MemAlign-aligned judge** — calibrate a judge to *your* quality bar (the quality unlock) | `quality_min_labels = 20` |
| **≥ 50 traces** (+ frozen suite) | Enough statistical power to **prove** a token/cost improvement (the leaderboard goes green) | `prove_min_traces = 50` |
| **≥ 50% of traces scored** | Judges are actually running, not just registered | `scored_coverage_floor = 0.5` |

**Reconciling "a minimum of ~30 traces":** 30 is a sensible *starting* target —
it's comfortably past the **10** needed to run RLM/HALO and produce the **20**
labels that calibrate a MemAlign judge, so at ~30 traces you can genuinely start
the **MemAlign + RLM + GEPA** loop. But be precise about what "30" does and
doesn't buy: it is **enough to start optimizing and to align a judge**, and **not
yet enough to statistically *prove* a token/cost reduction** — that needs **50**
(token distributions are heavy-tailed; a "50% cut" measured on a handful of
traces is noise, not signal). The app shows the real measured delta before 50,
but stamps it **amber ("controlled proof · collecting")**, never green, until the
prove-floor clears. That refusal is the point.

**Two ways to reach the threshold:**
- **Import an OTEL backlog** (Step 1, Path B) — fastest if you already have
  traces from prior runs.
- **Generate them** — run your agent on ~30+ representative tasks. Variety
  matters more than volume: a spread of easy/hard, short/long sessions teaches
  the judges and RLM far more than 30 near-identical runs.

### Preflight: am I ready?

Don't guess where you stand — ask. One command reads your experiment and prints,
per gate, exactly how far you are from each unlock:

```bash
ail-readiness <YOUR_EXPERIMENT_ID> --profile <profile> --warehouse-id <wh>
# scope to one agent in a shared experiment:
ail-readiness <YOUR_EXPERIMENT_ID> --cohort-tag <agent-name>
```

It counts your traces, human labels, and scored-coverage and runs them through
the **same** `compute_readiness` the app uses, so the output is the real verdict,
not an estimate:

```
GATE                        HAVE   NEED   GAP   STATUS
baseline / diagnosis (RLM)  28     10     0     READY
prove an improvement        28     50     22    NOT READY
frozen Task Suite           yes    yes    —     READY
MemAlign labels             12     20     8     NOT READY
...
Unlocked now: RLM+diagnosis: READY; MemAlign judge: need 8 more labels; prove a total_tokens win: need 22 more traces
```

It is **fail-closed**: a *not-ready* gate is a normal exit (the refusal is the
point), and it never prints a green verdict the readiness module didn't return.
It exits non-zero only when it can't reach the trace store at all — and then tells
you which profile/warehouse to fix (the UC trace store needs `CAN_USE`).

---

## Step 3 — Now the self-optimization loop is unblocked

Once Step 2's gates are met, the rest of the workflow (detailed in
[GETTING_STARTED](GETTING_STARTED.md)) becomes real for this agent:

1. **Label ~20 traces** (Stage 3) → align a **MemAlign** judge to your quality
   bar; the app tracks judge-vs-human agreement and distrusts an uncalibrated
   judge by default.
2. **Run RLM/HALO** (Stage 4) on the large traces → ranked, recurring asset
   recommendations.
3. **Prove a lever** (Stage 5b) WITH-vs-WITHOUT on the frozen suite (fail-closed:
   a crashed or non-improving candidate is never a "win"), then **GEPA**
   (Stage 5) to evolve the prompt/skill, and **Stage 6** to generate helper
   assets (e.g. a UC metric-view).

The app's **readiness panel**tells you, per agent and per goal, exactly how many
more traces or labels you still need — so "not ready yet" is always an explicit,
actionable state, never a silently fabricated green number.

---

## TL;DR

1. **Register the agent → it gets its own experiment.**
2. **"Upload" = start tracing into that experiment** (native autolog *or* OTEL —
   ingestion is producer-agnostic).
3. **Collect data:** ~10 traces to diagnose, ~20 labels to align a judge,
   **50** to *prove* an improvement. Import an OTEL backlog or generate ≥30 to
   get the MemAlign/RLM/GEPA loop going.
4. The loop and the live comparison view do the rest — and refuse to claim an
   improvement until the data actually supports it.
