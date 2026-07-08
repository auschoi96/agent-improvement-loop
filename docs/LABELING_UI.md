# In-app labeling UI (L4) — design of record

> **Status: BUILT** (this PR). L4 is the last Phase-1 lane: the human-facing surface
> that **produces the HUMAN labels** the loop needs. It builds on the app's
> authenticated write-path (introduced by the Phase C approval control plane, see
> [`LOOP_CONTROLLER.md`](LOOP_CONTROLLER.md), and extended by the onboarding wizard,
> see [`ONBOARDING_WIZARD.md`](ONBOARDING_WIZARD.md)) and reuses the same fail-closed,
> identity-from-headers engine-bridge pattern.

## Why it exists

The eval spine is already built: judges are authored via `ail.judges` (L1) with a
`{{ trace }}` template and a label schema **whose name exactly matches the judge
name**, and a scheduled auto-align (L2, `ail.judges.auto_align`) runs MemAlign once
enough HUMAN labels accrue. But MemAlign has nothing to align against until a human
grades some traces. L4 is that input stage, moved into the app: a signed-in user
labels traces along the dimensions that have **registered judges**, so L2's
auto-align can pair the labels and align the judge.

## The one load-bearing invariant: the name-match

MemAlign pairs a human label to a judge **by name**. A submitted label is written as
an MLflow assessment with `source_type=HUMAN` and `name` **exactly equal to the target
judge's name**; a mismatch silently breaks alignment (`ail.judges.labeling` §, and
`ail.judges.auto_align.read_human_labels`, which finds a judge's labels by that name).

L4 never reinvents that write or that naming. The `label` action reuses
`ail.judges.labeling.record_label`, which logs an `mlflow.log_feedback` keyed by the
judge name, and it offers labeling **only** along the names of the registered judges
(read from `ail.judges.registration.list_registered_scorers`). A label whose name is
not a currently-registered judge is **refused**, not written — a label that could
never align is worse than no label.

## How it writes (grounded, not guessed)

The app is Node/TypeScript. The write goes through the **MLflow Python SDK inside a
Python subprocess** — exactly how the onboarding plugin performs its MLflow writes
(`ail.onboarding.service` shells out to `python -m ail.onboarding.service`, which uses
the MLflow Python client; it does **not** call the MLflow REST API from TypeScript).
L4 mirrors that: the `labeling` server plugin's bridge runs
`python -m ail.labeling.service`, writes the JSON action on stdin, and reads a typed
JSON result on stdout. Reusing the Python helper (`record_label`) is what guarantees
the sacred name-match convention is honored rather than re-encoded (and likely
mis-encoded) in TypeScript.

MLflow is pointed at the workspace the same way
`ail.ingest.mlflow_source.MLflowTraceSource` and
`ail.onboarding.experiment.MlflowExperimentClient` do it: tracking URI `databricks`,
registry URI `databricks-uc`, and the active CLI profile — or, in the deployed app,
ambient service-principal auth (`DATABRICKS_HOST`/`DATABRICKS_TOKEN` or the SP).

That subprocess is the **local-dev / self-hosted** transport. On the **deployed**
Node-only image the `ail` wheel is not importable, so the same routes instead call the
Databricks-managed MLflow **assessments REST API** directly from TypeScript (the write is
the wire-equivalent of `record_label`'s `mlflow.log_feedback`, name-matched and
`HUMAN`-sourced). See [Deployed labeling transport](#deployed-labeling-transport-node-native-mlflow-rest).
The name-match convention itself is never re-encoded: the deployed write only ever names a
judge the live scorer list confirms, and refuses anything else.

**Deployment note.** The deployed Databricks App image is Node-only (the `ail` wheel
runs as serverless Jobs), so the subprocess bridge is the local-dev / self-hosted
transport. The Node-only deployed transport is now **built** as `restLabelingBridge`
(`server/plugins/labeling/bridge.ts`): the same `LabelingBridge` seam over the
Databricks-managed **MLflow assessments REST API** instead of a subprocess. It is
selected by env (`AIL_LABELING_TRANSPORT=rest`, committed in `app.yaml`) exactly as the
approvals bridge selects its transport; the authenticated route, the action contract
below, and the client are unchanged. It is **not** a per-grade Databricks Job trigger:
labeling is rapid-fire, and job-startup latency per grade would be unusable — the REST
write is a single low-latency call. See [Deployed labeling transport](#deployed-labeling-transport-node-native-mlflow-rest)
below for how it maps to the same three responsibilities.

### Deployed labeling transport (Node-native MLflow REST)

The REST transport speaks three **grounded** Databricks-managed MLflow endpoints (the
exact endpoints/bodies were captured live from the MLflow 3 Python SDK and confirmed
end-to-end, not guessed):

- **list judges** — `GET /api/2.0/managed-evals/scheduled-scorers/{experiment_id}` (what
  `mlflow.genai.scorers.list_scorers` resolves to on a Databricks backend, so the
  registered-judge set matches the engine's). This is both the labeling **dimensions**
  and the **name-match set** the write validates against.
- **scan traces** — `POST /api/4.0/mlflow/traces/search-long-running` then poll
  `GET /api/4.0/mlflow/traces/search/operations/{id}`; the operation response carries the
  recent `trace_infos[]` with their `assessments` inline (the read side's progress +
  worklist).
- **write label** — `POST /api/4.0/mlflow/traces/{location}/{trace_id}/assessments` with
  `{ assessment_name, trace_id, source:{source_type:"HUMAN", source_id}, feedback:{value},
  rationale }` — the wire-equivalent of `record_label`'s `mlflow.log_feedback`. The
  response's `assessment_id` is the **only** proof of a real write.

All v4 endpoints take the app's `DATABRICKS_WAREHOUSE_ID` as the SQL warehouse. Auth is
the app's ambient service principal via the `@databricks/sdk-experimental`
`WorkspaceClient` (the same client the approvals bridge builds). The SP needs the same
experiment read + assessment write access the labeling engine needs; the `sql-warehouse`
resource grant already covers the warehouse.

**Fail-closed on the deployed transport (same guarantee, enforced in TS + tested):** a
label is `labeled` **only** when the write returns an `assessment_id`. A missing warehouse,
an unresolvable trace location, a scorer-list failure, an unknown judge name, a write
error, or a write with no returned id all yield an honest `refused`/`error` — never a
fabricated `labeled` — and when a dependency can't be confirmed the message points the
user to the MLflow Traces UI. The written assessment's name equals the judge name and its
source is `HUMAN` with the **authenticated** labeler (`x-forwarded-*`, never the body) as
`source_id`. The label **floor** is not hardcoded in TS: it is relayed from the Python
engine via `AIL_LABEL_FLOOR` (unset → the client renders a neutral `—`, never a fabricated
number). These are guarded in `server/plugins/labeling/bridge.test.ts`.

## Fail-closed / no fabrication

Every failure yields an honest `refused`/`error` — never a fabricated `labeled`:

- empty labeler → `refused` (also enforced server-side; see auth below);
- missing trace/name/value → `refused`;
- a name that is not a registered judge → `refused` (the name-match guard);
- the registered-judge set cannot be determined (backend missing, no read access) →
  `error` — the UI says so and offers **no** dimensions rather than inventing any;
- the `log_feedback` write fails (auth, permission, trace not found) → `error`, with
  the failure surfaced in the UI.

## Authenticated

The labeler identity is resolved **server-side** from the platform-injected
`x-forwarded-email` (preferred) / `x-forwarded-user` headers (mirrors the approvals
and onboarding plugins). A `labeler` supplied in the request body is ignored; an
unauthenticated request is `401`. The resolved identity is the assessment's
`source_id`, so labels are attributable.

## Two-tier — no fabricated numbers in TypeScript

The label floor is the readiness floor,
`ail.readiness.ReadinessThresholds.quality_min_labels`, surfaced verbatim on every
dimension as `label_floor`. The progress counts (`labels_so_far`, `remaining`) are
computed in Python. TypeScript renders whatever numbers the engine sends and **never
hardcodes, invents, or branches on the floor's magnitude** — a missing number renders
a neutral `—` placeholder. This is the exact trap caught in the onboarding wizard +
tutorial; `client/src/lib/labeling.test.ts` guards it with a sentinel floor unequal to
any real default.

## Read side

`dimensions` scans the experiment's recent traces once (through the ingest
`TraceSource` seam — a read-only trace query, not a metric recomputation in SQL) and,
per registered judge, counts how many scanned traces already carry a HUMAN label named
for it (progress toward the floor) and lists the traces still missing one (the
worklist). When the experiment has more traces than the scan limit the result reports
`scan_capped=true` and counts reflect the most recent traces — honest, and
conservative (it never over-reports readiness). The label value control (numeric 1–5,
pass/fail, or free-form) is a best-effort hint read from each judge's L1 label schema;
it never blocks labeling and falls back to a free-form field.

## Contract (what the bridge speaks)

`python -m ail.labeling.service` reads one JSON action on stdin and prints one typed
result on stdout (`src/ail/labeling/service.py`). The Node route injects the
authenticated `actor`.

- **`dimensions`** — `{ action, actor, experiment_id }` →
  `{ outcome: "dimensions", experiment_id, label_floor, dimensions[], traces[], scanned, scan_capped, summary }`
  or `{ outcome: "error", error }`.
- **`label`** — `{ action, actor, experiment_id, trace_id, name, value, rationale? }` →
  `{ outcome: "labeled", name, value, labeler, labels_so_far, label_floor, remaining, complete }`
  or `{ outcome: "refused", refused_reason }` / `{ outcome: "error", error }`.

## Files

- `src/ail/labeling/service.py` — the engine (pure orchestration + live wiring + CLI).
- `ail-self-optimizer/server/plugins/labeling/` — the authenticated Node plugin
  (`labeling.ts` routes + auth, `bridge.ts` subprocess transport, `index.ts`,
  `manifest.json`). Registered in `server/server.ts` and `appkit.plugins.json`.
- `ail-self-optimizer/client/src/pages/LabelingPage.tsx` +
  `components/LabelingPanel.tsx` + `lib/labeling.ts` — the page (agent-scoped),
  wired into `App.tsx` and `lib/navigation.ts`.

## Operational prerequisites

- At least one **registered** judge on the agent's experiment (`ail.judges`
  authoring/registration) — otherwise the page shows an honest "no registered judges
  yet" state.
- The app's Python environment can list scorers (`ail[agents]` / the
  `databricks-agents` backend) and read/write the experiment's traces + assessments.
