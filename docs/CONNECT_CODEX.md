# Connect Codex → MLflow Tracing

**Status:** initial · **Module:** `ail.ingest.adapters.codex` · **Native integration:** `@mlflow/codex` (npm) · **Target experiment:** `660599403165942`

This is the operations guide for turning on Codex → MLflow tracing for a
deployment, pointed at a target experiment, using **MLflow's native Codex
integration**. It also documents the Python-native trace-capture path this repo
ships (`ail.ingest.adapters.codex`) and the exact gotchas for **our** Codex
worker — the codex-native CLI harness (OpenAI Codex CLI) running under Omnigent,
authenticating to GPT‑5 through the Databricks AI Gateway.

> [!IMPORTANT]
> **Two different auth surfaces.** How Codex talks to the *model* (the Databricks
> AI Gateway, `…/ai-gateway/codex/v1`) and how Codex traces reach *MLflow* are
> **separate** and may even point at different workspaces. Enabling tracing does
> not touch the gateway/model auth, and vice-versa. Keep them straight when
> debugging.

---

## 1. How MLflow's native Codex integration works

MLflow's native Codex integration is **not** Python autolog and **not** a
wrapper. It is a **Node package, `@mlflow/codex`**, that rides Codex's
**`notify` hook** (transcript ingestion):

1. You register the hook in Codex's `config.toml`:
   `notify = ["mlflow-codex", "notify-hook"]`.
2. After **each agent turn**, Codex invokes that program with an
   `agent-turn-complete` JSON payload (thread id, turn id, cwd, input messages,
   last assistant message).
3. The hook reads MLflow config (precedence: **env vars → `./.codex/mlflow-tracing.json` → `~/.codex/mlflow-tracing.json`**),
   calls `@mlflow/core` `init({ trackingUri, experimentId })`, and logs a trace.
   For richer detail it also parses the session's **rollout JSONL** transcript
   at `$CODEX_HOME/sessions/YYYY/MM/DD/rollout-*-<thread>.jsonl` (tool calls,
   token usage).

Codex also has a **newer "hooks" framework** (`SessionStart`, `PreToolUse`,
`Stop`, …) distinct from `notify`. As of `@mlflow/codex` 0.2.0‑rc.0 the MLflow
integration uses the **`notify`** mechanism, not the newer hooks.

### What lands in MLflow

A root **`codex_conversation`** span (`AGENT`), with child spans:

- **`llm_call`** (`LLM`) — one per assistant turn, with `model` and the
  reconstructed chat message history.
- **`tool_<name>`** (`TOOL`) — one per `function_call` / `function_call_output`
  pair (e.g. `tool_shell`), with `tool_name` / `tool_id`; failed shell commands
  (non‑zero `exit_code`) are marked failed.
- **Token usage** from the rollout's `token_count` events; **model** from
  `session_meta` / `turn_context`; session/user metadata on the trace.

The integration has **no custom-tag hook** — it cannot stamp `ail.agent=codex`
itself. See [§5 Tagging](#5-tagging-ailagent--codex).

---

## 2. Enable it for a deployment (target experiment `660599403165942`)

### 2a. Install the package (required at runtime)

```bash
npm install -g @mlflow/codex      # provides the `mlflow-codex` binary on PATH
```

The hook only runs if `mlflow-codex` is on the `PATH` of the process Codex
spawns it from.

### 2b. Write the config

Use this repo's helper — it writes the `notify` line **and** the
`mlflow-tracing.json`, and (unlike the stock CLI) accepts `trackingUri="databricks"`:

```python
from ail.ingest.adapters.codex import enable_codex_tracing

enable_codex_tracing(
    "660599403165942",          # target experiment id
    tracking_uri="databricks",  # Databricks-managed MLflow
    scope="user",               # ~/.codex (honors $CODEX_HOME); use "project" for ./.codex
)
```

This produces, under `~/.codex/`:

```toml
# config.toml  (notify prepended above any [table], as TOML requires)
notify = ["mlflow-codex", "notify-hook"]
```
```json
// mlflow-tracing.json
{ "trackingUri": "databricks", "experimentId": "660599403165942" }
```

It is **idempotent** (an existing `mlflow-codex` notify is left alone) and
**refuses to clobber** a different pre-existing `notify` entry unless you pass
`force=True`.

> [!IMPORTANT]
> **Why not `mlflow-codex setup`?** The stock CLI works for local servers, but
> its tracking‑URI validator **only accepts `http(s)` URLs** and rejects
> `databricks`. For Databricks-managed MLflow you must write
> `mlflow-tracing.json` directly — which `enable_codex_tracing` does. (`MLFLOW_TRACKING_URI=databricks`
> via env also works where env survives — but see [§4](#4-our-codex-worker-omnigent--gpt5).)

### 2c. Databricks-managed MLflow + workspace auth

The target experiment `660599403165942` lives in the workspace behind the
**`dais-demo`** profile (host `fevm-austin-choi-omni-agent`); its monitoring
warehouse is `7d1d3dbb3ba65f2a`.

With `trackingUri="databricks"`, `@mlflow/core` resolves Databricks credentials
the same way the Databricks tooling does — from `~/.databrickscfg` (a profile) or
`DATABRICKS_HOST` / `DATABRICKS_TOKEN`. Point it at the experiment's workspace:

```bash
# Refresh the dais-demo OAuth token (it currently needs a refresh):
databricks auth login --profile dais-demo

# Make dais-demo the default the Node SDK will pick up (it reads the DEFAULT
# profile unless told otherwise — see the live-verify note in §6):
export DATABRICKS_CONFIG_PROFILE=dais-demo
```

> [!NOTE]
> Whether `@mlflow/core` honors a **named** `~/.databrickscfg` profile (vs only
> `DEFAULT` / `DATABRICKS_HOST`+`DATABRICKS_TOKEN`) is a **live-verify** item
> (§6). If it only reads `DEFAULT`, either make `dais-demo` your `DEFAULT`
> profile or export `DATABRICKS_HOST`/`DATABRICKS_TOKEN` for that workspace.

---

## 3. What this repo ships (`ail.ingest.adapters.codex`)

Mirroring the Claude Code adapter, the Codex module carries two things:

- **`enable_codex_tracing(...)`** — the config writer in §2b.
- **A Python-native trace-capture path** — `normalize_codex_rollout(path_or_records)`
  reads the **same rollout JSONL** the native hook reads and projects it onto a
  `NormalizedTrace` (`producer="codex"`, tagged `ail.agent=codex`), and
  `CodexAdapter(AgentAdapter)` drives `codex exec` and captures that trace. This
  path is **independent of the Node package and of whether `notify` fires** — it
  is the fallback if the native hook can't be used, and the way a frozen task
  suite is replayed through Codex for apples-to-apples comparison.

```python
from ail.ingest.adapters.codex import normalize_codex_rollout
trace = normalize_codex_rollout(
    "~/.codex/sessions/2026/06/29/rollout-…-<session>.jsonl"
)  # -> NormalizedTrace(producer="codex", tags={"ail.agent": "codex"}, …)
```

> [!NOTE]
> The native `@mlflow/codex` hook is what writes traces **into MLflow**. The
> repo's normalizer produces the in-memory `NormalizedTrace` the loop consumes
> (cohorts / L0 / judges) and is the replay/capture path; it does not itself log
> to MLflow (by design — we use the native integration for that).

---

## 4. Our Codex worker (Omnigent → GPT‑5)

Our worker runs **`codex app-server`** under Omnigent. Three facts about how
Omnigent launches Codex change the enable steps materially:

1. **Per-session private `CODEX_HOME`.** Omnigent **copies** the user's
   `~/.codex/config.toml` into each session's private `CODEX_HOME` (and symlinks
   `auth.json`). → **Put the `notify` line in the user-level `~/.codex/config.toml`**
   (`scope="user"`); it then propagates to every session automatically.

2. **The env is filtered (allowlist).** Omnigent's `_clean_codex_env` passes only
   an allowlist of prefixes (`OPENAI_`, `HTTP_`, `CODEX_HOME`, `XDG_`, `LANG`, …)
   plus a few exact keys. **`MLFLOW_*`, `DATABRICKS_HOST`, `DATABRICKS_TOKEN`,
   `DATABRICKS_CONFIG_PROFILE` are stripped.** Consequences:
   - The **env-var config path does not work** inside Omnigent. You **must** use
     `~/.codex/mlflow-tracing.json` (the hook reads it via `os.homedir()`, and
     `$HOME` *is* preserved). `enable_codex_tracing(..., scope="user")` writes it
     there.
   - Databricks auth for the hook must come from **`~/.databrickscfg`** (HOME is
     preserved), not env. Refresh it with `databricks auth login --profile dais-demo`.

3. **`notify` in `app-server` mode is the #1 risk.** `@mlflow/codex` rides the
   `notify` hook, which is well-established for the TUI and `codex exec`. Whether
   it fires under **`codex app-server`** (what Omnigent runs) is **not confirmed
   from docs** and must be live-verified (§6). **If `notify` does not fire in
   app-server mode, fall back to the Python normalizer** (§3): the app-server
   still writes rollout JSONL (Omnigent does not use `--ephemeral`), so we can
   parse and ingest those transcripts ourselves.

---

## 5. Tagging `ail.agent = codex`

Cohorts separate Codex from `claude_code` traces by the `ail.agent` tag (see
[`COHORTS.md`](COHORTS.md)). The native `@mlflow/codex` integration cannot set
it, so there are two paths:

- **Repo capture path** — `normalize_codex_rollout` / `CodexAdapter` set
  `tags={"ail.agent": "codex"}` directly. Nothing more to do.
- **Native hook path (traces already in MLflow)** — backfill the tag:

  ```python
  from ail.ingest.adapters.codex import codex_cohort_tags          # {"ail.agent": "codex"}
  from ail.ingest.mlflow_source import apply_trace_tags

  apply_trace_tags(codex_trace_ids, codex_cohort_tags(), profile="dais-demo")
  ```

  Selecting which trace ids are Codex: filter on the root span / trace name
  **`codex_conversation`** (the native integration's name), or tag them in the
  MLflow UI. Note `apply_trace_tags` needs **write** access to the UC-backed
  trace store — expect `PermissionDenied` for a read-only identity, in which
  case tag in the UI instead.

---

## 6. Live-verify steps (auth-gated — do NOT block the build on these)

The build and tests are fully mocked; do not require a live run. When a human
with workspace access wants to confirm end-to-end:

```bash
# 0. Auth to the experiment's workspace (token currently needs a refresh).
databricks auth login --profile dais-demo
export DATABRICKS_CONFIG_PROFILE=dais-demo     # see §2c live-verify note

# 1. Enable tracing at user scope (writes ~/.codex/config.toml + mlflow-tracing.json).
python -c "from ail.ingest.adapters.codex import enable_codex_tracing; \
print(enable_codex_tracing('660599403165942', tracking_uri='databricks', scope='user'))"

# 2. Install the hook runtime.
npm install -g @mlflow/codex

# 3a. Confirm notify fires in the mode you run. For a quick check, exec mode:
codex exec --skip-git-repo-check "print hello"
#   Then look for a fresh rollout under ~/.codex/sessions/… and a new trace.
# 3b. For OUR worker, run a real Omnigent app-server Codex session and check
#     whether a trace appears — this is the app-server notify question (§4.3).

# 4. Verify the trace landed in the target experiment.
python -c "import mlflow; mlflow.set_tracking_uri('databricks'); \
print(mlflow.search_traces(experiment_ids=['660599403165942'], max_results=5))"
```

Checklist of the **two unknowns** to settle live:

- [ ] Does `notify` fire under `codex app-server` (Omnigent), not just `exec`/TUI?
- [ ] Does `@mlflow/core` authenticate to Databricks via a **named** `~/.databrickscfg`
      profile, or only `DEFAULT` / `DATABRICKS_HOST`+`TOKEN`?

If either is "no", the Python normalizer path (§3) ingests the rollout
transcripts the app-server still writes — no dependency on `notify` or the Node
SDK's auth.

---

## 7. Bonus — could the same mechanism capture Pi?

**Short answer: not the `@mlflow/codex` mechanism, and not OpenAI‑SDK/Agents‑SDK
autolog — but a transcript-ingestion path (like §3) is jerry-riggable.**

- **`@mlflow/codex` notify hook** is Codex‑CLI‑specific (it parses Codex's
  rollout format and rides Codex's `notify`). Pi is a different harness, so this
  does **not** capture Pi. There is no `@mlflow/pi` package.
- **`mlflow.openai.autolog()` / OpenAI‑Agents‑SDK autolog** instrument an
  **in-process Python** OpenAI client. Pi is a **separate CLI process** that
  speaks the gateway's **Anthropic Messages** surface by default (`api: anthropic-messages`),
  not the OpenAI SDK in our Python process — so OpenAI autolog won't see it.
  (`mlflow.anthropic.autolog()` has the same in-process limitation.)
- **What *is* jerry-riggable:** Omnigent's own code notes that, like Claude Code
  and Codex, **the Pi CLI persists a session rollout/transcript**. So the
  realistic MLflow path for Pi is **transcript ingestion** — a Pi rollout
  normalizer analogous to `ail.ingest.adapters.codex.normalize_codex_rollout`
  (or a future `@mlflow/pi`), **not** autolog. This is **assessment only** per
  the task; nothing here is implemented for Pi.

---

## References

- MLflow Codex integration: <https://mlflow.org/docs/latest/genai/tracing/integrations/listing/codex/>
- Codex hooks (the newer framework, distinct from `notify`): <https://developers.openai.com/codex/hooks>
- `@mlflow/codex` package (notify hook, rollout parsing): npm `@mlflow/codex`
- Cohorts / `ail.agent` convention: [`COHORTS.md`](COHORTS.md)
- Claude Code adapter (the pattern this mirrors): `src/ail/ingest/adapters/claude_code.py`
