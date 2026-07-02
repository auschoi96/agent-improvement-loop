# The open-ended executor (Lane L7b-2) — preview before approve, commit the exact preview

> **What this answers:** "When a human approves an open-ended `AGENT_TASK` proposal,
> how does the framework actually *make* the change — safely, revertibly, and without
> shipping something different from what the human reviewed?"

This is the executor that carries out an approved **open-ended** change
(`ActionKind.AGENT_TASK`, L7b-1 — see `docs/PRODUCT_ARCHITECTURE.md` §7). Unlike the
pre-specified kinds (`METRIC_VIEW` / `SKILL_UPDATE` / `INSTRUCTION_UPDATE` /
`GEPA_PROMPT` / `REVERT`), an `AGENT_TASK` can be *anything* the agent decides — a new
tool, an edited/new skill, a new table or metric view, cached examples, a multi-file
refactor. The safety is not in constraining the agent; it is in the **wrapper**.

Everything is **Databricks-native, no git**: the produced change-set is
versioned/revertible via the L6 UC-Volume snapshot substrate
(`ail.versioning.snapshot`), never a git tree.

## The design: two clearly-separated, independently-tested halves

The whole point is *preview before approve* (a human decision, load-bearing): the human
must review the **concrete produced change**, not just the NL plan. So the flow splits
into two functions in `ail.executor.executor`, kept separate so a reviewer can
scrutinize each on its own:

### (1) `produce_preview(proposal, agent, …)` — pre-approval, **no live effect**

1. Require `agent.target_workspace` (fail-closed refuse if unset/unreadable) and an
   `AGENT_TASK` proposal carrying a `plan`.
2. **Copy** the target workspace into an isolated sandbox dir.
3. Run the Claude Agent SDK agent (the reused `ail.ingest.adapters.claude_code`
   adapter) with `cwd` = the **sandbox copy**, edit tools allowed, the proposal's
   `plan` as the prompt — so the agent makes its edits **in the copy only**.
4. Capture the concrete diff (sandbox-copy vs original) as `preview_diff`, and snapshot
   the produced state via L6 → `produced_change_ref`.
5. Record `preview_diff` + `produced_change_ref` back onto the proposal (so the app
   shows the human the **real** diff).

**Fail-closed:** an agent error, a change that produces nothing, an unreadable
workspace, a deletion (see below), or a snapshot that cannot persist → **no** preview is
written, an honest error is raised, and the live `target_workspace` stays byte-for-byte
**untouched** (the agent only ever edits the sandbox copy). Never a fabricated preview.

### (2) `commit_approved(proposal, agent, …)` — post-approval, **live**

- **Precondition (fail-closed):** the proposal is `APPROVED` **and** carries a
  `produced_change_ref` (the exact preview the human saw). Refuse otherwise.
- **The load-bearing safety invariant:** commit applies the **stored** produced change
  (from `produced_change_ref`). It **never re-runs the agent** — the SDK is
  non-deterministic, so a re-run at commit could produce a *different* change than the
  one that was approved. There is deliberately **no agent-runner parameter** on
  `commit_approved`.
- **Order:** snapshot the live workspace **first** (the revert point) → apply the stored
  change to live via L6's all-or-nothing restore → record the commit (snapshot refs +
  summary + approver). A failed apply cannot leave a half-applied tree (L6 verifies
  every object before writing any, and rolls back a mid-swap failure); **revert** = an
  L6 restore of the pre-change snapshot.

## The load-bearing invariants (and where they are tested)

| # | Invariant | Test (`tests/test_executor.py`) |
|---|---|---|
| a | `produce_preview` leaves the live workspace byte-for-byte untouched, even while the agent runs | `test_preview_leaves_live_workspace_untouched` |
| b | `commit_approved` applies the **stored** change and never re-runs the agent (no runner param; never builds one; applied bytes are the stored ones, not a re-run) | `test_commit_has_no_agent_runner_parameter`, `test_commit_applies_stored_change_never_reruns_agent` |
| c | Committed live bytes == the previewed produced bytes (`preview_diff` renders from the **stored snapshot**, not a mutable sandbox re-read) | `test_committed_equals_previewed`, `test_preview_diff_from_snapshot_ignores_post_snapshot_sandbox_mutation` |
| d | Snapshot-live-first crash-safety: a failed apply leaves the tree untouched, with the revert point already taken | `test_commit_snapshot_first_crash_safety` |
| e | Every fail-closed refusal (missing workspace, non-`AGENT_TASK`, non-`APPROVED`, missing/stale `produced_change_ref`, agent-produced-nothing, agent error, deletions, path escape) | `test_preview_refuses_*`, `test_commit_refuses_*` |
| f | No live SDK / MLflow / Databricks call — only injected seams | `test_no_live_seams_touched` (+ all tests use fakes) |

The runner's contract is pinned in `tests/test_agent_executor.py` (static-auth refusal,
fail-closed on an unreadable table, dry-run writes nothing, a real run
previews/skips/commits, the persistence SQL, and the row-count-checked writes).

## Cross-review hardening (BLOCKERs 1–5)

An independent cross-review found five safety issues; each is fixed and tested
(fail-without / pass-with):

1. **Sandbox symlink-escape (preview).** Escaping symlinks copied from the workspace
   are neutralized in the sandbox before the agent runs, so a write through a
   pre-existing symlink lands inside the sandbox, never through to a live/outside file;
   a produced file whose real path escapes the sandbox is refused
   (`test_preview_symlink_escape_does_not_write_outside`,
   `test_preview_refuses_agent_created_escaping_symlink`).
2. **Commit containment is realpath-based.** Containment resolves the workspace root and
   every target with `os.path.realpath` (symlinks in every parent component resolved),
   so a `<workspace>/link/evil` with a symlinked parent escaping the root is refused
   before any write (`test_commit_refuses_symlinked_parent_escape`).
3. **A committed pure-add is revertible.** The commit records the exact set of *added*
   file paths (`added_paths`); `revert_committed_change` restores the overwritten files
   (from `pre_change_ref`) **and deletes the added ones** — the revert of an addition is
   a delete (L6 restore cannot delete) — fail-loud and containment-checked. Wired to the
   runner as `--revert <proposal_id>` (`test_pure_add_commit_is_revertible`,
   `test_revert_restores_overwritten_and_deletes_added`,
   `test_revert_refuses_added_path_outside_workspace`,
   `test_revert_mode_removes_recorded_added_files`).
4. **`preview_diff` reflects the stored snapshot bytes.** The diff (and the returned
   change list) render from the snapshot blobs — the exact bytes `produced_change_ref`
   holds and commit applies — not a fresh sandbox re-read a background process could
   mutate, so what the human approves is byte-identical to what commit applies
   (`test_preview_diff_from_snapshot_ignores_post_snapshot_sandbox_mutation`).
5. **Guarded UPDATEs verify affected-row-count.** `write_preview` / `mark_committed` run
   through a row-count-checked path (`num_affected_rows` via `_query_rows`, leaving
   `ail.publish._execute` unchanged for other callers): a zero-row guard match (or an
   unconfirmable count) is a **failure**, not a success — the runner never prints
   PREVIEWED / COMMITTED on a no-op, and a zero-row status-mark after a live commit is
   surfaced as *committed-but-unrecorded* (`test_write_preview_zero_rows_fails_closed`,
   `test_mark_committed_zero_rows_fails_closed`,
   `test_commit_zero_row_status_mark_is_committed_but_unrecorded`).

## Preview writes are confined by the Claude Agent SDK's native filesystem sandbox

The preview agent in `produce_preview` runs with edit + Bash tools, so symlink-neutralization
(BLOCKER 1) alone cannot stop an *absolute-path* write escaping the sandbox copy during a
preview that must be side-effect-free. The preview therefore also runs under the Claude Agent
SDK's **own native filesystem sandbox**: `ClaudeAgentOptions(sandbox=…)` scopes the agent's
writes to the sandbox copy directory (write tools allowed only under `{sandbox}/**`,
`permission_mode="dontAsk"`), so the agent cannot write outside the copy regardless of the path
form it attempts.

This is **fail-closed**: the adapter validates that the installed `claude-agent-sdk` actually
exposes the native sandbox option, and if it does not, `produce_preview` refuses to run and
returns an ERROR result ("refusing to run unsandboxed") rather than silently running an
unsandboxed preview. Symlink-neutralization (BLOCKER 1) remains in place as defense-in-depth.
Covered by `test_preview_sdk_sandbox_blocks_absolute_write_outside`.

## Known residuals & trust model

The preview escape above is closed; what remains below is the commit/revert path. B2/B3 close the concrete symlink-parent escape and pure-add revert bugs, but they do not
turn the local filesystem into an adversarially safe object store. `commit_approved` and
`revert_committed_change` validate target containment before writing/deleting; a malicious
process on the same host with write access to the target workspace could still race that
validation by swapping a path component between the check and the filesystem operation
(classic filesystem TOCTOU). That residual is accepted for L7b-2 under the
**trusted-companion-host** model: the executor runs as the deployer on a local workspace
the deployer controls, not on an untrusted multi-tenant checkout.

A future hardening pass belongs in the L6 versioning layer, where all restore/delete
filesystem primitives can share it: open parent directories by fd, walk path components
with no-follow semantics, and use `O_NOFOLLOW` / dir-fd operations where the platform
supports them. Until that exists, the executor's contract is containment-checked and
fail-closed against stale/tampered manifests and existing symlink escapes, but not
resistant to same-host malicious races.

## Injectable seams (so everything is offline-testable)

`produce_preview` / `commit_approved` take their side-effects as seams, so the module
imports offline and every test runs with fakes:

- **`agent_runner`** — the Claude Agent SDK (`AgentRunner`, the `AgentAdapter` surface;
  defaults to `ClaudeCodeAdapter`, lazily built). **Only** `produce_preview` takes one.
- **`volume_client`** — the L6 UC-Volume client (no implicit live path).
- **`preview_writer`** / **`commit_recorder`** — persistence, supplied live by the
  runner.

## How the produced snapshot is applied to the live paths

`produce_preview` snapshots the produced (post-edit) bytes from the sandbox via
`snapshot_paths`, then remaps each manifest entry's `original_path` from the sandbox
back to the **live** workspace and re-writes the manifest — so a later `restore_snapshot`
at commit writes the produced bytes to the **live** paths (blobs are content-addressed,
so they are valid regardless of which path they were read from). `commit_approved`
re-validates (via `os.path.realpath`, symlinks resolved) that every stored target path
is **inside** the workspace before applying, so a tampered or symlinked manifest can
never write outside the agent's own source.

## Revert — restore overwritten files, delete added ones

A committed change is revertible in **two** parts, because L6 restore can *write* but not
*delete*: `commit_approved` records both the `pre_change_ref` (the pre-commit snapshot of
the files it **overwrote**) and `added_paths` (the files it **created**).
`revert_committed_change` restores the former (L6 restore, all-or-nothing) and deletes the
latter — so even a pure-addition commit (no `pre_change_ref`) is fully revertible. It is
fail-loud (a delete it cannot perform raises, naming the partial state) and
containment-checked (never deletes outside the recorded workspace). The runner exposes it
as `ail-agent-executor --revert <proposal_id>`, reading the recorded change from
`agent_executor_commits`.

## The Databricks-native tradeoff: deletions are refused (fail-closed)

The L6 restore substrate versions file **writes** (add/modify): its restore recreates
files but cannot *delete* them. So a produced change that **deletes** files is refused
fail-closed in `produce_preview` — committing it would silently fail to remove the
files, which would violate "committed change == approved preview". This mirrors
`docs/PRODUCT_ARCHITECTURE.md` §8's explicit "a Volume snapshot is coarser than git"
tradeoff; a git-backed executor (which can express deletions and line-level reverts) is
the documented future option. `.git` and tool caches
(`__pycache__`/`.mypy_cache`/…) are excluded from the sandbox copy and the change diff,
so they never register as a change.

## The local companion runner — `ail-agent-executor`

`ail.jobs.agent_executor` is the deployer-run companion (Claude Agent SDK compute, **not**
Databricks serverless — same as the planner) that drives the lane end-to-end against the
app's `agent_proposed_actions` table. One run:

1. **polls PENDING `AGENT_TASK` proposals** with no preview yet → `produce_preview`,
   recording the diff + ref back onto the row. A proposal that **already** carries a
   preview is **skipped** (never re-previewed — that would move the diff out from under
   a reviewer);
2. **polls APPROVED `AGENT_TASK` proposals** → `commit_approved`, applying the stored
   change and advancing the row to `applied`;
3. **surfaces every step** to the operator (what it previewed / committed, and the
   fail-closed reason for anything skipped/refused).

It reuses the lane-3b "apply_service reader" (`_row_to_proposal` / `_query_rows`) for the
proposal reads, `ail.publish`'s SQL primitives for the targeted writes, and records each
commit to a dedicated append-only `agent_executor_commits` audit (the revert-point +
approved change-set snapshot refs, the file count, the approver, and when). It does
**not** touch `agent_prompt_lineage` — an arbitrary file change-set is not a prompt
version.

**Auth — a static token, matched to the workspace host (the hard-won lesson):** the
runner is a long-lived local process, so it reuses the companion's `resolve_static_auth`
— a **static** `DATABRICKS_TOKEN` pinned to `DATABRICKS_HOST`, dropping any ambient
`DATABRICKS_CONFIG_PROFILE`, refusing to run without one (a `--profile` OAuth login's
mid-run refresh cannot persist from a background process).

```bash
export DATABRICKS_HOST=https://<workspace-host>
export DATABRICKS_TOKEN=<pat-or-static-token>
python scripts/run_agent_executor.py \
    --agent claude_code \
    --registry config/agents.yaml \
    --warehouse-id <sql-warehouse-id> \
    --volume-root /Volumes/<catalog>/<schema>/<volume>/ail_snapshots \
    --dry-run
```

`config/agents.yaml` must set `target_workspace` for the agent (the in-code seed has
none, so the executor fails closed against it — by design).

## Boundaries (what this lane does *not* touch)

- The **deterministic apply path** (`ail.loop.apply`) still **refuses** an `AGENT_TASK`
  fail-closed and it stays excluded from `_EVIDENCE_ONLY_APPLYABLE_KINDS` — an
  open-ended agent change never ships without the executor + a human diff-preview. This
  lane is a separate companion; it does not modify that engine, the app client, the
  prover, or the L6 / L7b-1 sources.
- Routing an `AGENT_TASK` approval so the proposal reaches `status = 'approved'` (rather
  than the deterministic apply's refusal) is an app-side concern out of this lane's
  scope; the executor's contract is simply: an approved `AGENT_TASK` carrying a
  `produced_change_ref` is committed from its stored snapshot, with the authenticated
  approver read from the `agent_action_decisions` audit (falling back to the configured
  operator identity).
```
