# Cohort recommendation planner and decision memory

> **Status: core implementation plus Managed Memory bridge implemented.** The hosted planner uses governed
> evidence, cohort, pattern-event/state, action-lineage, and outcome tables plus a
> structured planning tool. A Unity Catalog Managed Memory store now supplies bounded,
> supplemental cross-cohort retrieval; the Delta tables remain authoritative. The former opaque
> `agent_recommendation_memory.memory_json` snapshot is no longer read or written.
> Embedding-based retrieval, organic version-outcome attribution, and the pattern UI
> are follow-on slices; current queue coverage uses canonical keys plus conservative
> title similarity, and human/Tier-2 outcomes are synchronized now.

## 1. Desired behavior

The recommendation planner is a scheduled Databricks Job. It does not make a
recommendation for each trace. It waits for a configurable cohort of at least ten
completed subject traces, reads their RLM/HALO and judge feedback together, compares
the findings with durable pattern history and the full approval queue, and then
decides whether any new approval card is warranted.

The planner must:

- review at least `recommendation_planner_min_traces` distinct subject traces in one
  planning pass (default `10`);
- treat all feedback for those traces as one cohort, not as independent tasks;
- retain low-confidence and lower-priority signals as pattern observations instead
  of turning them into cards;
- create only a small configurable number of distinct, top-priority cards (default
  `3`);
- enrich an existing pattern/action when the queue already covers it;
- keep recommendation categories open-ended; examples are guidance, never an
  allowlist of recommendation types;
- preserve every human queue decision; the planner never marks an approval rejected;
- learn from approved, rejected, applied, superseded, and measured outcomes; and
- make all evidence, state transitions, and decisions queryable in Unity Catalog.

An empty recommendation batch is a successful and expected result. It means the
cohort was learned from but did not justify another interruption for the user.

## 2. Three complementary memory layers

The existing `agent_memory` table and the new recommendation memory solve different
problems and must remain separate.

| Memory | Purpose | Unit | Consumer |
|---|---|---|---|
| Advisory memory (`agent_memory`) | Distilled behavioral guidance that may be injected into an agent | guideline | target agent / intervention |
| Recommendation decision memory (Delta) | Authoritative record of what the planner observed, proposed, what the human decided, and what happened afterward | cohort, pattern, action, outcome | hosted recommendation planner and approval UI |
| UC Managed Memory | Bounded retrieval summaries of prior cohorts, decisions, rejection reasons, and measured outcomes | current state + compact cohort summary | hosted recommendation planner only |

The recommendation layer should reuse `ail.memory.assessments`, the reserved-pool
provenance wall, SQL execution helpers, deterministic IDs, and fail-closed MERGE
semantics. It should not store its state in `agent_memory` or inject planner decision
history into the target agent.

### Managed Memory safety boundary

Managed Memory is a Beta, REST-only retrieval layer, not a replacement database:

- the framework derives one stable opaque scope from `agent_name + experiment_id` in
  trusted code; the model can never choose a scope;
- `/memories/recommendations/state.md` is an idempotently replaced summary of current
  patterns plus every proposal kind (skills, metric views, prompts, reverts, and
  open-ended tasks), decisions, rejection reasons, applied artifacts, and measured outcomes;
- `/memories/recommendations/cohorts/<cohort_id>.md` records compact learned patterns
  and candidate actions without raw trace bodies or trace IDs;
- retrieval follows the Beta-safe list/get pattern: it always fetches the state entry,
  lists bounded cohort metadata, gets the most recently updated entries, and then
  enforces a hard prompt character budget;
- the prompt labels retrieved text as potentially stale, untrusted historical data;
  it cannot override current evidence or the human-owned queue; and
- Managed Memory errors are fail-soft. The governed Delta cohort still commits and a
  later run can rebuild the state summary from authoritative tables.

## 3. Data flow

```text
RLM/HALO + judge assessments
              |
              v
      evidence ledger  -----> cohort builder (>= 10 complete traces)
                                      |
                                      v
                     +----------------------------------+
                     | hosted Claude recommendation agent|
                     | cohort + patterns + queue/outcomes|
                     +----------------------------------+
                         |                         |
                         v                         v
                 pattern/event updates      top distinct actions
                         |                         |
                         +----------+--------------+
                                    v
                         human-owned approval queue
                                    |
                                    v
                     decisions, execution, measured impact
                                    |
                                    +----> outcome memory
```

Per-trace evaluation remains appropriate: judges and HALO attach feedback to each
trace. Only recommendation planning is cohort-level.

## 4. Governed Unity Catalog state

Delta does not enforce primary keys, so every key below is enforced by deterministic
IDs plus `MERGE`. All tables carry `agent_name` and `experiment_id`; a combined
opaque `scope` must not be the only queryable identity.

### `agent_recommendation_cohorts`

One immutable evidence snapshot and lifecycle row per planning pass.

Key fields:

- `agent_name`, `experiment_id`, `cohort_id`, `cohort_sequence`;
- `status`: `forming`, `ready`, `planning`, `committed`, or `failed`;
- `min_traces`, `trace_count`, `assessment_count`, `trace_ids`;
- `evidence_cutoff_at`, `queue_snapshot_at`;
- `planner_model`, `planner_prompt_version`, `planner_run_id`;
- `created_at`, `started_at`, `completed_at`, `error`.

`cohort_id` is a deterministic hash of agent identity, ordered trace IDs, and their
frozen evidence hashes. Retrying the same cohort cannot create a different cohort.

### `agent_recommendation_evidence`

An append-only ledger of the exact feedback available to planning.

Key fields:

- `evidence_id`: hash of normalized assessment identity and content;
- `trace_id`, `cohort_id`, `assessment_name`, `source_signal`;
- `value`, `comment`, `metadata_json`, `assessment_created_at`;
- `subject_or_reviewer`, `reserved_pool`, `ingested_at`.

The ingestion watermark advances after evidence is safely merged into this ledger,
even when fewer than ten traces are ready. This removes the interim design's need to
re-read the same unbounded assessment window while it waits for a cohort.

### `agent_recommendation_patterns`

The materialized current state of each stable, cross-trace issue or opportunity.

Key fields:

- `pattern_id` and stable `canonical_key`;
- `category`, `title`, `root_cause`, `status`;
- `first_seen_cohort_id`, `last_seen_cohort_id`, `cohort_count`;
- `distinct_trace_count`, `recent_trace_count`, `recent_prevalence`;
- `severity`, `confidence`, `trend_score`, `trend_label`;
- `current_action_id`, `summary_embedding`, `created_at`, `updated_at`.

Pattern status evolves through:

```text
emerging -> active -> queued -> monitoring -> resolved
               ^          |          |           |
               |          |          +-> regressed
               +----------+-------------- dormant
```

Rows are never deleted because they became low priority. A dormant or resolved
pattern can become active again with new evidence.

### `agent_recommendation_pattern_events`

Append-only history explaining every pattern transition.

Key fields:

- deterministic `event_id`, `pattern_id`, `cohort_id`;
- `event_type`: `created`, `reinforced`, `contradicted`, `merged`, `split`,
  `queued`, `resolved`, `regressed`, or `dormant`;
- `evidence_ids`, `source_trace_ids`, `observation_summary`;
- cohort-local `severity`, `confidence`, and `created_at`.

This table is the audit trail. `agent_recommendation_patterns` is a rebuildable
materialized view of it.

### `agent_recommendation_actions`

Planner-side lineage for a recommended action and its corresponding queue card.

Key fields:

- `action_id`, stable `canonical_action_key`, `category`, `title`, `plan`;
- `status`: `candidate`, `queued`, `approved`, `rejected`, `applied`,
  `superseded`, `monitoring`, or `resolved`;
- `proposal_id`, `first_proposed_cohort_id`, `last_supported_cohort_id`;
- `supersedes_action_id`, `merged_into_action_id`;
- `human_decided_at`, `applied_at`, `created_at`, `updated_at`.

### `agent_recommendation_action_patterns`

A many-to-many link between actions and the patterns they address. Its relation is
`addresses`, `covered_by`, or `follow_up`. This is what allows one broad action to
cover several patterns without creating one queue card per pattern.

### `agent_recommendation_outcomes`

Append-only observations after verification or application.

Key fields:

- `outcome_id`, `action_id`, `proposal_id`, `observed_at`;
- `source`: `tier2_verification`, `organic_version_comparison`, or `human_decision`;
- `metric_name`, `baseline_value`, `candidate_value`, `delta`;
- `result`: `improved`, `no_change`, `regressed`, or `inconclusive`;
- `n_traces`, `window_start`, `window_end`, `details_json`.

The current proposal, lineage, and version-comparison tables remain authoritative.
The outcome sync records the facts the planner needs; it does not invent a second
approval or apply mechanism.

## 5. Cohort formation

One scheduled firing performs these steps independently for each registered agent:

1. Upsert every new RLM and enabled-judge assessment into the evidence ledger.
   Enabled judges are discovered from the agent's registered MLflow scorers/goal
   configuration; the recommendation path must not hardcode the initial four judge
   names.
   Exclude HALO reviewer traces and apply the same frozen-suite / human-anchor
   provenance wall used by advisory memory.
2. Mark a subject trace eligible only after it has a successful `rlm_review`.
3. Wait for all enabled judge results known for that agent. To avoid waiting forever
   on a failed scorer, snapshot the available judge results after a configurable
   `recommendation_judge_grace_minutes` (default `30`). Late feedback is retained as
   an amendment and joins the next normal cohort; it never triggers a one-trace pass.
4. If fewer than `recommendation_planner_min_traces` eligible, unassigned traces
   exist, stop successfully after ingestion. Do not call the planning model.
5. Otherwise select the oldest eligible traces, including all available up to
   `recommendation_planner_max_traces` (default `25`), and freeze a deterministic
   cohort snapshot.

An ingestion watermark and cohort membership replace one assessment timestamp as
the source of planning truth. Pagination must drain all pages up to a bounded safety
cap; a SQL `LIMIT` must never silently skip older evidence.

## 6. Planning contract

The agent receives:

- the complete frozen cohort grouped by trace, including HALO verdict structure and
  judge rationales;
- computed cohort facts such as prevalence, cross-signal agreement, and change from
  recent cohorts;
- relevant `emerging`, `active`, `queued`, `monitoring`, and `regressed` patterns;
- semantically similar dormant/resolved patterns retrieved by embedding;
- all pending/approved actions plus recent rejected/applied/superseded actions; and
- measured outcome observations for related actions.

It does not submit an entire replacement memory document. It must call a structured
tool once with:

1. `pattern_observations`: grounded `create`, `reinforce`, `contradict`, `merge`, or
   `split` operations referencing cohort evidence IDs and existing pattern IDs; and
2. `action_candidates`: ranked actions referencing one or more resulting patterns,
   with a stable canonical action key, open-ended category, and concrete
   implementation plan. `AGENT_TASK` remains the queue envelope, not a restriction
   on the kinds of changes the planner can recommend.

The tool validates all evidence and pattern references, stages the result, derives
stable IDs, and performs idempotent MERGEs. The model may classify and reason; it may
not directly write SQL, replace memory wholesale, delete evidence, or change a human
queue status.

If tool validation or a write fails, the cohort remains retryable and no later cohort
is formed. Deterministic IDs make a partial cross-table retry converge without
duplicate patterns, events, actions, or proposals.

## 7. Priority and card eligibility

The agent decides priority using deterministic features supplied by the framework:

- prevalence across distinct traces, never assessment count;
- recurrence across cohorts;
- severity of the failure/opportunity;
- agreement between HALO and judge signals;
- recent trend (`rising`, `stable`, `falling`, or `new`);
- expected impact and implementation feasibility; and
- whether a queued or already-applied action covers the same root cause.

Default guardrails:

- a normal card requires support from at least `3` traces in the current cohort, or
  at least `5` traces across at least `2` cohorts;
- a security, privacy, or data-loss issue may use an explicit critical exception,
  but is still considered only after the full cohort is reviewed;
- no more than `recommendation_planner_max_recommendations=3` new cards per cohort;
- every card must address at least one durable pattern and cite its evidence; and
- lower-ranked or ineligible patterns stay in memory.

These are promotion thresholds, not rejection rules. Nothing is silently discarded.

## 8. Queue-aware decision rules

| Existing queue/action state | Planner behavior |
|---|---|
| Same canonical action is pending | Link new patterns/evidence to it; create no card |
| A broader pending action semantically covers the pattern | Record `covered_by`; create no card |
| Approved but not applied | Keep accumulating evidence; do not compete with the approved action |
| Applied | Enter monitoring; use later cohorts/outcomes to mark improved, persistent, or regressed |
| Rejected | Preserve the rejection and the pattern; do not immediately paraphrase it |
| Superseded/merged | Follow the surviving action lineage |
| No action covers a top eligible pattern | A new card may be created |

A rejected action may be raised again only after a configurable cooldown (default
three cohorts), materially stronger/new evidence, and a materially different action
plan. The new card must link to the rejected predecessor and explain what changed.
This respects the human decision without pretending the underlying feedback ceased
to exist.

Semantic coverage is resolved in two stages: retrieve likely matches using canonical
keys and embeddings, then have the planner choose `same`, `broader`, `related`, or
`different`. Only high-confidence `same`/`broader` matches suppress a new card;
ambiguous matches are recorded for later inspection.

## 9. Learning from outcomes

After each cohort, the Job also synchronizes queue and execution state:

- human approval/rejection becomes action history, not a pattern verdict;
- Tier-2 verification attaches controlled evidence;
- applied changes enter `monitoring` until a configured number of organic traces is
  available;
- falling prevalence plus non-regressing metrics moves a pattern to `resolved`;
- persistent prevalence moves it back to `active` without duplicating the old card;
- a material rebound after improvement becomes `regressed` and may justify a linked
  follow-up action.

This closes the learning loop: the planner remembers not only what it recommended,
but which kinds of recommendations humans accepted and which changes actually helped.

## 10. Configuration

| Variable | Default | Meaning |
|---|---:|---|
| `recommendation_planner_min_traces` | `10` | strict minimum distinct completed traces per pass |
| `recommendation_planner_max_traces` | `25` | maximum traces in one bounded cohort |
| `recommendation_judge_grace_minutes` | `30` | wait for enabled judges after RLM completes |
| `recommendation_planner_max_recommendations` | `3` | maximum new cards per cohort |
| `recommendation_pattern_min_current_traces` | `3` | normal current-cohort support floor |
| `recommendation_pattern_min_total_traces` | `5` | cross-cohort support floor |
| `recommendation_pattern_min_cohorts` | `2` | cross-cohort recurrence floor |
| `recommendation_managed_memory_store` | `ail_recommendation_memory` | short or three-level UC Managed Memory store; empty disables it |
| `recommendation_managed_memory_top_k` | `10` | maximum state + recent cohort entries retrieved per planning cohort |
| `recommendation_managed_memory_max_chars` | `12000` | hard prompt budget for retrieved memory contents |
| `recommendation_pattern_window_cohorts` | `6` | recent window used for trend features |
| `recommendation_rejection_cooldown_cohorts` | `3` | minimum wait before reconsidering a rejected action |
| `recommendation_outcome_min_traces` | `20` | organic evidence floor before outcome classification |

The five-minute schedule can remain. Most runs will only ingest evidence and exit;
Claude is invoked only after a cohort is ready.

## 11. Safe migration from the interim planner

The existing approval queue is human-owned history. The migration must not reject,
archive, supersede, or rewrite any of the current rows automatically.

1. Create the new tables through the existing writer-owned bootstrap and additive
   column reconciliation path.
2. Snapshot the interim watermark and opaque `memory_json` for audit. Treat parsed
   legacy patterns as low-confidence hints until supported by new grounded evidence.
3. Backfill all existing approval rows into `agent_recommendation_actions`, retaining
   their current proposal IDs and statuses. Cluster likely duplicates only as
   suggested lineage; do not mutate the queue.
4. Backfill referenced trace evidence as `historical` and build an initial pattern
   bank in shadow mode. Shadow mode writes memory but emits zero proposals.
5. Run at least two successful shadow cohorts and inspect pattern merges, queue
   coverage, and retry behavior.
6. Cut over the existing Job to the structured planner and enable proposal writes.

Reducing the existing queue is a separate human action. The UI can later present
duplicate groups with a “keep this card and supersede these” control, but the planner
must never make that decision on the user's behalf.

## 12. Acceptance tests

- Nine eligible traces ingest successfully and never call Claude; the tenth creates
  exactly one cohort.
- Ten traces with ten unrelated weak signals create pattern observations and zero
  cards.
- One issue recurring on four traces creates one pattern and at most one card.
- The same issue in the next cohort strengthens the pending card and creates no new
  proposal.
- A broader pending action can cover multiple patterns without duplicate cards.
- RLM and judge rationales for every selected trace are present in the frozen cohort.
- Reviewer traces and reserved frozen-pool traces cannot seed patterns.
- A failed tool/write retries the identical cohort with no duplicate rows.
- Pagination cannot strand evidence beyond the per-query limit.
- Rejection preserves evidence and suppresses immediate paraphrases; only a grounded,
  materially changed follow-up may return after cooldown.
- Applied actions transition through monitoring to resolved, persistent, or regressed
  from real outcomes.
- Migration leaves every pre-existing proposal ID and status unchanged.

## 13. Implementation slices

1. **Ledger and schema:** writer-owned DDL, evidence ingestion, cohort formation,
   reserved-pool wall, deterministic IDs, retry tests.
2. **Structured pattern planner:** pattern/event reducer, semantic matching, tool
   contract, ranking features, and no-card success path.
3. **Queue and outcome sync:** action lineage, queue coverage, human decisions,
   version-comparison outcome observations.
4. **Shadow backfill and cutover:** index legacy queue/evidence, run two shadow
   cohorts, then replace opaque-memory writes.
5. **UI follow-on:** pattern/trend evidence on each card and human-driven duplicate
   consolidation.
