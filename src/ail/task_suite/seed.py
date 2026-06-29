"""The curated ``v1-seed`` Task Suite, abstracted from the real L0 diagnosis.

These 22 tasks are derived **only** from committed artifacts —
``artifacts/example1_diagnosis.{md,json}`` and
``artifacts/l0_baseline_660599403165942.json`` (the deterministic L0 baseline
over experiment ``660599403165942``, 91 traces). No live MLflow trace content
was read (the v4 trace store routes reads through a SQL warehouse this identity
is not authorized for).

Each task is keyed to a real ``source_trace_id`` and its category is one of the
dominant patterns the diagnosis surfaces. Because the raw trace *content* is not
yet readable, each ``prompt`` is a **v1-seed reconstruction** of the task from
the trace's observable L0 profile (project path, tool mix, repeated targets),
and ``notes`` records the hard metrics plus what is derived vs. unknown. When
warehouse access lands, the prompts can be enriched from real trace content
without changing the schema (bump to a new artifact version, re-freeze).

:func:`build_seed_suite` is the single source of the curated content; the
committed ``eval/task_suite/v1/tasks.yaml`` is its frozen serialization, and a
test pins that the artifact and this builder agree (so the artifact cannot drift
from its curated source).
"""

from __future__ import annotations

from ail.task_suite.schema import Difficulty, Task, TaskCategory, TaskSuite

#: Content label of the seeded artifact. ``-seed`` marks that prompts are
#: reconstructions from L0 profiles, not verbatim trace inputs (see module docs).
SEED_VERSION = "v1-seed"

#: Fixed so the frozen artifact is byte-deterministic (re-running the builder
#: yields an identical hash). The L0 baseline was generated 2026-06-29.
SEED_CREATED_AT = "2026-06-29T00:00:00+00:00"

C = TaskCategory
D = Difficulty

SEED_TASKS: tuple[Task, ...] = (
    # ---- heavy_tail_high_token: the bimodal tail where the token spend lives ----
    Task(
        task_id="ts-001",
        category=C.HEAVY_TAIL_HIGH_TOKEN,
        difficulty=D.HARD,
        source_trace_id="a9b23c23c1efb90c576cd012f0a70522",
        prompt=(
            "In the omniagent/agent-framework repository, carry an end-to-end change to "
            "completion: update the Databricks governed-demo example config, validate it with "
            "repeated shell runs, and produce a written deliverable via the google-docs workflow."
        ),
        notes=(
            "Reconstructed from L0 profile (largest session in the corpus): 942,929 tokens, "
            "60 tool calls (35 Bash / 6 Write / 3 Edit), ~$6.21. Re-ran "
            "`cd .../omniagent/agent-framework` 27x; edited examples/databricks_governed_demo "
            "config.yaml 2x. Prompt is a v1-seed abstraction; enrich from trace content later."
        ),
    ),
    Task(
        task_id="ts-002",
        category=C.HEAVY_TAIL_HIGH_TOKEN,
        difficulty=D.HARD,
        source_trace_id="201506dc57618323b322df3a4c328f0f",
        prompt=(
            "In the ail wave0b-l0-metrics worktree, build the L0 deterministic metrics module "
            "(src/ail/metrics/l0_deterministic.py) and its report, iterating with tests until "
            "the numbers reconcile."
        ),
        notes=(
            "612,820 tokens, 90 tool calls (37 Bash / 12 Edit / 9 Write), ~$5.28. Re-ran "
            "`cd .../wave0b-l0-metrics` 20x; edited l0_deterministic.py 3x. v1-seed reconstruction."
        ),
    ),
    Task(
        task_id="ts-003",
        category=C.HEAVY_TAIL_HIGH_TOKEN,
        difficulty=D.HARD,
        source_trace_id="14da0deab1fee461c836a529d9f1e5ae",
        prompt=(
            "In the emerging-epl project, implement an API route end-to-end "
            "(server/routes/epl.ts) with the supporting edits, iterating until it compiles "
            "and passes."
        ),
        notes=(
            "549,300 tokens, 57 tool calls (26 Edit / 14 Bash), ~$4.15. Edited "
            "server/routes/epl.ts 6x. v1-seed reconstruction from observable targets."
        ),
    ),
    Task(
        task_id="ts-004",
        category=C.HEAVY_TAIL_HIGH_TOKEN,
        difficulty=D.HARD,
        source_trace_id="680444e7d8a9679489700fb3d2958dc6",
        prompt=(
            "In the emerging-epl project, deliver a multi-file feature: scaffold new modules, "
            "wire an API route (server/routes/epl.ts), and validate via repeated shell runs."
        ),
        notes=(
            "543,910 tokens and 156 tool calls — the highest action count in the corpus "
            "(55 Bash / 27 Write / 22 Edit), ~$5.78. Re-ran `cd .../emerging-epl` 15x. "
            "v1-seed reconstruction."
        ),
    ),
    Task(
        task_id="ts-005",
        category=C.HEAVY_TAIL_HIGH_TOKEN,
        difficulty=D.HARD,
        source_trace_id="0d810f8f550f2c71206b64ba7e174a7c",
        prompt=(
            "In the omniagent/agent-framework repository, diagnose and resolve an issue that "
            "requires reasoning over a large amount of context, validated with a handful of "
            "shell commands."
        ),
        notes=(
            "486,351 tokens with only 14 tool calls (13 Bash / 1 Read), ~$2.81 — the "
            "large-context / low-action shape: the spend is in context, not actions. "
            "Re-ran `cd ...` 12x. v1-seed reconstruction."
        ),
    ),
    Task(
        task_id="ts-006",
        category=C.HEAVY_TAIL_HIGH_TOKEN,
        difficulty=D.HARD,
        source_trace_id="606dc92782af",
        prompt=(
            "Carry out a large analysis task that loads substantial context (one file read, a "
            "few shell commands) and produces a written result."
        ),
        notes=(
            "347,215 tokens across just 5 tool calls (4 Bash / 1 Read), ~$1.92 — a near-pure "
            "context session; no repository identified in the L0 profile. v1-seed reconstruction."
        ),
    ),
    Task(
        task_id="ts-007",
        category=C.HEAVY_TAIL_HIGH_TOKEN,
        difficulty=D.HARD,
        source_trace_id="c950d1d7b9d3",
        prompt=(
            "In the omniagent/agent-framework repository, run a context-heavy investigation "
            "driven almost entirely by shell commands and report the findings."
        ),
        notes=(
            "289,180 tokens across 3 Bash calls, ~$1.56. v1-seed reconstruction from the "
            "(minimal) observable tool profile."
        ),
    ),
    Task(
        task_id="ts-008",
        category=C.HEAVY_TAIL_HIGH_TOKEN,
        difficulty=D.HARD,
        source_trace_id="663a514caa9f",
        prompt=(
            "In the omniagent/agent-framework repository, run a long-lived task that schedules "
            "its own follow-ups and tracks progress across steps."
        ),
        notes=(
            "249,891 tokens (about the corpus p90) across 10 tool calls (6 Bash, 2 "
            "ScheduleWakeup), ~$1.58. v1-seed reconstruction."
        ),
    ),
    # ---- high_tool_call_volume: outsized action counts ----
    Task(
        task_id="ts-009",
        category=C.HIGH_TOOL_CALL_VOLUME,
        difficulty=D.HARD,
        source_trace_id="44e3e9921222",
        prompt=(
            "In the ail wave0-scaffold worktree, build the Claude Code ingest adapter "
            "(src/ail/ingest/adapters/claude_code.py) against a live Databricks/MLflow "
            "workspace, iterating with many short shell probes and edits."
        ),
        notes=(
            "241,872 tokens but 101 tool calls (48 Bash / 13 Read / 13 Write / 10 Edit), "
            "~$2.55 — defining trait is action volume. Edited claude_code.py 3x. "
            "v1-seed reconstruction."
        ),
    ),
    Task(
        task_id="ts-010",
        category=C.HIGH_TOOL_CALL_VOLUME,
        difficulty=D.MEDIUM,
        source_trace_id="123536feb158",
        prompt=(
            "Drive a deployed app end-to-end: probe its serving/app URL repeatedly, exercise "
            "the endpoints, and iterate on configuration until it responds correctly."
        ),
        notes=(
            "162,262 tokens, 65 tool calls (34 Bash), ~$1.64. Repeatedly curled a "
            "self-improving-agents app URL. v1-seed reconstruction."
        ),
    ),
    Task(
        task_id="ts-011",
        category=C.HIGH_TOOL_CALL_VOLUME,
        difficulty=D.MEDIUM,
        source_trace_id="37ed22abfdb1",
        prompt=(
            "Research a topic across the web (search + fetch), install and run a uv tool, and "
            "synthesize the findings into a written plan."
        ),
        notes=(
            "202,107 tokens, 61 tool calls (30 Bash / 5 WebFetch / 1 WebSearch), ~$2.07 — a "
            "web-research-heavy session. v1-seed reconstruction."
        ),
    ),
    Task(
        task_id="ts-012",
        category=C.HIGH_TOOL_CALL_VOLUME,
        difficulty=D.MEDIUM,
        source_trace_id="e10ee82da241",
        prompt=(
            "Write and iterate on a Databricks tracing-repoint script (repoint_cc_tracing.py), "
            "exercising a SQL statements API, until it runs cleanly."
        ),
        notes=(
            "162,816 tokens, 45 tool calls (31 Bash / 4 Edit), ~$1.91. Edited "
            "repoint_cc_tracing.py 3x. v1-seed reconstruction."
        ),
    ),
    Task(
        task_id="ts-013",
        category=C.HIGH_TOOL_CALL_VOLUME,
        difficulty=D.MEDIUM,
        source_trace_id="1fcf56aeaf92",
        prompt=(
            "Audit a codebase by reading widely across many files before producing a focused "
            "summary or change."
        ),
        notes=(
            "130,655 tokens, 39 tool calls of which 34 are Read with ZERO redundancy, ~$1.03 — "
            "a legitimately read-heavy, non-wasteful session. Carried as a contrast case: high "
            "tool count that the optimizer must NOT 'fix'. v1-seed reconstruction."
        ),
    ),
    # ---- repeated_target_boilerplate: re-run shell prologue / repeated file edits ----
    Task(
        task_id="ts-014",
        category=C.REPEATED_TARGET_BOILERPLATE,
        difficulty=D.HARD,
        source_trace_id="bdb3b11e597555cda869ed7ab5b123dd",
        prompt=(
            "In the emerging-epl project, iterate on the logfood sync script "
            "(scripts/sync_from_logfood.mjs) until it works, re-establishing the working "
            "directory and re-editing the same file across many turns."
        ),
        notes=(
            "390,077 tokens, 66 tool calls (30 Edit / 14 Bash / 14 Read), ~$2.73. Re-ran "
            "`cd .../emerging-epl` 13x and edited sync_from_logfood.mjs 8x — the strongest "
            "repeated-target boilerplate in the corpus. v1-seed reconstruction."
        ),
    ),
    Task(
        task_id="ts-015",
        category=C.REPEATED_TARGET_BOILERPLATE,
        difficulty=D.MEDIUM,
        source_trace_id="f9eb702f32e1f531944ecad247a4deea",
        prompt=(
            "In the emerging-epl project, make a focused fix to the logfood sync script, "
            "re-running shell setup repeatedly between edits."
        ),
        notes=(
            "84,040 tokens, 19 tool calls (12 Bash / 7 Edit), ~$1.14. Re-ran `cd ...` 12x; "
            "edited sync_from_logfood.mjs 3x. v1-seed reconstruction."
        ),
    ),
    Task(
        task_id="ts-016",
        category=C.REPEATED_TARGET_BOILERPLATE,
        difficulty=D.MEDIUM,
        source_trace_id="acb3925baed7d48deb0ce7441f8cb0de",
        prompt=(
            "In the emerging-epl project, evolve the database schema (server/db/schema.ts) "
            "across several edits, re-establishing the working directory each time."
        ),
        notes=(
            "93,453 tokens, 24 tool calls (12 Bash / 6 Edit), ~$1.31. Re-ran `cd ...` 8x; "
            "edited schema.ts 6x. v1-seed reconstruction."
        ),
    ),
    Task(
        task_id="ts-017",
        category=C.REPEATED_TARGET_BOILERPLATE,
        difficulty=D.EASY,
        source_trace_id="bb4362c1c8c7",
        prompt=(
            "In the omniagent/agent-framework repository, adjust the Databricks deploy script "
            "(deploy/databricks/deploy.py) and re-run the deploy via uv until it succeeds."
        ),
        notes=(
            "32,763 tokens, 18 tool calls (12 Bash / 2 Edit / 2 Read), ~$0.48. Repeated "
            "`cd ... && uv run ... deploy`; edited deploy.py 2x. v1-seed reconstruction."
        ),
    ),
    Task(
        task_id="ts-018",
        category=C.REPEATED_TARGET_BOILERPLATE,
        difficulty=D.EASY,
        source_trace_id="66ddf223a8b0",
        prompt=(
            "Draft and revise a deployment guide (markdown), editing the same document across "
            "a couple of passes with light shell validation."
        ),
        notes=(
            "19,149 tokens (the corpus median), 11 tool calls (4 TaskUpdate / 3 Edit / 3 Bash), "
            "~$0.28. Edited omniagents-self-host-deployment.md 2x. v1-seed reconstruction."
        ),
    ),
    # ---- typical_short_session: the low-median bulk (do-not-regress coverage) ----
    Task(
        task_id="ts-019",
        category=C.TYPICAL_SHORT_SESSION,
        difficulty=D.EASY,
        source_trace_id="6ad7ed98e4bc",
        prompt=(
            "Answer a focused question and write out a short result file or two — a typical "
            "small session."
        ),
        notes=(
            "18,508 tokens (about the corpus median), 3 tool calls (1 Bash / 2 Write), ~$0.21. "
            "Representative of the low-median bulk. v1-seed reconstruction."
        ),
    ),
    Task(
        task_id="ts-020",
        category=C.TYPICAL_SHORT_SESSION,
        difficulty=D.EASY,
        source_trace_id="931c6c12fadc",
        prompt=(
            "Check the status of an in-flight task and decide next steps with a few light "
            "tool calls."
        ),
        notes=(
            "16,435 tokens, 7 tool calls (5 Bash), ~$0.22 — a typical short coordination "
            "session. v1-seed reconstruction."
        ),
    ),
    Task(
        task_id="ts-021",
        category=C.TYPICAL_SHORT_SESSION,
        difficulty=D.EASY,
        source_trace_id="72155b4ab5a3",
        prompt=(
            "In the omniagent/agent-framework repository, run a quick two-command check and "
            "report the result."
        ),
        notes=(
            "12,013 tokens, 2 tool calls, ~$0.17 — a minimal typical session. "
            "v1-seed reconstruction."
        ),
    ),
    Task(
        task_id="ts-022",
        category=C.TYPICAL_SHORT_SESSION,
        difficulty=D.EASY,
        source_trace_id="eb44fdd947bd",
        prompt=(
            "Answer a question directly from context with no tool use — the pure "
            "conversational case the suite must not regress."
        ),
        notes=(
            "19,355 tokens and 0 tool calls — a tool-free Q&A session. Carried so the "
            "benchmark covers the no-action case. v1-seed reconstruction."
        ),
    ),
)


def build_seed_suite() -> TaskSuite:
    """Construct the unfrozen ``v1-seed`` suite from the curated tasks.

    Call :meth:`~ail.task_suite.schema.TaskSuite.freeze` to seal it; that frozen
    form is what is serialized to ``eval/task_suite/v1/tasks.yaml``.
    """
    return TaskSuite(
        version=SEED_VERSION,
        created_at=SEED_CREATED_AT,
        tasks=SEED_TASKS,
    )
