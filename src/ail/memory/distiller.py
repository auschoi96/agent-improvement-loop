"""The advisory-memory distiller — the driver a scheduled Databricks Job runs.

One firing:

1. resolves auth for the run-as identity (reusing :func:`ail.jobs.publish_job.resolve_job_auth`);
2. reads the idempotency **watermark** for this ``experiment:cohort`` scope;
3. reads the RLM + L2-judge **assessments** created since the watermark
   (:func:`ail.memory.assessments.read_assessments`) — **fail-closed**: an empty
   window writes nothing;
4. resolves the frozen **reserved pools** (:func:`ail.memory.provenance.resolve_reserved_pools`)
   — fail-closed: if the Task Suite can't load, nothing is written;
5. drives the **Claude Agent SDK** ``query()`` loop, exposing the ``submit_memory``
   tool (:func:`ail.memory.writeback.create_submit_memory_tool`) which validates,
   grounds, walls, and MERGEs (idempotent upsert); and
6. advances the watermark to the newest assessment it processed, so a re-run over
   the same window is a no-op.

Zero-secret FMAPI auth, mirrored from the reference agent: the workspace OAuth
bearer is set as ``ANTHROPIC_API_KEY``/``ANTHROPIC_AUTH_TOKEN`` with
``ANTHROPIC_BASE_URL`` pointing at the workspace's ``/serving-endpoints/anthropic``
— no external Anthropic key, no secret scope. The token is minted once at startup
(no mid-run refresh), so the distill run is bounded (``--max-turns`` and a cap on
assessments per firing) to stay well under the token lifetime; a window large
enough to risk that should be split across firings.

Every seam the driver needs is injectable (``client``, ``reserved``, ``distill``,
``now``) so the whole control flow — fail-closed, watermark idempotency — is tested
without a live model or workspace.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ail.memory.assessments import AssessmentRow, max_created_at, read_assessments
from ail.memory.provenance import ReservedPools, resolve_reserved_pools
from ail.memory.schema import MEMORY_TABLE
from ail.memory.watermark import read_watermark, watermark_scope, write_watermark
from ail.memory.writeback import WriteTally, create_submit_memory_tool

DEFAULT_MODEL = "databricks-claude-opus-4-6"
DEFAULT_COHORT = "claude_code"


def _utc_now_iso() -> str:
    """UTC now as ``YYYY-MM-DDTHH:MM:SS.mmmZ`` (matches the other tables' STRING stamps)."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


@dataclass(frozen=True, slots=True)
class DistillerConfig:
    """Everything one distiller firing needs; all identifiers come from bundle vars."""

    experiment_id: str
    warehouse_id: str
    catalog: str
    schema: str
    annotations_table: str
    cohort: str = DEFAULT_COHORT
    model: str = DEFAULT_MODEL
    max_turns: int = 30
    max_assessments: int = 200
    task_suite_version: str = "v1"
    task_suite_root: str | None = None
    groundtruth_root: str | None = None
    token_secret_scope: str = ""
    token_secret_key: str = ""


@dataclass(slots=True)
class DistillerReport:
    """The outcome of one firing (also the log line)."""

    scope: str
    wrote: bool
    n_assessments: int = 0
    n_written: int = 0
    n_dropped_provenance: int = 0
    n_invalid: int = 0
    watermark_before: str | None = None
    watermark_after: str | None = None
    note: str = ""

    def __str__(self) -> str:
        return (
            f"[ail.memory.distiller] scope={self.scope} wrote={self.wrote} "
            f"assessments={self.n_assessments} written={self.n_written} "
            f"dropped_provenance={self.n_dropped_provenance} invalid={self.n_invalid} "
            f"watermark {self.watermark_before!r}->{self.watermark_after!r} {self.note}".strip()
        )


@dataclass(slots=True)
class DistillerDeps:
    """Injectable seams (real defaults built lazily). Tests override these."""

    client: Any | None = None
    reserved: ReservedPools | None = None
    distill: Callable[[list[AssessmentRow], WriteTally], None] | None = None
    now: Callable[[], str] = field(default=_utc_now_iso)


# ---------------------------------------------------------------------------
# Auth + client (reuses the publish job's resolver + workspace-client builder)
# ---------------------------------------------------------------------------


def _build_client(config: DistillerConfig) -> Any:
    """Resolve run-as auth into ``DATABRICKS_HOST``/``TOKEN`` and build a WorkspaceClient.

    Reuses :func:`ail.jobs.publish_job.resolve_job_auth` (env > secret-scope > mint
    a short-lived OAuth bearer) and :func:`ail.publish._build_workspace_client`, so
    the memory job authenticates identically to every other framework job.
    """
    from ail.jobs.publish_job import resolve_job_auth
    from ail.publish import _build_workspace_client

    auth_path = resolve_job_auth(
        token_secret_scope=config.token_secret_scope or None,
        token_secret_key=config.token_secret_key or None,
    )
    print(f"[ail.memory.distiller] auth={auth_path} host={os.environ.get('DATABRICKS_HOST')}")
    return _build_workspace_client(None)


def _claude_env(model: str) -> dict[str, str]:
    """Zero-secret FMAPI env for the Claude Agent SDK (mirrors the reference agent)."""
    host = os.environ["DATABRICKS_HOST"]
    token = os.environ["DATABRICKS_TOKEN"]
    clean_host = host.replace("https://", "").replace("http://", "").rstrip("/")
    return {
        "ANTHROPIC_BASE_URL": f"https://{clean_host}/serving-endpoints/anthropic",
        "ANTHROPIC_API_KEY": token,
        "ANTHROPIC_AUTH_TOKEN": token,
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_CUSTOM_HEADERS": "x-databricks-use-coding-agent-mode: true",
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
        "CLAUDE_CODE_STREAM_CLOSE_TIMEOUT": "3600000",
    }


# ---------------------------------------------------------------------------
# The prompt + the Claude Agent SDK loop (the only live-model part)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You distill agent-evaluation feedback into a small set of short, actionable "
    "'memory guidelines' for a coding agent. You write ONLY through the submit_memory "
    "tool. Prefer few high-signal, generalizable guidelines over many narrow ones. "
    "Never invent feedback; cite only the trace ids you are given."
)


def build_distill_prompt(assessments: list[AssessmentRow]) -> str:
    """Render the assessments into the distillation prompt (grouped by trace)."""
    by_trace: dict[str, list[AssessmentRow]] = {}
    for a in assessments:
        by_trace.setdefault(a.trace_id, []).append(a)

    lines: list[str] = []
    for trace_id, items in by_trace.items():
        lines.append(f"\n### trace {trace_id}")
        for a in items:
            comment = a.comment.strip().replace("\n", " ")
            lines.append(
                f"- [{a.source_signal}] {a.name} = {a.value or 'n/a'}"
                + (f" — {comment}" if comment else "")
            )
    feedback_block = "\n".join(lines)

    return (
        "Below is recent evaluation feedback attached to agent traces — RLM/HALO "
        "reviews (source_signal 'rlm') and L2 LLM-judge assessments (source_signal "
        "'judge:<name>'). Each item is a score and a rationale.\n"
        f"{feedback_block}\n\n"
        "Distill this into a SMALL set of short, generalizable memory guidelines the "
        "agent should follow next time. For each guideline call submit_memory with: a "
        "category, one imperative guideline_text, a 0-1 confidence score, the "
        "source_trace_ids it is grounded in (from the trace ids above), and the "
        "source_signal. Group related feedback; skip low-signal or one-off noise. "
        "Submit all guidelines in as few submit_memory calls as possible, then stop."
    )


def _default_distill(
    config: DistillerConfig,
    client: Any,
    reserved: ReservedPools,
    read_trace_ids: frozenset[str],
    now: Callable[[], str],
) -> Callable[[list[AssessmentRow], WriteTally], None]:
    """Build the real agent-driven distill step (Claude Agent SDK ``query()`` loop)."""

    def distill(assessments: list[AssessmentRow], tally: WriteTally) -> None:
        import asyncio
        import shutil
        import tempfile
        from pathlib import Path

        import nest_asyncio
        from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server, query
        from claude_agent_sdk.types import AssistantMessage, ResultMessage, TextBlock, ToolUseBlock

        nest_asyncio.apply()

        submit_tool = create_submit_memory_tool(
            client=client,
            warehouse_id=config.warehouse_id,
            catalog=config.catalog,
            schema=config.schema,
            cohort=config.cohort,
            reserved=reserved,
            read_trace_ids=read_trace_ids,
            tally=tally,
            now=now,
        )
        server = create_sdk_mcp_server(name="memory-tools", tools=[submit_tool])
        tool_names = ["mcp__memory-tools__submit_memory"]
        project_dir = Path(tempfile.mkdtemp(prefix="ail_memory_"))

        options = ClaudeAgentOptions(
            cwd=str(project_dir),
            # submit_memory ONLY: the feedback is in the prompt and output goes back
            # through the tool, so the loop needs no Read/Write/TodoWrite — narrowing
            # the tool surface removes fabrication vectors (reading unrelated files).
            allowed_tools=list(tool_names),
            permission_mode="bypassPermissions",
            mcp_servers={"memory-tools": server},
            system_prompt=_SYSTEM_PROMPT,
            env=_claude_env(config.model),
            max_turns=config.max_turns,
        )

        async def _run() -> None:
            async for msg in query(prompt=build_distill_prompt(assessments), options=options):
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, ToolUseBlock):
                            print(f"[ail.memory.distiller] tool-use: {block.name}")
                        elif isinstance(block, TextBlock) and block.text:
                            print(f"[ail.memory.distiller] {block.text[:200]}")
                elif isinstance(msg, ResultMessage):
                    print("[ail.memory.distiller] agent result received")

        try:
            asyncio.run(_run())
        finally:
            shutil.rmtree(project_dir, ignore_errors=True)

    return distill


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------


def run_memory_distiller(
    config: DistillerConfig,
    *,
    deps: DistillerDeps | None = None,
) -> DistillerReport:
    """Run one firing; return its :class:`DistillerReport`.

    Fail-closed throughout: an empty assessment window returns without writing or
    advancing the watermark; a Task-Suite pool that cannot load raises before any
    write; a memory row may cite ONLY trace ids read this run (anti-fabrication); and
    the watermark is advanced ONLY if the distill step recorded no errors (a SQL
    failure or a provenance-wall regression), so a failed run re-processes the same
    window next time — duplicate-free, thanks to the deterministic id + MERGE upsert.
    """
    deps = deps or DistillerDeps()
    now = deps.now
    client = deps.client if deps.client is not None else _build_client(config)
    scope = watermark_scope(config.experiment_id, config.cohort)

    watermark_before = read_watermark(
        client, config.warehouse_id, catalog=config.catalog, schema=config.schema, scope=scope
    )
    assessments = read_assessments(
        client,
        config.warehouse_id,
        annotations_table=config.annotations_table,
        since_created_at=watermark_before,
        max_results=config.max_assessments,
    )
    if not assessments:
        report = DistillerReport(
            scope=scope,
            wrote=False,
            n_assessments=0,
            watermark_before=watermark_before,
            watermark_after=watermark_before,
            note="no new assessments since watermark — nothing to distill",
        )
        print(report)
        return report

    # Fail-closed: the wall must be resolvable BEFORE we distill/write.
    reserved = (
        deps.reserved
        if deps.reserved is not None
        else resolve_reserved_pools(
            task_suite_version=config.task_suite_version,
            task_suite_root=config.task_suite_root,
            groundtruth_root=config.groundtruth_root,
        )
    )

    # The anti-fabrication grounding set: a memory row may cite ONLY these trace ids
    # (the ones whose feedback we actually read this run).
    read_trace_ids = frozenset(a.trace_id for a in assessments)

    tally = WriteTally()
    distill = deps.distill or _default_distill(config, client, reserved, read_trace_ids, now)
    distill(assessments, tally)

    # Fail-closed: if any submit_memory call failed (a SQL/MERGE error, or a
    # provenance-wall regression), do NOT advance the watermark — surface it loudly so
    # the window is retried next run. The deterministic memory_id + MERGE upsert make
    # that retry duplicate-free.
    if tally.errors:
        raise RuntimeError(
            f"memory distiller: {len(tally.errors)} submit_memory failure(s), "
            f"watermark NOT advanced: {'; '.join(tally.errors)}"
        )

    # Advance the watermark to the newest assessment PROCESSED (whether it produced
    # a memory row or was dropped by the wall) so this window is never re-distilled.
    watermark_after = max_created_at(assessments) or watermark_before or now()
    write_watermark(
        client,
        config.warehouse_id,
        catalog=config.catalog,
        schema=config.schema,
        scope=scope,
        last_created_at=watermark_after,
        run_at=now(),
        n_assessments_seen=len(assessments),
        n_memories_written=tally.written,
        n_dropped_provenance=len(tally.dropped_provenance),
    )

    report = DistillerReport(
        scope=scope,
        wrote=tally.written > 0,
        n_assessments=len(assessments),
        n_written=tally.written,
        n_dropped_provenance=len(tally.dropped_provenance),
        n_invalid=len(tally.invalid),
        watermark_before=watermark_before,
        watermark_after=watermark_after,
    )
    print(report)
    for dropped in tally.dropped_provenance:
        print(f"[ail.memory.distiller] DROPPED (provenance): {dropped.reason}")
    for _candidate, reason in tally.invalid:
        print(f"[ail.memory.distiller] REJECTED (invalid): {reason}")
    return report


# ---------------------------------------------------------------------------
# CLI / Job entrypoint
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> DistillerConfig:
    parser = argparse.ArgumentParser(
        description="Distill recent RLM + judge feedback into governed advisory-memory rows."
    )
    parser.add_argument("--experiment-id", default=os.environ.get("AIL_EXPERIMENT_ID", ""))
    parser.add_argument("--warehouse-id", default=os.environ.get("AIL_WAREHOUSE_ID", ""))
    parser.add_argument("--catalog", default=os.environ.get("AIL_CATALOG", ""))
    parser.add_argument("--schema", default=os.environ.get("AIL_SCHEMA", ""))
    parser.add_argument(
        "--annotations-table",
        default=os.environ.get("AIL_MEMORY_ANNOTATIONS_TABLE", ""),
        help="Fully-qualified OTEL annotations table, e.g. "
        "austin_choi_omni_agent_catalog.mlflow_traces.cc_otel_annotations",
    )
    parser.add_argument("--cohort", default=DEFAULT_COHORT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--max-assessments", type=int, default=200)
    parser.add_argument("--task-suite-version", default="v1")
    parser.add_argument("--task-suite-root", default=os.environ.get("AIL_TASK_SUITE_ROOT") or None)
    parser.add_argument(
        "--groundtruth-root", default=os.environ.get("AIL_GROUNDTRUTH_ROOT") or None
    )
    parser.add_argument(
        "--token-secret-scope", default=os.environ.get("AIL_TOKEN_SECRET_SCOPE", "")
    )
    parser.add_argument("--token-secret-key", default=os.environ.get("AIL_TOKEN_SECRET_KEY", ""))
    args = parser.parse_args(argv)

    missing = [
        name
        for name, value in (
            ("--experiment-id", args.experiment_id),
            ("--warehouse-id", args.warehouse_id),
            ("--catalog", args.catalog),
            ("--schema", args.schema),
            ("--annotations-table", args.annotations_table),
        )
        if not value
    ]
    if missing:
        # Fail-closed on the workspace-safety vars: no defaults are baked in, so a
        # deploy that forgot to set them errors here rather than reading/writing the
        # wrong workspace (the #67/#5 pattern).
        parser.error(f"missing required arg(s): {', '.join(missing)}")

    return DistillerConfig(
        experiment_id=args.experiment_id,
        warehouse_id=args.warehouse_id,
        catalog=args.catalog,
        schema=args.schema,
        annotations_table=args.annotations_table,
        cohort=args.cohort,
        model=args.model,
        max_turns=args.max_turns,
        max_assessments=args.max_assessments,
        task_suite_version=args.task_suite_version,
        task_suite_root=args.task_suite_root,
        groundtruth_root=args.groundtruth_root,
        token_secret_scope=args.token_secret_scope,
        token_secret_key=args.token_secret_key,
    )


def main(argv: list[str] | None = None) -> int:
    config = _parse_args(argv)
    print(
        f"[ail.memory.distiller] experiment={config.experiment_id} model={config.model} "
        f"-> {config.catalog}.{config.schema}.{MEMORY_TABLE} "
        f"(annotations={config.annotations_table})"
    )
    run_memory_distiller(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
