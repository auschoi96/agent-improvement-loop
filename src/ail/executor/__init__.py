"""Lane L7b-2 — the open-ended Claude Agent SDK executor (local companion).

Carries out an approved ``AGENT_TASK`` proposal by running a sandboxed Claude Agent
SDK agent to make arbitrary, Databricks-native (no-git) changes to the target agent's
source. The safety is in the wrapper, split into two independently-tested halves:
:func:`~ail.executor.executor.produce_preview` (pre-approval, no live effect) and
:func:`~ail.executor.executor.commit_approved` (post-approval, applies the *stored*
previewed change — never re-runs the agent). See :mod:`ail.executor.executor` and
``docs/EXECUTOR.md``.
"""

from __future__ import annotations

from ail.executor.executor import (
    DEFAULT_TIMEOUT_SECONDS,
    EXECUTOR_SYSTEM_PROMPT,
    AgentRunner,
    CommitRecorder,
    CommitRecordError,
    CommitRefused,
    CommitResult,
    CommittedChangeRecord,
    ExecutorError,
    FileChange,
    PreviewError,
    PreviewResult,
    PreviewWriter,
    commit_approved,
    produce_preview,
)

__all__ = [
    "EXECUTOR_SYSTEM_PROMPT",
    "DEFAULT_TIMEOUT_SECONDS",
    "ExecutorError",
    "PreviewError",
    "CommitRefused",
    "CommitRecordError",
    "AgentRunner",
    "PreviewWriter",
    "CommitRecorder",
    "FileChange",
    "CommittedChangeRecord",
    "PreviewResult",
    "CommitResult",
    "produce_preview",
    "commit_approved",
]
