"""Concrete :class:`~ail.ingest.base.AgentAdapter` implementations.

One module per agent runtime. Both Claude Code and Codex ship today. Codex does
not autolog to MLflow the way Claude Code does — MLflow's native Codex
integration is the Node ``@mlflow/codex`` notify hook — so the Codex adapter
also owns a Python trace-capture path that normalizes the rollout transcript.
"""

from ail.ingest.adapters.claude_code import ClaudeCodeAdapter
from ail.ingest.adapters.codex import (
    CODEX_AGENT,
    CodexAdapter,
    enable_codex_tracing,
    normalize_codex_rollout,
)

__all__ = [
    "ClaudeCodeAdapter",
    "CodexAdapter",
    "CODEX_AGENT",
    "enable_codex_tracing",
    "normalize_codex_rollout",
]
