"""Concrete :class:`~ail.ingest.base.AgentAdapter` implementations.

One module per agent runtime. Claude Code ships today; Codex is next (it does
not autolog to MLflow the way Claude Code does, so its adapter also owns a
trace-capture path).
"""

from ail.ingest.adapters.claude_code import ClaudeCodeAdapter

__all__ = ["ClaudeCodeAdapter"]
