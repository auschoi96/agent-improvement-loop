"""agent-improvement-loop (``ail``).

A reusable, agent-agnostic self-improvement loop for LLM agents. See
``docs/ARCHITECTURE.md`` for the full design.
"""

__version__ = "0.0.0"

from ail.sdk import ImprovementAgent, RunResult, improve, trace

__all__ = ["ImprovementAgent", "RunResult", "__version__", "improve", "trace"]
