#!/usr/bin/env python3
"""Run the local companion executor (L7b-2) — a thin CLI wrapper.

All logic lives in (and is unit-tested via) :mod:`ail.jobs.agent_executor`; this
script is the ``scripts/`` counterpart of the ``ail-agent-executor`` console entry,
mirroring the other ``scripts/run_*.py`` wrappers. It previews PENDING ``AGENT_TASK``
proposals in an isolated sandbox copy of the agent's ``target_workspace`` and commits
APPROVED ones to the live workspace via the L6 UC-Volume snapshot substrate — applying
the exact change the human approved, never re-running the agent at commit.

Auth is a **static** ``DATABRICKS_TOKEN`` pinned to ``DATABRICKS_HOST`` — a ``--profile``
OAuth login is refused, because its mid-run token refresh cannot persist from a
long-running local process.

Example
-------
    export DATABRICKS_HOST=https://<workspace-host>
    export DATABRICKS_TOKEN=<pat-or-static-token>
    python scripts/run_agent_executor.py \
        --agent claude_code \
        --registry config/agents.yaml \
        --warehouse-id <sql-warehouse-id> \
        --volume-root /Volumes/<catalog>/<schema>/<volume>/ail_snapshots \
        --dry-run
"""

from __future__ import annotations

from ail.jobs.agent_executor import main

if __name__ == "__main__":
    raise SystemExit(main())
