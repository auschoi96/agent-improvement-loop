#!/usr/bin/env python3
"""Run the local companion planner (evidence-first, no proving) — a thin CLI wrapper.

All logic lives in (and is unit-tested via) :mod:`ail.jobs.companion_planner`; this
script is the ``scripts/`` counterpart of the ``ail-companion-planner`` console entry,
mirroring the other ``scripts/run_*.py`` wrappers. It reads an agent's judge/RLM/L0
evidence, plans (Lane A + Lane B), gates on readiness + judge trust, and publishes
PENDING evidence-backed proposals to ``agent_proposed_actions``. It proves nothing and
applies nothing — proving is opt-in Tier-2, run later on the user's frozen suite.

Auth is a **static** ``DATABRICKS_TOKEN`` pinned to ``DATABRICKS_HOST`` (the workspace
the experiment lives in) — a ``--profile`` OAuth login is refused, because its mid-run
token refresh cannot persist from a long-running local process.

Example
-------
    export DATABRICKS_HOST=https://<workspace-host>
    export DATABRICKS_TOKEN=<pat-or-static-token>
    python scripts/run_companion_planner.py \
        --agent claude_code \
        --experiment 660599403165942 \
        --warehouse-id <sql-warehouse-id> \
        --objective-metric total_tokens --goal-target -0.30 \
        --guardrail-judge modularity:4.0 \
        --goal-confirmed true \
        --dry-run
"""

from __future__ import annotations

from ail.jobs.companion_planner import main

if __name__ == "__main__":
    raise SystemExit(main())
