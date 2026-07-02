#!/usr/bin/env python3
"""Thin CLI wrapper for the LOCAL optimization cycle (the real prover runs here).

All logic lives in (and is unit-tested in) :mod:`ail.jobs.local_cycle`; this is the
``scripts/`` entry point mirroring the other thin wrappers in this directory. Prefer
the installed console command ``ail-local-cycle`` (see ``pyproject.toml``); this file
is the run-from-a-checkout equivalent.

    python scripts/run_local_cycle.py \
        --experiment 660599403165942 \
        --warehouse-id <sql-warehouse-id> \
        --judge-model databricks-claude-sonnet-4-6 \
        --confirm-goal

Requires a STATIC ``DATABRICKS_HOST`` + ``DATABRICKS_TOKEN`` (matched to the
experiment's workspace) and local Claude auth for the prover. See
``docs/LOCAL_RUNNER.md`` for the full prerequisites and what it prints.
"""

from __future__ import annotations

from ail.jobs.local_cycle import main

if __name__ == "__main__":
    raise SystemExit(main())
