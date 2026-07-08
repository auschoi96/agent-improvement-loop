"""Databricks Job entrypoint for the scheduled advisory-memory distiller.

Referenced by ``resources/memory_distiller.job.yml`` as the serverless
``spark_python_task`` ``python_file`` (and exposed as the ``ail-memory-distiller``
wheel console script). All logic lives in :mod:`ail.memory.distiller`; this is the
thin launcher, matching the other ``ail.jobs.*`` entrypoints.
"""

from __future__ import annotations

from ail.memory.distiller import main

if __name__ == "__main__":
    raise SystemExit(main())
