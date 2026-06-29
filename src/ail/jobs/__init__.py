"""Deployment glue for running :mod:`ail` steps as Databricks Jobs.

This package holds the thin, Databricks-runtime-specific entrypoints that adapt
the reusable library (``ail.publish`` / ``ail.metrics``) to a scheduled Job. The
library modules stay free of any job/secret/runtime coupling; everything that is
specific to *running on Databricks serverless as a scheduled job* lives here.
"""

from __future__ import annotations
