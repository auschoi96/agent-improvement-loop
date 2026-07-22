"""Governed cohort, pattern, action, and outcome memory for recommendations."""

from ail.recommendations.managed_memory import ManagedMemoryClient, MemoryEntry
from ail.recommendations.schema import (
    ACTION_PATTERN_TABLE,
    ACTION_TABLE,
    COHORT_TABLE,
    EVIDENCE_TABLE,
    INGESTION_WATERMARK_TABLE,
    OUTCOME_TABLE,
    PATTERN_EVENT_TABLE,
    PATTERN_TABLE,
    RECOMMENDATION_TABLES,
)

__all__ = [
    "ACTION_PATTERN_TABLE",
    "ACTION_TABLE",
    "COHORT_TABLE",
    "EVIDENCE_TABLE",
    "INGESTION_WATERMARK_TABLE",
    "ManagedMemoryClient",
    "MemoryEntry",
    "OUTCOME_TABLE",
    "PATTERN_EVENT_TABLE",
    "PATTERN_TABLE",
    "RECOMMENDATION_TABLES",
]
