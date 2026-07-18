"""Writer-owned Unity Catalog schema for recommendation decision memory.

The recommendation planner uses normalized, queryable state rather than asking a
model to replace an opaque JSON document.  This module is the single DDL source for
the evidence ledger, immutable cohort snapshots, pattern event/state tables, action
lineage, and measured outcomes.  The shared bootstrap creates and additively
migrates these framework tables.
"""

from __future__ import annotations

INGESTION_WATERMARK_TABLE = "agent_recommendation_ingestion_watermarks"
EVIDENCE_TABLE = "agent_recommendation_evidence"
COHORT_TABLE = "agent_recommendation_cohorts"
PATTERN_TABLE = "agent_recommendation_patterns"
PATTERN_EVENT_TABLE = "agent_recommendation_pattern_events"
ACTION_TABLE = "agent_recommendation_actions"
ACTION_PATTERN_TABLE = "agent_recommendation_action_patterns"
OUTCOME_TABLE = "agent_recommendation_outcomes"

RECOMMENDATION_TABLES: frozenset[str] = frozenset(
    {
        INGESTION_WATERMARK_TABLE,
        EVIDENCE_TABLE,
        COHORT_TABLE,
        PATTERN_TABLE,
        PATTERN_EVENT_TABLE,
        ACTION_TABLE,
        ACTION_PATTERN_TABLE,
        OUTCOME_TABLE,
    }
)


def _ddl(catalog: str, schema: str) -> list[str]:
    """Idempotent CREATE statements for the recommendation decision-memory store."""
    fqn = f"`{catalog}`.`{schema}`"
    return [
        f"CREATE SCHEMA IF NOT EXISTS {fqn} "
        "COMMENT 'Agent self-optimization loop: L0 deterministic metrics (Tier A).'",
        f"""CREATE TABLE IF NOT EXISTS {fqn}.{INGESTION_WATERMARK_TABLE} (
            agent_name STRING,
            experiment_id STRING,
            last_created_at STRING,
            updated_at STRING,
            n_assessments_ingested BIGINT
        ) USING DELTA
        COMMENT 'Recommendation evidence-ingestion cursor, independent of cohort planning.'""",
        f"""CREATE TABLE IF NOT EXISTS {fqn}.{EVIDENCE_TABLE} (
            agent_name STRING,
            experiment_id STRING,
            evidence_id STRING,
            trace_id STRING,
            cohort_id STRING,
            assessment_name STRING,
            source_signal STRING,
            value STRING,
            comment STRING,
            metadata_json STRING,
            assessment_created_at STRING,
            subject_or_reviewer STRING,
            reserved_pool STRING,
            ingested_at STRING
        ) USING DELTA
        COMMENT 'Grounded RLM and judge evidence ledger used by cohort recommendation planning.'""",
        f"""CREATE TABLE IF NOT EXISTS {fqn}.{COHORT_TABLE} (
            agent_name STRING,
            experiment_id STRING,
            cohort_id STRING,
            cohort_sequence BIGINT,
            status STRING,
            min_traces INT,
            trace_count INT,
            assessment_count INT,
            trace_ids ARRAY<STRING>,
            evidence_ids ARRAY<STRING>,
            evidence_cutoff_at STRING,
            queue_snapshot_at STRING,
            planner_model STRING,
            planner_prompt_version STRING,
            planner_run_id STRING,
            created_at STRING,
            started_at STRING,
            completed_at STRING,
            error STRING
        ) USING DELTA
        COMMENT 'Deterministic recommendation cohorts and their retryable planning lifecycle.'""",
        f"""CREATE TABLE IF NOT EXISTS {fqn}.{PATTERN_TABLE} (
            agent_name STRING,
            experiment_id STRING,
            pattern_id STRING,
            canonical_key STRING,
            category STRING,
            title STRING,
            root_cause STRING,
            status STRING,
            first_seen_cohort_id STRING,
            last_seen_cohort_id STRING,
            cohort_count BIGINT,
            distinct_trace_count BIGINT,
            recent_trace_count BIGINT,
            recent_prevalence DOUBLE,
            severity DOUBLE,
            confidence DOUBLE,
            trend_score DOUBLE,
            trend_label STRING,
            current_action_id STRING,
            summary_embedding ARRAY<FLOAT>,
            created_at STRING,
            updated_at STRING
        ) USING DELTA
        COMMENT 'Materialized current state of stable cross-trace recommendation patterns.'""",
        f"""CREATE TABLE IF NOT EXISTS {fqn}.{PATTERN_EVENT_TABLE} (
            agent_name STRING,
            experiment_id STRING,
            event_id STRING,
            pattern_id STRING,
            cohort_id STRING,
            event_type STRING,
            evidence_ids ARRAY<STRING>,
            source_trace_ids ARRAY<STRING>,
            observation_summary STRING,
            severity DOUBLE,
            confidence DOUBLE,
            created_at STRING
        ) USING DELTA
        COMMENT 'Append-only grounded event history for recommendation pattern evolution.'""",
        f"""CREATE TABLE IF NOT EXISTS {fqn}.{ACTION_TABLE} (
            agent_name STRING,
            experiment_id STRING,
            action_id STRING,
            canonical_action_key STRING,
            category STRING,
            title STRING,
            plan STRING,
            status STRING,
            proposal_id STRING,
            first_proposed_cohort_id STRING,
            last_supported_cohort_id STRING,
            supersedes_action_id STRING,
            merged_into_action_id STRING,
            human_decided_at STRING,
            applied_at STRING,
            created_at STRING,
            updated_at STRING
        ) USING DELTA
        COMMENT 'Queue-aware recommendation action lineage and synchronized human status.'""",
        f"""CREATE TABLE IF NOT EXISTS {fqn}.{ACTION_PATTERN_TABLE} (
            agent_name STRING,
            experiment_id STRING,
            link_id STRING,
            action_id STRING,
            pattern_id STRING,
            relation STRING,
            first_linked_cohort_id STRING,
            last_linked_cohort_id STRING,
            created_at STRING,
            updated_at STRING
        ) USING DELTA
        COMMENT 'Many-to-many lineage between broad recommendation actions and patterns.'""",
        f"""CREATE TABLE IF NOT EXISTS {fqn}.{OUTCOME_TABLE} (
            agent_name STRING,
            experiment_id STRING,
            outcome_id STRING,
            action_id STRING,
            proposal_id STRING,
            observed_at STRING,
            source STRING,
            metric_name STRING,
            baseline_value DOUBLE,
            candidate_value DOUBLE,
            delta DOUBLE,
            result STRING,
            n_traces BIGINT,
            window_start STRING,
            window_end STRING,
            details_json STRING
        ) USING DELTA
        COMMENT 'Append-only verification, human-decision, and organic outcome memory.'""",
    ]
