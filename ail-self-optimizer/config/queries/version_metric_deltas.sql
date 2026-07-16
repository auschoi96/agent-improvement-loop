-- Per-metric version-over-version deltas for one baseline->candidate comparison.
-- Each row is a MetricDelta computed in Tier A Python (ail.compare semantics):
-- the app reads baseline/candidate/delta and renders direction; it does NOT
-- recompute any metric or delta in SQL. Read-only.
-- @param agent_name STRING
-- @param experiment_id STRING
-- @param baseline_version STRING
-- @param candidate_version STRING
SELECT
  metric,
  unit,
  metric_tier,
  lower_is_better,
  baseline_value,
  candidate_value,
  delta_absolute,
  delta_pct,
  improved
FROM austin_choi_omni_agent_catalog.agent_improvement_loop.agent_version_comparison
WHERE agent_name = :agent_name
  AND experiment_id = :experiment_id
  AND baseline_version = :baseline_version
  AND candidate_version = :candidate_version
ORDER BY CASE metric
  WHEN 'total_tokens' THEN 0
  WHEN 'tokens_per_trace' THEN 1
  WHEN 'total_tool_calls' THEN 2
  WHEN 'redundancy_rate' THEN 3
  WHEN 'total_usd' THEN 4
  ELSE 9
END
