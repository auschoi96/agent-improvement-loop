-- Per-producer rollup over precomputed per-session facts (plain aggregation).
-- @param experiment_id STRING
SELECT
  COALESCE(producer, '(undetected)') AS producer,
  COUNT(*) AS sessions,
  SUM(total_tokens) AS total_tokens,
  ROUND(SUM(est_cost_usd), 2) AS est_cost_usd,
  SUM(total_tool_calls) AS tool_calls
FROM austin_choi_omni_agent_catalog.agent_improvement_loop.l0_session_metrics
WHERE experiment_id = :experiment_id
GROUP BY COALESCE(producer, '(undetected)')
ORDER BY total_tokens DESC
