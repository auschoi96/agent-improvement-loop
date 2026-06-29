-- Per-model rollup over precomputed per-session facts (plain aggregation; the
-- pricing/redundancy logic itself lives in Tier A Python, not here).
SELECT
  COALESCE(model, '(unknown)') AS model,
  COUNT(*) AS sessions,
  SUM(total_tokens) AS total_tokens,
  ROUND(SUM(est_cost_usd), 2) AS est_cost_usd,
  SUM(total_tool_calls) AS tool_calls
FROM austin_choi_omni_agent_catalog.agent_improvement_loop.l0_session_metrics
WHERE experiment_id = '660599403165942'
GROUP BY COALESCE(model, '(unknown)')
ORDER BY total_tokens DESC
