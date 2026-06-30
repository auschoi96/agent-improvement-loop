-- Top sessions by total tokens, for the heavy-tail bar chart (small payload).
-- @param experiment_id STRING
SELECT
  substr(trace_id, -8) AS trace,
  total_tokens
FROM austin_choi_omni_agent_catalog.agent_improvement_loop.l0_session_metrics
WHERE experiment_id = :experiment_id
ORDER BY total_tokens DESC
LIMIT 15
