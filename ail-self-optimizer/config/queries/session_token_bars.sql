-- Top sessions by total tokens, for the heavy-tail bar chart (small payload).
SELECT
  substr(trace_id, -8) AS trace,
  total_tokens
FROM austin_choi_omni_agent_catalog.agent_improvement_loop.l0_session_metrics
WHERE experiment_id = '660599403165942'
ORDER BY total_tokens DESC
LIMIT 15
