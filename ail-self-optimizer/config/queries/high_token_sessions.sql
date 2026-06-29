-- Token heavy-tail: the highest-token sessions. Read-only from precomputed L0.
SELECT
  substr(trace_id, -12) AS trace,
  model,
  total_tokens,
  input_tokens,
  output_tokens,
  total_tool_calls AS tools,
  ROUND(duration_seconds, 1) AS duration_s,
  ROUND(est_cost_usd, 2) AS est_cost_usd,
  cost_priced,
  ROUND(redundancy_rate, 3) AS redundancy_rate
FROM austin_choi_omni_agent_catalog.agent_improvement_loop.l0_session_metrics
WHERE experiment_id = '660599403165942'
ORDER BY total_tokens DESC
LIMIT 25
