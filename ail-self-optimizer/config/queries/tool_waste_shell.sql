-- Tool-waste: re-run shell-setup boilerplate (same normalized prologue repeated
-- within a trace). Precomputed in Tier A; read-only here.
SELECT
  substr(trace_id, -12) AS trace,
  tool,
  repeat_count,
  trace_total_tool_calls AS trace_tools,
  substr(identity, 1, 70) AS prologue
FROM austin_choi_omni_agent_catalog.agent_improvement_loop.l0_diagnosis
WHERE experiment_id = '660599403165942'
  AND signature_kind = 'shell'
ORDER BY repeat_count DESC
LIMIT 20
