-- Tool-waste: repeated file access (same path read/edited repeatedly within a
-- trace). Precomputed in Tier A; read-only here.
-- @param experiment_id STRING
SELECT
  substr(trace_id, -12) AS trace,
  tool,
  repeat_count,
  trace_total_tool_calls AS trace_tools,
  substr(identity, 1, 70) AS path
FROM austin_choi_omni_agent_catalog.agent_improvement_loop.l0_diagnosis
WHERE experiment_id = :experiment_id
  AND signature_kind = 'path'
ORDER BY repeat_count DESC
LIMIT 20
