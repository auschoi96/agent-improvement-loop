-- Per-version L0 aggregate for one agent (the version list / per-version numbers).
-- All metrics precomputed per (agent, version) in Tier A Python; read-only here.
-- @param agent_name STRING
-- @param experiment_id STRING
SELECT
  agent_version,
  n_traces,
  n_traces_total,
  total_tokens,
  tokens_per_trace,
  total_tool_calls,
  redundancy_rate,
  total_cost_usd,
  cost_priced,
  basis,
  source,
  model_type,
  agent_kind,
  logged_model_id,
  config_name,
  config_fingerprint,
  config_json,
  git_commit
FROM austin_choi_omni_agent_catalog.agent_improvement_loop.agent_version_l0
WHERE agent_name = :agent_name
  AND experiment_id = :experiment_id
ORDER BY agent_version
