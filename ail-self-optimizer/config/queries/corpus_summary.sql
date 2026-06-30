-- Corpus-wide L0 KPIs (one row). Read-only; all metrics precomputed in Tier A.
-- @param experiment_id STRING
SELECT
  trace_count,
  total_tokens,
  total_input_tokens,
  total_output_tokens,
  median_tokens,
  mean_tokens,
  p90_tokens,
  max_tokens,
  min_tokens,
  total_tool_calls,
  redundancy_rate,
  total_cost_usd,
  priced_traces,
  unpriced_traces
FROM austin_choi_omni_agent_catalog.agent_improvement_loop.l0_corpus_summary
WHERE experiment_id = :experiment_id
