-- Baseline-vs-candidate comparison header(s) for one agent: the trust-gated
-- display status, the real readiness tier, the controlled-proof header, and the
-- headline objective delta. The `status` is decided in Python (Tier A), never
-- here — the app renders it, it does not compute trust. Read-only.
-- @param agent_name STRING
SELECT
  baseline_version,
  candidate_version,
  objective_metric,
  status,
  readiness_tier,
  can_prove_improvement,
  trace_count,
  frozen_suite_present,
  n_promote,
  n_block,
  n_errored,
  correctness_held,
  proof_source,
  headline_metric,
  headline_baseline,
  headline_candidate,
  headline_delta_pct,
  headline_improved,
  reasons
FROM austin_choi_omni_agent_catalog.agent_improvement_loop.agent_version_readiness
WHERE agent_name = :agent_name
ORDER BY candidate_version
