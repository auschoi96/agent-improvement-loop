-- Human-gated proposed actions for the in-app approval queue (Phase C lane 3b).
-- Read-only, two-tier: SELECT from the controller-published agent_proposed_actions
-- table (docs/LOOP_CONTROLLER.md). Pending first — those need a decision — with any
-- recently-decided rows following as read-only history. Each row carries the WHY
-- (the trigger signal), the WHAT (the concrete change under review), the PROOF (the
-- frozen-suite objective delta with correctness held + promote/block counts) and the
-- GATE status (readiness tier + judge agreement + scored coverage), so the reviewer
-- approves on evidence, not a bare request. The authenticated approve/reject WRITE is
-- the app's only write-path (a custom AppKit server route → ail.loop.apply_service);
-- this query stays SELECT-only.
-- @param agent_name STRING
-- @param experiment_id STRING
SELECT
  proposal_id,
  agent_name,
  experiment_id,
  status,
  action_kind,
  risk_class,
  objective_metric,
  created_at,
  -- why (the trigger signal that fired)
  trigger_kind,
  trigger_summary,
  trigger_metric,
  trigger_observed_value,
  trigger_threshold,
  trigger_n_traces,
  trigger_judge_name,
  -- what (the concrete change the human is approving)
  change_kind,
  change_summary,
  change_sql,
  change_diff,
  change_evolved_body_ref,
  change_revert_target,
  -- proof (frozen-suite WITH/WITHOUT, correctness held). For an evidence-first
  -- proposal these are NULL until a reviewer runs the opt-in Tier-2 "verify on my
  -- suite" (below), whose result the companion poll handler writes back into them.
  proof_proved_improvement,
  proof_correctness_held,
  proof_realized_savings_pct,
  proof_n_promote,
  proof_n_block,
  proof_n_errored,
  proof_suite_version,
  -- gate (readiness wall + certifying-judge trust + coverage)
  gate_readiness_tier,
  gate_gated,
  gate_judge_agreement,
  gate_scored_coverage,
  gate_n_distrusted_judges,
  -- verify (opt-in Tier-2 "verify on my suite" lifecycle, L9). Read-only here — the
  -- request WRITE is the authenticated /api/approvals/verify route, and the RESULT is
  -- written by the companion poll handler into proof_* above + verify_status below.
  verify_requested,
  verify_status,
  verify_requested_by,
  verify_requested_at,
  verify_completed_at,
  verify_error
FROM austin_choi_omni_agent_catalog.agent_improvement_loop.agent_proposed_actions
WHERE agent_name = :agent_name
  AND experiment_id = :experiment_id
ORDER BY CASE WHEN status = 'pending' THEN 0 ELSE 1 END, created_at DESC
