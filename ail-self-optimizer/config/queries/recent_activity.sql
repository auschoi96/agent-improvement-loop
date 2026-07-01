-- Recent proposal/decision outcomes for the Activity page — the observability
-- complement to the approval queue's pending-only actionable view. Read-only,
-- two-tier: SELECT from the controller-published agent_proposed_actions table
-- (docs/LOOP_CONTROLLER.md), across ALL agents, most-recent first. Each row is a
-- proposal and its CURRENT lifecycle outcome (status: pending/approved/rejected/
-- applied/superseded — advanced in place by lane 3b) plus the WHY (trigger summary)
-- and the timestamps the table records. No metric is recomputed here and no state is
-- reinterpreted — Python / the table is the source of truth; the app only reads and
-- renders these columns verbatim.
SELECT
  proposal_id,
  agent_name,
  status,
  action_kind,
  risk_class,
  objective_metric,
  trigger_summary,
  created_at,
  generated_at
FROM austin_choi_omni_agent_catalog.agent_improvement_loop.agent_proposed_actions
ORDER BY created_at DESC
LIMIT 50
