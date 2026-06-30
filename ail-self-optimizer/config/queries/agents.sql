-- Registered agents for the agent switcher. One MLflow experiment per agent
-- (docs/OBSERVABILITY_APP.md); the registry is the app's primary key onto agents.
-- Read-only from the Python-published agent_registry table.
SELECT
  agent_name,
  experiment_id,
  description
FROM austin_choi_omni_agent_catalog.agent_improvement_loop.agent_registry
ORDER BY agent_name
