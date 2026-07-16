-- Prompt-registry lineage / audit timeline for one agent: every registered prompt
-- version, newest first, with its provenance (source seed/gepa-evolved, the proven
-- held-out evolved/seed/delta savings, GEPA scores, candidate artifact), which
-- version is the CHAMPION, and an honest forced/not-an-improvement flag + reason.
-- All provenance is stamped at registration in the prompt registry and published to
-- agent_prompt_lineage by Tier A Python; this is read-only — the app renders the
-- audit trail, it never computes trust here.
-- @param agent_name STRING
-- @param experiment_id STRING
SELECT
  version,
  source,
  changed,
  gepa_best_val_score,
  gepa_num_candidates,
  holdout_evolved_savings_pct,
  holdout_seed_savings_pct,
  holdout_savings_delta_pct,
  candidate_artifact,
  suite_version,
  is_champion,
  is_forced_non_improving,
  registration_reason,
  uri,
  registered_at
FROM austin_choi_omni_agent_catalog.agent_improvement_loop.agent_prompt_lineage
WHERE agent_name = :agent_name
  AND experiment_id = :experiment_id
ORDER BY version DESC
