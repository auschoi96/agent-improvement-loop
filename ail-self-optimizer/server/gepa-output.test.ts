import { describe, expect, it } from 'vitest';
import { extractGepaResult, GEPA_RESULT_MARKER } from './gepa-output';

const candidate = {
  schema_version: 'ail.jobs.gepa_result/v1',
  agent_name: 'claude_code',
  subject_experiment_id: 'subject-1',
  reviewer_experiment_id: 'reviewer-1',
  mlflow_run_id: 'run-1',
  mlflow_run_url: 'https://workspace.example/ml/experiments/reviewer-1/runs/run-1',
  artifact_path: 'gepa/gepa_candidate.json',
  artifact_uri: 'runs:/run-1/gepa/gepa_candidate.json',
  optimizer: 'gepa.optimize (Optimize Anything)',
  proposal_id: 'proposal-1',
  proposal_status: 'pending',
  proposal_created: true,
  proposal_reason: 'held-out savings delta +4.2 pct-pts beats seed',
  candidate_changed: true,
  candidate_promoted: false,
  human_gate_required: true,
  suite_version: 'phase2-mini-v1',
  suite_content_hash: 'abc',
  max_metric_calls: 6,
  gepa_total_metric_calls: 8,
  gepa_num_candidates: 2,
  gepa_best_val_score: 0.75,
  holdout_savings_delta_pct: 4.2,
  holdout_evolved_savings_pct: 12.2,
  holdout_seed_savings_pct: 8,
};

describe('extractGepaResult', () => {
  it('extracts the schema-validated candidate marker from wheel logs', () => {
    const logs = `startup\n${GEPA_RESULT_MARKER}${JSON.stringify(candidate)}\nfinished`;
    expect(extractGepaResult({ logs })).toEqual(candidate);
  });

  it('fails closed on arbitrary, malformed, or promoted output', () => {
    expect(extractGepaResult({ logs: 'ordinary output' })).toBeNull();
    expect(extractGepaResult({ logs: `${GEPA_RESULT_MARKER}{bad` })).toBeNull();
    expect(
      extractGepaResult({
        logs: `${GEPA_RESULT_MARKER}${JSON.stringify({ ...candidate, candidate_promoted: true })}`,
      })
    ).toBeNull();
  });
});
