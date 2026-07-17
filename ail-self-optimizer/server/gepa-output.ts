import { z } from 'zod';

export const GEPA_RESULT_MARKER = 'AIL_GEPA_RESULT=';

const gepaResultSchema = z.object({
  schema_version: z.literal('ail.jobs.gepa_result/v1'),
  agent_name: z.string(),
  subject_experiment_id: z.string(),
  reviewer_experiment_id: z.string(),
  mlflow_run_id: z.string(),
  mlflow_run_url: z.string().url().nullable(),
  artifact_path: z.string(),
  artifact_uri: z.string(),
  optimizer: z.string(),
  proposal_id: z.string().nullable(),
  proposal_status: z.string().nullable(),
  proposal_created: z.boolean(),
  proposal_reason: z.string(),
  candidate_changed: z.boolean(),
  candidate_promoted: z.literal(false),
  human_gate_required: z.literal(true),
  suite_version: z.string(),
  suite_content_hash: z.string(),
  max_metric_calls: z.number(),
  gepa_total_metric_calls: z.number().nullable(),
  gepa_num_candidates: z.number().nullable(),
  gepa_best_val_score: z.number().nullable(),
  holdout_savings_delta_pct: z.number().nullable(),
  holdout_evolved_savings_pct: z.number().nullable(),
  holdout_seed_savings_pct: z.number().nullable(),
});

export type GepaCandidateResult = z.infer<typeof gepaResultSchema>;

interface RunOutputLike {
  logs?: unknown;
}

// Wheel-task stdout is the durable handoff surface for the AppKit Jobs plugin.
// Parse only the explicitly prefixed, schema-validated line; arbitrary job logs are
// never trusted as UI state and are not returned to the browser.
export function extractGepaResult(output: unknown): GepaCandidateResult | null {
  if (typeof output !== 'object' || output === null) return null;
  const logs = (output as RunOutputLike).logs;
  if (typeof logs !== 'string') return null;

  const lines = logs.split(/\r?\n/).reverse();
  for (const line of lines) {
    const markerAt = line.indexOf(GEPA_RESULT_MARKER);
    if (markerAt < 0) continue;
    const raw = line.slice(markerAt + GEPA_RESULT_MARKER.length).trim();
    try {
      const parsed: unknown = JSON.parse(raw);
      const result = gepaResultSchema.safeParse(parsed);
      if (result.success) return result.data;
    } catch {
      // Keep scanning in case a later stderr fragment contains marker-like text.
    }
  }
  return null;
}
