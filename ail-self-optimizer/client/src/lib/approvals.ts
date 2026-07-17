// Approval-queue view logic (Phase C lane 3b), kept as pure functions so the queue
// rendering + the decision request/response mapping are unit-testable without a DOM
// (mirrors lib/lineage.ts). The ApprovalQueue component is a thin renderer over these.
import { fmtInt, fmtPct, fmtSignedPct } from './formatters';
import { toBool } from './lineage';

export const STATUS_PENDING = 'pending';

export type DecisionKind = 'approve' | 'reject';

// One row of config/queries/proposed_actions.sql. SQL scalars can arrive as strings
// at runtime (and nullable columns as null), so the helpers below coerce defensively.
export interface ProposedActionRow {
  proposal_id: string;
  agent_name: string;
  experiment_id: string;
  status: string;
  action_kind: string;
  risk_class: string;
  objective_metric: string;
  created_at: string;
  // why
  trigger_kind: string;
  trigger_summary: string;
  trigger_metric: string;
  trigger_observed_value: number;
  trigger_threshold: number;
  trigger_n_traces: number;
  trigger_judge_name: string;
  // what
  change_kind: string;
  change_summary: string;
  change_sql: string;
  change_diff: string;
  change_evolved_body_ref: string;
  change_revert_target: string;
  change_local_apply_spec_json: string;
  local_apply_status: string;
  local_apply_error: string;
  local_apply_completed_at: string;
  local_apply_pre_change_ref: string;
  local_apply_validation_output: string;
  // proof
  proof_proved_improvement: boolean;
  proof_correctness_held: boolean;
  proof_realized_savings_pct: number;
  proof_n_promote: number;
  proof_n_block: number;
  proof_n_errored: number;
  proof_suite_version: string;
  // gate
  gate_readiness_tier: string;
  gate_gated: boolean;
  gate_judge_agreement: number;
  gate_scored_coverage: number;
  gate_n_distrusted_judges: number;
  // verify (opt-in Tier-2 "verify on my suite" lifecycle). Nullable — a proposal
  // that was never verify-requested has verify_status null/''.
  verify_requested: boolean;
  verify_status: string;
  verify_requested_by: string;
  verify_requested_at: string;
  verify_completed_at: string;
  verify_error: string;
}

// The action kinds the frozen suite can run a baseline-vs-candidate proof for — a
// skill / instruction / prompt-behaviour change. MUST mirror PROVABLE_ACTION_KINDS in
// src/ail/loop/verify_service.py: the client greys the "Verify on my suite" button for
// a non-provable kind (metric view / revert / agent task), and the engine independently
// refuses a request for one (defence in depth — the browser is never trusted).
const PROVABLE_ACTION_KINDS: ReadonlySet<string> = new Set(['skill_update', 'instruction_update', 'gepa_prompt']);

export const isProvable = (actionKind: string): boolean => PROVABLE_ACTION_KINDS.has(actionKind);

const ACTION_KIND_LABELS: Record<string, string> = {
  metric_view: 'Metric view',
  skill_update: 'Skill update',
  instruction_update: 'Instruction update',
  gepa_prompt: 'GEPA-evolved prompt',
  revert: 'Revert',
};

export const actionKindLabel = (kind: string): string => ACTION_KIND_LABELS[kind] ?? kind;

const RISK_CLASS_LABELS: Record<string, string> = {
  additive_asset: 'additive asset · low blast radius',
  agent_change: 'agent change · higher blast radius',
};

export const riskClassLabel = (riskClass: string): string => RISK_CLASS_LABELS[riskClass] ?? riskClass;

export const isPending = (row: { status?: string | null }): boolean =>
  (row.status ?? '').toLowerCase() === STATUS_PENDING;

// A SQL numeric that may be null/'' at runtime — distinguish "no value" from 0.
const hasNum = (v: number | string | null | undefined): boolean =>
  v !== null && v !== undefined && String(v) !== '' && !Number.isNaN(Number(v));

// The PROOF line: the frozen-suite objective delta WITH correctness held, plus the
// promote/block counts. Honest by construction — a proposal only exists when proven
// (fail-closed), so if a row ever shows unproven it is flagged, never dressed up.
export function proofSummary(row: ProposedActionRow): string {
  const proven = toBool(row.proof_proved_improvement);
  const correct = toBool(row.proof_correctness_held);
  const savings = hasNum(row.proof_realized_savings_pct) ? fmtSignedPct(row.proof_realized_savings_pct) : '—';
  const counts = `${fmtInt(row.proof_n_promote)} promote / ${fmtInt(row.proof_n_block)} block`;
  if (proven && correct) {
    return `Proven: ${savings} on ${row.objective_metric}, correctness held · ${counts}`;
  }
  return `NOT proven / correctness not held — should never have been proposed (fail-closed) · ${counts}`;
}

// The GATE line: readiness tier + certifying-judge agreement + scored coverage +
// distrusted-judge count. Coverage/agreement are fractions (0..1) shown as %.
export function gateSummary(row: ProposedActionRow): string {
  const tier = row.gate_readiness_tier || 'unknown';
  const agreement = hasNum(row.gate_judge_agreement) ? fmtPct(row.gate_judge_agreement) : 'n/a';
  const coverage = hasNum(row.gate_scored_coverage) ? fmtPct(row.gate_scored_coverage) : 'n/a';
  return `readiness ${tier} · judge agreement ${agreement} · scored coverage ${coverage} · ${fmtInt(row.gate_n_distrusted_judges)} distrusted judge(s)`;
}

export interface ChangeView {
  label: string;
  body: string;
}

// The concrete change under review, picking whichever payload field the change kind
// populated (the human approves exactly what ships).
export function changeUnderReview(row: ProposedActionRow): ChangeView {
  if (row.action_kind === 'gepa_prompt' && row.change_diff) {
    return { label: 'Proposed local prompt / skill rewrite diff', body: row.change_diff };
  }
  if (row.change_sql) return { label: 'Metric-view CREATE SQL', body: row.change_sql };
  if (row.change_diff) return { label: 'Skill / instruction diff', body: row.change_diff };
  if (row.change_evolved_body_ref) return { label: 'GEPA-evolved body reference', body: row.change_evolved_body_ref };
  if (row.change_revert_target) return { label: 'Revert target', body: row.change_revert_target };
  return { label: 'Change', body: row.change_summary || '(no change body recorded)' };
}

export interface LocalApplySpecView {
  target_kind: string;
  target_path: string;
  artifact_uri: string;
  baseline_sha256: string;
  candidate_sha256: string;
  validation_command: string[];
  validation_timeout_seconds: number;
  holdout_evolved_savings_pct?: number | null;
  holdout_seed_savings_pct?: number | null;
  holdout_savings_delta_pct?: number | null;
  holdout_task_ids?: string[];
}

export function localApplySpec(row: ProposedActionRow): LocalApplySpecView | null {
  if (!row.change_local_apply_spec_json) return null;
  try {
    const value = JSON.parse(row.change_local_apply_spec_json) as Partial<LocalApplySpecView>;
    if (
      typeof value.target_path !== 'string' ||
      typeof value.artifact_uri !== 'string' ||
      !Array.isArray(value.validation_command)
    ) {
      return null;
    }
    return value as LocalApplySpecView;
  } catch {
    return null;
  }
}

// Pending first (they need a decision), then most-recently-created — mirrors the
// SQL ORDER BY so a client re-sort after an optimistic refresh stays consistent.
export function sortRows(rows: readonly ProposedActionRow[]): ProposedActionRow[] {
  return [...rows].sort((a, b) => {
    const ap = isPending(a) ? 0 : 1;
    const bp = isPending(b) ? 0 : 1;
    if (ap !== bp) return ap - bp;
    return (b.created_at || '').localeCompare(a.created_at || '');
  });
}

export interface DecisionRequest {
  proposal_id: string;
  agent_name: string;
  decision: DecisionKind;
  reason?: string;
}

// A reject must carry a reason (it is signal — it tells the controller a rule
// mis-fired). Returns an error string when invalid, else null.
export function rejectReasonError(reason: string | null | undefined): string | null {
  return reason && reason.trim() ? null : 'A reject needs a reason — it tells the controller a rule mis-fired.';
}

// Build the POST body for the authenticated approve/reject route. The approver is
// NOT included here: it is resolved server-side from the authenticated request
// (never trusted from the browser). Throws if a reject has no reason (fail-closed).
export function buildDecisionRequest(
  row: Pick<ProposedActionRow, 'proposal_id' | 'agent_name'>,
  decision: DecisionKind,
  reason?: string
): DecisionRequest {
  if (decision === 'reject') {
    const err = rejectReasonError(reason);
    if (err) throw new Error(err);
    return { proposal_id: row.proposal_id, agent_name: row.agent_name, decision, reason: reason?.trim() };
  }
  return {
    proposal_id: row.proposal_id,
    agent_name: row.agent_name,
    decision,
    ...(reason && reason.trim() ? { reason: reason.trim() } : {}),
  };
}

// The server action's result (ail.loop.apply_service.ApplyServiceResult), narrowed
// to what the queue renders.
export interface DecisionResponse {
  outcome: string;
  status?: string;
  approver?: string;
  summary?: string;
  refused_reason?: string | null;
  error?: string | null;
  new_version?: number | null;
  created_view?: string | null;
  reverted_to_version?: number | null;
  lineage_recorded?: boolean;
}

export type DecisionTone = 'success' | 'error' | 'warning';

// Map an outcome to a human, honest message + tone. An applied-but-unrecorded apply
// is a WARNING (the change is live; the audit must be reconciled), never a plain
// success; a refusal surfaces WHY it was refused.
export function decisionMessage(resp: DecisionResponse): { tone: DecisionTone; text: string } {
  switch (resp.outcome) {
    case 'applied': {
      const detail =
        resp.created_view ??
        (resp.new_version != null ? `champion → v${resp.new_version}` : undefined) ??
        (resp.reverted_to_version != null ? `reverted to v${resp.reverted_to_version}` : undefined) ??
        resp.summary;
      return { tone: 'success', text: `Applied${detail ? ` — ${detail}` : ''}.` };
    }
    case 'approved':
      return {
        tone: 'success',
        text: 'Approved — waiting for the local companion to verify, snapshot, rewrite, and validate the reviewed artifact.',
      };
    case 'rejected':
      return { tone: 'success', text: 'Rejected — recorded; nothing was applied.' };
    case 'refused':
      return {
        tone: 'error',
        text: `Refused — ${resp.refused_reason ?? 'a fail-closed gate blocked the apply'}. Nothing was applied.`,
      };
    case 'applied_unrecorded':
      return {
        tone: 'warning',
        text: `Applied, but the audit record failed — reconcile. ${resp.error ?? ''}`.trim(),
      };
    default:
      return { tone: 'error', text: `Error — ${resp.error ?? 'the decision could not be completed'}.` };
  }
}

// ---------------------------------------------------------------------------
// Opt-in Tier-2 "Verify on my suite" (L9) — request + result rendering.
// ---------------------------------------------------------------------------

// The POST body for the authenticated "Verify on my suite" route. The requester is NOT
// included — it is resolved server-side from the authenticated request (never trusted
// from the browser), exactly like the approver on a decision.
export interface VerifyRequest {
  proposal_id: string;
  agent_name: string;
}

export function buildVerifyRequest(row: Pick<ProposedActionRow, 'proposal_id' | 'agent_name'>): VerifyRequest {
  return { proposal_id: row.proposal_id, agent_name: row.agent_name };
}

// The verify route's result (ail.loop.verify_service.VerifyRequestResult), narrowed to
// what the queue renders. This is the outcome of REQUESTING a proof, not the proof.
export interface VerifyResponse {
  outcome: string;
  verify_status?: string | null;
  action_kind?: string | null;
  refused_reason?: string | null;
  error?: string | null;
}

// Map a verify-REQUEST outcome to an honest message + tone. A "requested" is a neutral
// success (the proof runs later on the companion); a refusal surfaces WHY it was
// refused; a bridge/infra failure is an honest error — never a fabricated proof.
export function verifyRequestMessage(resp: VerifyResponse): { tone: DecisionTone; text: string } {
  switch (resp.outcome) {
    case 'requested':
      return {
        tone: 'success',
        text: 'Verification requested — the frozen-suite proof runs on your companion; refresh for the result.',
      };
    case 'refused':
      return {
        tone: 'error',
        text: `Not requested — ${resp.refused_reason ?? 'a fail-closed gate blocked the request'}.`,
      };
    default:
      return { tone: 'error', text: `Error — ${resp.error ?? 'the verify request could not be completed'}.` };
  }
}

export interface VerifyEvidence {
  tone: DecisionTone;
  label: string;
  detail: string;
}

// The ADDED Tier-2 evidence block, rendered NEXT TO the Tier-1 judge/RLM proof/gate
// lines. Honest by construction: a blocked / errored / no-suite verify is shown AS SUCH
// and never dressed up as verified; every number comes straight from the proof_* the
// companion wrote (SELECT-only — the app never recomputes proving). Returns null when
// no verify was ever requested for this proposal.
export function verifyEvidence(row: ProposedActionRow): VerifyEvidence | null {
  const status = (row.verify_status ?? '').toLowerCase();
  if (!status) return null;
  switch (status) {
    case 'requested':
      return {
        tone: 'warning',
        label: 'Tier-2 verify: requested',
        detail: 'Frozen-suite proof queued on the companion — refresh for the result.',
      };
    case 'verified': {
      const savings = hasNum(row.proof_realized_savings_pct) ? fmtSignedPct(row.proof_realized_savings_pct) : '—';
      return {
        tone: 'success',
        label: 'Tier-2 verify: PROMOTE',
        detail: `Frozen-suite proof: ${savings} on ${row.objective_metric}, correctness held · ${fmtInt(row.proof_n_promote)} promote / ${fmtInt(row.proof_n_block)} block${row.proof_suite_version ? ` · suite ${row.proof_suite_version}` : ''}`,
      };
    }
    case 'blocked':
      return {
        tone: 'error',
        label: 'Tier-2 verify: BLOCK',
        detail: `Frozen-suite proof did NOT clear the bar — ${fmtInt(row.proof_n_promote)} promote / ${fmtInt(row.proof_n_block)} block / ${fmtInt(row.proof_n_errored)} errored${row.proof_correctness_held ? '' : ', correctness not held'}.`,
      };
    case 'errored':
      return {
        tone: 'error',
        label: 'Tier-2 verify: errored',
        detail: `The proof could not run — ${row.verify_error || 'unknown error'}. No proof recorded (fail-closed).`,
      };
    case 'no_suite':
      return {
        tone: 'error',
        label: 'Tier-2 verify: no suite',
        detail: `No frozen suite is configured — ${row.verify_error || 'the deployer must freeze a suite first'}. Nothing was proven (fail-closed).`,
      };
    default:
      return {
        tone: 'warning',
        label: `Tier-2 verify: ${status}`,
        detail: row.verify_error || 'Unknown verify state.',
      };
  }
}
