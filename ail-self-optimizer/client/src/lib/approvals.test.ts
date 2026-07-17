import { describe, it, expect } from 'vitest';
import {
  actionKindLabel,
  buildDecisionRequest,
  buildVerifyRequest,
  changeUnderReview,
  decisionMessage,
  gateSummary,
  isPending,
  isProvable,
  proofSummary,
  rejectReasonError,
  riskClassLabel,
  sortRows,
  verifyEvidence,
  verifyRequestMessage,
  type ProposedActionRow,
} from './approvals';

// Overrides are loosely typed so a test can inject a runtime null (SQL columns can
// arrive null even though the generated row type is non-null) — one assertion, no
// double-assertion (appkit lint: no-double-type-assertion).
function row(overrides: Partial<Record<keyof ProposedActionRow, unknown>> = {}): ProposedActionRow {
  const base: ProposedActionRow = {
    proposal_id: 'p1',
    agent_name: 'claude_code',
    experiment_id: '660599403165942',
    status: 'pending',
    action_kind: 'metric_view',
    risk_class: 'additive_asset',
    objective_metric: 'total_tokens',
    created_at: '2026-06-30T00:00:00+00:00',
    trigger_kind: 'rlm_recommended_asset',
    trigger_summary: 'RLM recommended a token-waste view',
    trigger_metric: 'total_tokens',
    trigger_observed_value: 0,
    trigger_threshold: 0,
    trigger_n_traces: 12,
    trigger_judge_name: '',
    change_kind: 'metric_view_sql',
    change_summary: 'token waste view',
    change_sql: 'CREATE OR REPLACE VIEW cat.sch.v AS SELECT 1',
    change_diff: '',
    change_evolved_body_ref: '',
    change_revert_target: '',
    change_local_apply_spec_json: '',
    local_apply_status: '',
    local_apply_error: '',
    local_apply_completed_at: '',
    local_apply_pre_change_ref: '',
    local_apply_validation_output: '',
    proof_proved_improvement: true,
    proof_correctness_held: true,
    proof_realized_savings_pct: 35.4,
    proof_n_promote: 3,
    proof_n_block: 0,
    proof_n_errored: 0,
    proof_suite_version: 'phase2-mini',
    gate_readiness_tier: 'ready_to_prove',
    gate_gated: true,
    gate_judge_agreement: 0.92,
    gate_scored_coverage: 0.8,
    gate_n_distrusted_judges: 0,
    verify_requested: false,
    verify_status: '',
    verify_requested_by: '',
    verify_requested_at: '',
    verify_completed_at: '',
    verify_error: '',
  };
  return { ...base, ...overrides } as ProposedActionRow;
}

describe('isPending', () => {
  it('matches only the pending status (case-insensitive)', () => {
    expect(isPending({ status: 'pending' })).toBe(true);
    expect(isPending({ status: 'PENDING' })).toBe(true);
    expect(isPending({ status: 'applied' })).toBe(false);
    expect(isPending({ status: null })).toBe(false);
  });
});

describe('labels', () => {
  it('humanizes action kinds and risk classes, passing unknowns through', () => {
    expect(actionKindLabel('gepa_prompt')).toMatch(/gepa/i);
    expect(actionKindLabel('mystery')).toBe('mystery');
    expect(riskClassLabel('agent_change')).toMatch(/higher blast radius/);
  });
});

describe('proofSummary — honest by construction', () => {
  it('states the proven delta with correctness held and promote/block counts', () => {
    const s = proofSummary(row());
    expect(s).toMatch(/Proven/);
    expect(s).toMatch(/\+35\.4%/);
    expect(s).toMatch(/correctness held/);
    expect(s).toMatch(/3 promote \/ 0 block/);
  });

  it('flags an unproven row instead of dressing it up (fail-closed)', () => {
    const s = proofSummary(row({ proof_proved_improvement: false }));
    expect(s).toMatch(/NOT proven/);
    expect(s).not.toMatch(/^Proven/);
  });

  it('renders an em dash rather than a fake 0% when savings is missing', () => {
    const s = proofSummary(row({ proof_realized_savings_pct: null }));
    expect(s).toMatch(/—/);
  });
});

describe('gateSummary', () => {
  it('shows readiness tier, judge agreement, coverage as percentages, and distrust count', () => {
    const s = gateSummary(row());
    expect(s).toMatch(/readiness ready_to_prove/);
    expect(s).toMatch(/judge agreement 92\.0%/);
    expect(s).toMatch(/scored coverage 80\.0%/);
    expect(s).toMatch(/0 distrusted/);
  });

  it('shows n/a for a missing judge agreement rather than 0%', () => {
    const s = gateSummary(row({ gate_judge_agreement: null }));
    expect(s).toMatch(/judge agreement n\/a/);
  });
});

describe('changeUnderReview', () => {
  it('picks the populated change payload by kind', () => {
    expect(changeUnderReview(row()).label).toMatch(/CREATE SQL/);
    const diff = changeUnderReview(row({ change_sql: '', change_diff: '--- a\n+++ b' }));
    expect(diff.label).toMatch(/diff/i);
    expect(diff.body).toContain('+++ b');
    const revert = changeUnderReview(row({ change_sql: '', change_kind: 'revert_ref', change_revert_target: 'v3' }));
    expect(revert.label).toMatch(/Revert/);
    expect(revert.body).toBe('v3');
  });
});

describe('sortRows — pending first', () => {
  it('orders pending ahead of decided, then most recent first', () => {
    const rows = [
      row({ proposal_id: 'applied-old', status: 'applied', created_at: '2026-06-01T00:00:00Z' }),
      row({ proposal_id: 'pending-old', status: 'pending', created_at: '2026-06-02T00:00:00Z' }),
      row({ proposal_id: 'pending-new', status: 'pending', created_at: '2026-06-29T00:00:00Z' }),
    ];
    expect(sortRows(rows).map((r) => r.proposal_id)).toEqual(['pending-new', 'pending-old', 'applied-old']);
  });
});

describe('buildDecisionRequest — approver is never client-supplied', () => {
  it('builds an approve body without an approver field', () => {
    const req = buildDecisionRequest(row(), 'approve');
    expect(req).toEqual({ proposal_id: 'p1', agent_name: 'claude_code', decision: 'approve' });
    expect('approver' in req).toBe(false);
  });

  it('requires a reason for a reject (fail-closed)', () => {
    expect(rejectReasonError('')).toMatch(/needs a reason/);
    expect(rejectReasonError('mis-fired')).toBeNull();
    expect(() => buildDecisionRequest(row(), 'reject')).toThrow(/reason/);
    expect(buildDecisionRequest(row(), 'reject', ' not useful ')).toEqual({
      proposal_id: 'p1',
      agent_name: 'claude_code',
      decision: 'reject',
      reason: 'not useful',
    });
  });
});

describe('decisionMessage — honest outcomes', () => {
  it('applied is success with the artifact detail', () => {
    expect(decisionMessage({ outcome: 'applied', created_view: 'cat.sch.v' })).toEqual({
      tone: 'success',
      text: 'Applied — cat.sch.v.',
    });
  });

  it('refused surfaces the reason and states nothing was applied', () => {
    const m = decisionMessage({ outcome: 'refused', refused_reason: 'gate re-check failed' });
    expect(m.tone).toBe('error');
    expect(m.text).toMatch(/gate re-check failed/);
    expect(m.text).toMatch(/Nothing was applied/);
  });

  it('applied_unrecorded is a WARNING (live change, reconcile the audit) — never plain success', () => {
    const m = decisionMessage({ outcome: 'applied_unrecorded', error: 'lineage write failed' });
    expect(m.tone).toBe('warning');
    expect(m.text).toMatch(/reconcile/);
  });
});

describe('isProvable — only kinds the frozen suite can run', () => {
  it('is true for skill/instruction/prompt changes, false for asset/revert/agent-task', () => {
    expect(isProvable('skill_update')).toBe(true);
    expect(isProvable('instruction_update')).toBe(true);
    expect(isProvable('gepa_prompt')).toBe(true);
    expect(isProvable('metric_view')).toBe(false);
    expect(isProvable('revert')).toBe(false);
    expect(isProvable('agent_task')).toBe(false);
  });
});

describe('buildVerifyRequest — requester is never client-supplied', () => {
  it('builds a body with only proposal_id + agent_name (requester is server-side)', () => {
    const req = buildVerifyRequest(row({ action_kind: 'skill_update' }));
    expect(req).toEqual({ proposal_id: 'p1', agent_name: 'claude_code' });
    expect('requested_by' in req).toBe(false);
  });
});

describe('verifyRequestMessage — honest request outcomes', () => {
  it('requested is a neutral success pointing at the companion', () => {
    const m = verifyRequestMessage({ outcome: 'requested' });
    expect(m.tone).toBe('success');
    expect(m.text).toMatch(/requested/i);
  });

  it('refused surfaces the reason', () => {
    const m = verifyRequestMessage({ outcome: 'refused', refused_reason: 'cannot be proven on the frozen suite' });
    expect(m.tone).toBe('error');
    expect(m.text).toMatch(/cannot be proven/);
  });

  it('error is honest — no fabricated proof', () => {
    const m = verifyRequestMessage({ outcome: 'error', error: 'verify-service exited 1' });
    expect(m.tone).toBe('error');
    expect(m.text).toMatch(/exited 1/);
  });
});

describe('verifyEvidence — Tier-2 proof shown honestly (fail-closed)', () => {
  it('returns null when no verify was ever requested', () => {
    expect(verifyEvidence(row())).toBeNull();
  });

  it('requested is a pending/warning state, not a proof', () => {
    const e = verifyEvidence(row({ verify_status: 'requested' }));
    expect(e?.tone).toBe('warning');
    expect(e?.label).toMatch(/requested/i);
  });

  it('verified shows the PROMOTE delta with correctness held', () => {
    const e = verifyEvidence(row({ verify_status: 'verified' }));
    expect(e?.tone).toBe('success');
    expect(e?.label).toMatch(/PROMOTE/);
    expect(e?.detail).toMatch(/\+35\.4%/);
    expect(e?.detail).toMatch(/correctness held/);
  });

  it('blocked is shown AS a block — never dressed up as verified', () => {
    const e = verifyEvidence(
      row({
        verify_status: 'blocked',
        proof_proved_improvement: false,
        proof_correctness_held: false,
        proof_n_block: 2,
      })
    );
    expect(e?.tone).toBe('error');
    expect(e?.label).toMatch(/BLOCK/);
    expect(e?.detail).toMatch(/correctness not held/);
  });

  it('errored is an honest error state with no proof', () => {
    const e = verifyEvidence(row({ verify_status: 'errored', verify_error: 'prove failed: RuntimeError: boom' }));
    expect(e?.tone).toBe('error');
    expect(e?.detail).toMatch(/boom/);
    expect(e?.detail).toMatch(/fail-closed/);
  });

  it('no_suite fails closed with an honest "no frozen suite" message', () => {
    const e = verifyEvidence(row({ verify_status: 'no_suite', verify_error: 'no frozen suite configured' }));
    expect(e?.tone).toBe('error');
    expect(e?.label).toMatch(/no suite/i);
    expect(e?.detail).toMatch(/fail-closed/);
  });
});
