import { useMemo, useState } from 'react';
import { useAnalyticsQuery, Card, CardContent, Badge, Button, Textarea, Skeleton } from '@databricks/appkit-ui/react';
import { sql } from '@databricks/appkit-ui/js';
import {
  actionKindLabel,
  buildDecisionRequest,
  changeUnderReview,
  decisionMessage,
  gateSummary,
  isPending,
  proofSummary,
  rejectReasonError,
  riskClassLabel,
  sortRows,
  type DecisionKind,
  type DecisionResponse,
  type DecisionTone,
  type ProposedActionRow,
} from '../lib/approvals';

const DECISION_ENDPOINT = '/api/approvals/decision';

const TONE_CLASS: Record<DecisionTone, string> = {
  success: 'text-emerald-700 dark:text-emerald-300',
  warning: 'text-amber-700 dark:text-amber-300',
  error: 'text-destructive',
};

// The in-app approval queue (Phase C lane 3b) — the human control plane that closes
// the loop. Pending proposals show the WHY (trigger), the WHAT (the exact change),
// the PROOF (frozen-suite delta with correctness held), and the GATE status, so the
// reviewer approves on evidence. Approve/Reject POST to the authenticated server
// route (ail.loop.apply_service via a custom AppKit plugin); after a decision that
// changes state the list is remounted (via `reloadKey`) so it refetches.
export function ApprovalQueue({ agentName }: { agentName: string }) {
  const [reloadKey, setReloadKey] = useState(0);
  return <QueueList key={reloadKey} agentName={agentName} onDecided={() => setReloadKey((k) => k + 1)} />;
}

function QueueList({ agentName, onDecided }: { agentName: string; onDecided: () => void }) {
  const params = useMemo(() => ({ agent_name: sql.string(agentName) }), [agentName]);
  const { data, loading, error } = useAnalyticsQuery('proposed_actions', params);

  if (loading) {
    return (
      <div className="space-y-3">
        {Array.from({ length: 2 }, (_, i) => (
          <Skeleton key={`approval-skeleton-${i}`} className="h-40 w-full" />
        ))}
      </div>
    );
  }
  if (error) {
    return <div className="text-destructive bg-destructive/10 p-3 rounded-md">Error: {error}</div>;
  }

  const rows = sortRows((data ?? []) as ProposedActionRow[]);
  const pending = rows.filter(isPending);
  const decided = rows.filter((r) => !isPending(r));

  if (pending.length === 0 && decided.length === 0) {
    return (
      <div className="text-muted-foreground border rounded-md p-4">
        No pending proposals — the controller has not proposed any change for this agent (or every proposal has been
        decided).
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {pending.length === 0 ? (
        <div className="text-muted-foreground border rounded-md p-4">No pending proposals.</div>
      ) : (
        <ol className="space-y-4">
          {pending.map((row) => (
            <li key={row.proposal_id}>
              <ProposalCard row={row} onDecided={onDecided} />
            </li>
          ))}
        </ol>
      )}

      {decided.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-semibold text-muted-foreground">Recently decided</h3>
          <ul className="space-y-1">
            {decided.map((row) => (
              <li key={row.proposal_id} className="text-sm text-muted-foreground flex items-center gap-2">
                <Badge variant={row.status === 'applied' ? 'default' : 'outline'}>{row.status}</Badge>
                <span>{actionKindLabel(row.action_kind)}</span>
                <span className="text-xs">· {row.trigger_summary}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function ProposalCard({ row, onDecided }: { row: ProposedActionRow; onDecided: () => void }) {
  const [reason, setReason] = useState('');
  const [busy, setBusy] = useState<DecisionKind | null>(null);
  const [message, setMessage] = useState<{ tone: DecisionTone; text: string } | null>(null);
  const change = changeUnderReview(row);
  const rejectError = rejectReasonError(reason);

  async function decide(decision: DecisionKind) {
    setMessage(null);
    let request;
    try {
      request = buildDecisionRequest(row, decision, reason);
    } catch (err) {
      setMessage({ tone: 'error', text: err instanceof Error ? err.message : String(err) });
      return;
    }
    setBusy(decision);
    try {
      const res = await fetch(DECISION_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
      });
      const body = (await res.json().catch(() => ({}))) as DecisionResponse;
      if (!res.ok && !body.outcome) {
        setMessage({
          tone: 'error',
          text:
            res.status === 401
              ? 'Not authenticated — sign in to approve or reject.'
              : `Request failed (${res.status}).`,
        });
        return;
      }
      const msg = decisionMessage(body);
      setMessage(msg);
      // Refresh the queue only when the proposal's state actually changed.
      if (['applied', 'rejected', 'applied_unrecorded'].includes(body.outcome)) {
        onDecided();
      }
    } catch (err) {
      setMessage({ tone: 'error', text: err instanceof Error ? err.message : 'Network error.' });
    } finally {
      setBusy(null);
    }
  }

  return (
    <Card className="shadow-sm">
      <CardContent className="p-4 space-y-3">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-base font-semibold">{actionKindLabel(row.action_kind)}</span>
          <Badge variant={row.risk_class === 'agent_change' ? 'destructive' : 'outline'}>
            {riskClassLabel(row.risk_class)}
          </Badge>
          <Badge variant="outline">objective: {row.objective_metric}</Badge>
        </div>

        <div className="space-y-1 text-sm">
          <p>
            <span className="font-semibold">Why:</span> {row.trigger_summary}
            {row.trigger_n_traces > 0 && (
              <span className="text-muted-foreground"> ({row.trigger_n_traces} traces)</span>
            )}
            {row.trigger_judge_name && (
              <span className="text-muted-foreground"> · certifying judge {row.trigger_judge_name}</span>
            )}
          </p>
          <p>
            <span className="font-semibold">Proof:</span> {proofSummary(row)}
            {row.proof_suite_version && (
              <span className="text-muted-foreground"> · suite {row.proof_suite_version}</span>
            )}
          </p>
          <p>
            <span className="font-semibold">Gate:</span> {gateSummary(row)}
          </p>
        </div>

        <details className="text-sm">
          <summary className="cursor-pointer text-muted-foreground">
            {change.label} — the exact change under review
          </summary>
          <pre className="mt-2 max-h-64 overflow-auto rounded-md bg-muted p-3 text-xs whitespace-pre-wrap">
            {change.body}
          </pre>
        </details>

        <div className="space-y-2">
          <Textarea
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            placeholder="Reason (required to reject; optional context on approve)…"
            rows={2}
          />
          <div className="flex flex-wrap items-center gap-2">
            <Button onClick={() => void decide('approve')} disabled={busy !== null}>
              {busy === 'approve' ? 'Approving…' : 'Approve'}
            </Button>
            <Button
              variant="destructive"
              onClick={() => void decide('reject')}
              disabled={busy !== null || rejectError !== null}
              title={rejectError ?? undefined}
            >
              {busy === 'reject' ? 'Rejecting…' : 'Reject'}
            </Button>
            {message && <span className={`text-sm ${TONE_CLASS[message.tone]}`}>{message.text}</span>}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
