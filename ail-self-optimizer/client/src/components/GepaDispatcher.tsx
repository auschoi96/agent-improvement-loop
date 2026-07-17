import { useEffect, useMemo, useState, type FormEvent } from 'react';
import {
  Alert,
  AlertDescription,
  AlertTitle,
  Badge,
  Button,
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Checkbox,
  Input,
  Label,
  Separator,
} from '@databricks/appkit-ui/react';
import {
  AlertTriangle,
  CheckCircle2,
  ExternalLink,
  LoaderCircle,
  Play,
  RotateCcw,
  ShieldCheck,
  Sparkles,
  XCircle,
} from 'lucide-react';
import type { AgentRow } from '../context/agent-context';
import {
  dispatchGepaRun,
  fetchGepaOutput,
  fetchGepaRun,
  GEPA_POLL_INTERVAL_MS,
  GEPA_REFLECTION_LM,
  GEPA_SUITE_VERSION,
  gepaRunLabel,
  isGepaSupportedAgent,
  isSuccessfulGepaRun,
  isTerminalGepaRun,
  type GepaCandidateResult,
  type GepaRun,
} from '../lib/gepa';

const DEFAULT_METRIC_CALLS = 6;
const DEFAULT_HOLDOUT_FRACTION = 0.4;
const DEFAULT_MAX_TRAIN_TASKS = 2;

function storageKey(agentName: string, experimentId: string): string {
  return `ail.gepa.active-run.${agentName}.${experimentId}`;
}

function readStoredRun(agentName: string, experimentId: string): number | null {
  try {
    const value = Number(window.localStorage.getItem(storageKey(agentName, experimentId)));
    return Number.isSafeInteger(value) && value > 0 ? value : null;
  } catch {
    return null;
  }
}

function storeRun(agentName: string, experimentId: string, runId: number | null): void {
  try {
    const key = storageKey(agentName, experimentId);
    if (runId === null) window.localStorage.removeItem(key);
    else window.localStorage.setItem(key, String(runId));
  } catch {
    // Storage is an optional continuity aid; polling still works in this mount.
  }
}

function percent(value: number | null): string {
  return value === null ? '—' : `${value.toFixed(2)}%`;
}

function numberOrDash(value: number | null): string {
  return value === null ? '—' : String(value);
}

function CandidateSummary({ result }: { result: GepaCandidateResult }) {
  return (
    <Card className="border-emerald-500/30">
      <CardHeader>
        <div className="flex flex-wrap items-center gap-2">
          <CheckCircle2 className="h-5 w-5 text-emerald-600" />
          <CardTitle>Candidate ready for review</CardTitle>
          <Badge variant="outline">not promoted</Badge>
        </div>
        <CardDescription>
          The candidate and held-out proof were logged to the reviewer experiment. No active agent configuration
          changed. {result.proposal_created ? 'A pending local-rewrite approval was created.' : result.proposal_reason}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-3 text-sm sm:grid-cols-2 lg:grid-cols-4">
          <div>
            <p className="text-xs text-muted-foreground">Candidate changed</p>
            <p className="font-medium">{result.candidate_changed ? 'Yes' : 'No — seed remained best'}</p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Held-out savings delta</p>
            <p className="font-medium tabular-nums">{percent(result.holdout_savings_delta_pct)}</p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Metric calls</p>
            <p className="font-medium tabular-nums">{numberOrDash(result.gepa_total_metric_calls)}</p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Candidates explored</p>
            <p className="font-medium tabular-nums">{numberOrDash(result.gepa_num_candidates)}</p>
          </div>
        </div>
        <Separator />
        <div className="space-y-1 text-sm">
          <p>
            <span className="text-muted-foreground">Optimizer:</span>{' '}
            <span className="font-mono text-xs">{result.optimizer}</span>
          </p>
          <p>
            <span className="text-muted-foreground">MLflow run:</span>{' '}
            {result.mlflow_run_url ? (
              <a
                href={result.mlflow_run_url}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1 font-mono text-xs text-primary hover:underline"
              >
                {result.mlflow_run_id} <ExternalLink className="h-3 w-3" />
              </a>
            ) : (
              <span className="font-mono text-xs">{result.mlflow_run_id}</span>
            )}
          </p>
          <p>
            <span className="text-muted-foreground">Candidate artifact:</span>{' '}
            <span className="break-all font-mono text-xs">{result.artifact_uri}</span>
          </p>
          <p>
            <span className="text-muted-foreground">Reviewer experiment:</span>{' '}
            <span className="font-mono text-xs">{result.reviewer_experiment_id}</span>
          </p>
        </div>
        <Alert>
          <ShieldCheck className="h-4 w-4" />
          <AlertTitle>{result.proposal_created ? 'Pending approval created' : 'No applyable proposal'}</AlertTitle>
          <AlertDescription>
            {result.proposal_created ? (
              <>
                Review the exact local diff, target hashes, held-out evidence, and validation command on the{' '}
                <a href={`/approvals?agent=${encodeURIComponent(result.agent_name)}`} className="underline">
                  Approvals page
                </a>
                . Approval waits for the local companion; this hosted app cannot write your machine.
              </>
            ) : (
              <>The artifact remains in MLflow, but it did not clear the held-out gate: {result.proposal_reason}</>
            )}
          </AlertDescription>
        </Alert>
      </CardContent>
    </Card>
  );
}

export function GepaDispatcher({ agent }: { agent: AgentRow }) {
  const [maxMetricCalls, setMaxMetricCalls] = useState(DEFAULT_METRIC_CALLS);
  const [holdoutFraction, setHoldoutFraction] = useState(DEFAULT_HOLDOUT_FRACTION);
  const [maxTrainTasks, setMaxTrainTasks] = useState(DEFAULT_MAX_TRAIN_TASKS);
  const [confirmed, setConfirmed] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [runId, setRunId] = useState<number | null>(() => readStoredRun(agent.agent_name, agent.experiment_id));
  const [run, setRun] = useState<GepaRun | null>(null);
  const [candidate, setCandidate] = useState<GepaCandidateResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const supported = isGepaSupportedAgent(agent.agent_name);
  const hasReviewer = Boolean(agent.reviewer_experiment_id?.trim());
  const active = runId !== null && !isTerminalGepaRun(run);
  const validBudget =
    Number.isInteger(maxMetricCalls) &&
    maxMetricCalls >= 1 &&
    maxMetricCalls <= 500 &&
    holdoutFraction > 0 &&
    holdoutFraction < 1 &&
    Number.isInteger(maxTrainTasks) &&
    maxTrainTasks >= 1 &&
    maxTrainTasks <= 20;

  const statusTone = useMemo(() => {
    if (!run || !isTerminalGepaRun(run)) return 'outline' as const;
    return isSuccessfulGepaRun(run) ? ('secondary' as const) : ('destructive' as const);
  }, [run]);

  useEffect(() => {
    if (runId === null) return;
    let cancelled = false;
    let timer: number | undefined;

    const poll = async () => {
      try {
        const latest = await fetchGepaRun(runId);
        if (cancelled) return;
        setRun(latest);
        setError(null);
        if (isTerminalGepaRun(latest)) {
          storeRun(agent.agent_name, agent.experiment_id, null);
          if (isSuccessfulGepaRun(latest)) {
            const output = await fetchGepaOutput(runId);
            if (cancelled) return;
            if (output.result) setCandidate(output.result);
            else {
              setError(
                output.task_error ??
                  'The job succeeded but returned no validated GEPA candidate marker. Open the job run for details.'
              );
            }
          }
          return;
        }
      } catch (reason) {
        if (cancelled) return;
        setError(reason instanceof Error ? reason.message : 'Could not refresh the GEPA run');
      }
      timer = window.setTimeout(() => void poll(), GEPA_POLL_INTERVAL_MS);
    };

    void poll();
    return () => {
      cancelled = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [agent.agent_name, agent.experiment_id, runId]);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!supported || !hasReviewer || !confirmed || !validBudget || active) return;
    setSubmitting(true);
    setError(null);
    setCandidate(null);
    setRun(null);
    try {
      const id = await dispatchGepaRun({
        agentName: agent.agent_name,
        experimentId: agent.experiment_id,
        maxMetricCalls,
        holdoutFraction,
        maxTrainTasks,
      });
      storeRun(agent.agent_name, agent.experiment_id, id);
      setRunId(id);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'GEPA dispatch failed');
    } finally {
      setSubmitting(false);
    }
  };

  const reset = () => {
    storeRun(agent.agent_name, agent.experiment_id, null);
    setRunId(null);
    setRun(null);
    setCandidate(null);
    setError(null);
    setConfirmed(false);
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <div className="flex flex-wrap items-center gap-2">
            <Sparkles className="h-5 w-5" />
            <CardTitle>GEPA candidate search</CardTitle>
            <Badge variant={supported ? 'secondary' : 'outline'}>
              {supported ? 'Claude Code adapter ready' : 'adapter unavailable'}
            </Badge>
          </div>
          <CardDescription>
            Launch a bounded Databricks Job for this agent. Status polling updates this panel only—the page and your
            work are never refreshed.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form className="space-y-5" onSubmit={(event) => void submit(event)}>
            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-1.5">
                <Label>Agent</Label>
                <Input value={agent.agent_name} readOnly aria-label="GEPA agent" />
              </div>
              <div className="space-y-1.5">
                <Label>Subject experiment</Label>
                <Input value={agent.experiment_id} readOnly aria-label="GEPA subject experiment" />
              </div>
              <div className="space-y-1.5">
                <Label>Reviewer experiment</Label>
                <Input
                  value={agent.reviewer_experiment_id ?? 'Not configured'}
                  readOnly
                  aria-label="GEPA reviewer experiment"
                />
              </div>
              <div className="space-y-1.5">
                <Label>Frozen suite</Label>
                <Input value={GEPA_SUITE_VERSION} readOnly aria-label="GEPA frozen suite" />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="gepa-metric-calls">Maximum metric calls</Label>
                <Input
                  id="gepa-metric-calls"
                  type="number"
                  min={1}
                  max={500}
                  value={maxMetricCalls}
                  disabled={active}
                  onChange={(event) => setMaxMetricCalls(Number(event.target.value))}
                />
                <p className="text-xs text-muted-foreground">
                  Dominant optimizer budget; higher values increase cost and runtime.
                </p>
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="gepa-max-train">Maximum train tasks</Label>
                <Input
                  id="gepa-max-train"
                  type="number"
                  min={1}
                  max={20}
                  value={maxTrainTasks}
                  disabled={active}
                  onChange={(event) => setMaxTrainTasks(Number(event.target.value))}
                />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="gepa-holdout">Holdout fraction</Label>
                <Input
                  id="gepa-holdout"
                  type="number"
                  min={0.1}
                  max={0.9}
                  step={0.1}
                  value={holdoutFraction}
                  disabled={active}
                  onChange={(event) => setHoldoutFraction(Number(event.target.value))}
                />
                <p className="text-xs text-muted-foreground">
                  Held-out tasks are never exposed to GEPA during optimization.
                </p>
              </div>
              <div className="space-y-1.5">
                <Label>Reflection model</Label>
                <Input value={GEPA_REFLECTION_LM} readOnly aria-label="GEPA reflection model" />
              </div>
            </div>

            {!supported && (
              <Alert>
                <AlertTriangle className="h-4 w-4" />
                <AlertTitle>This agent cannot run GEPA on job compute yet</AlertTitle>
                <AlertDescription>
                  Coding-agent execution is currently implemented only for the registered claude_code adapter. Codex and
                  custom agents fail closed until their executable adapters are packaged for this job.
                </AlertDescription>
              </Alert>
            )}
            {!hasReviewer && (
              <Alert variant="destructive">
                <AlertTriangle className="h-4 w-4" />
                <AlertTitle>Reviewer experiment required</AlertTitle>
                <AlertDescription>
                  Finish onboarding this agent with a separate reviewer experiment before dispatching optimizer work.
                </AlertDescription>
              </Alert>
            )}

            <div className="flex items-start gap-3 rounded-md border p-4">
              <Checkbox
                id="confirm-gepa-cost"
                checked={confirmed}
                disabled={active}
                onCheckedChange={(checked) => setConfirmed(checked === true)}
              />
              <div className="space-y-1">
                <Label htmlFor="confirm-gepa-cost">I understand this launches a live, costly coding-agent run</Label>
                <p className="text-xs text-muted-foreground">
                  GEPA evaluates real baseline and candidate arms. The result is a reviewable candidate and is never
                  promoted automatically.
                </p>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <Button
                type="submit"
                disabled={!supported || !hasReviewer || !confirmed || !validBudget || active || submitting}
              >
                {submitting || active ? (
                  <LoaderCircle className="h-4 w-4 animate-spin" />
                ) : (
                  <Play className="h-4 w-4" />
                )}
                {submitting ? 'Dispatching…' : active ? 'GEPA is running' : 'Run GEPA'}
              </Button>
              {runId !== null && isTerminalGepaRun(run) && (
                <Button type="button" variant="outline" onClick={reset}>
                  <RotateCcw className="h-4 w-4" /> Configure another run
                </Button>
              )}
            </div>
          </form>
        </CardContent>
      </Card>

      {runId !== null && (
        <Card>
          <CardHeader>
            <div className="flex flex-wrap items-center gap-2">
              {active ? (
                <LoaderCircle className="h-5 w-5 animate-spin text-primary" />
              ) : isSuccessfulGepaRun(run) ? (
                <CheckCircle2 className="h-5 w-5 text-emerald-600" />
              ) : (
                <XCircle className="h-5 w-5 text-destructive" />
              )}
              <CardTitle>Databricks Job run</CardTitle>
              <Badge variant={statusTone}>{gepaRunLabel(run)}</Badge>
            </div>
            <CardDescription>
              Run {runId}. This status is fetched in the background; it does not navigate, reload, or remount the page.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {run?.state?.state_message && <p className="text-sm text-muted-foreground">{run.state.state_message}</p>}
            {run?.run_page_url && (
              <Button asChild variant="outline" size="sm">
                <a href={run.run_page_url} target="_blank" rel="noreferrer">
                  Open job run <ExternalLink className="h-3.5 w-3.5" />
                </a>
              </Button>
            )}
          </CardContent>
        </Card>
      )}

      {error && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>GEPA dispatcher error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {candidate && <CandidateSummary result={candidate} />}
    </div>
  );
}
