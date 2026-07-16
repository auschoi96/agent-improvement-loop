import { useEffect, useRef, useState } from 'react';
import {
  Badge,
  Button,
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Input,
  Label,
  Textarea,
} from '@databricks/appkit-ui/react';
import { postOnboardingJson } from '../lib/onboarding-api';

interface BootstrapResponse {
  outcome: string;
  agent_name?: string;
  experiment_id?: string;
  reviewer_experiment_id?: string;
  tracing_hint?: string;
  authored_judges?: string[];
  error?: string | null;
  request_id?: string;
  run_id?: number;
}

export function QuickConnectPanel({
  onAdvanced,
  onRegistered,
}: {
  onAdvanced: () => void;
  onRegistered: (name: string) => void;
}) {
  const [copied, setCopied] = useState(false);
  const [name, setName] = useState('my-agent');
  const [objective, setObjective] = useState('Improve quality while reducing cost and latency.');
  const [targetWorkspace, setTargetWorkspace] = useState('');
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<BootstrapResponse | null>(null);
  const setupController = useRef<AbortController | null>(null);
  const copyTimer = useRef<number | null>(null);

  useEffect(
    () => () => {
      setupController.current?.abort();
      if (copyTimer.current !== null) window.clearTimeout(copyTimer.current);
    },
    []
  );

  const experimentId = result?.experiment_id;
  const snippet = `from ail import improve\n\nagent = improve(\n    name=${JSON.stringify(name || 'my-agent')},\n    run=my_agent,\n    objective=${JSON.stringify(objective)},${experimentId ? `\n    experiment_id=${JSON.stringify(experimentId)},` : ''}\n)\nresult = agent.run(task)`;

  async function setup() {
    setupController.current?.abort();
    const controller = new AbortController();
    setupController.current = controller;
    setBusy(true);
    setResult(null);
    try {
      const { ok, body } = await postOnboardingJson<BootstrapResponse>(
        '/api/onboarding/bootstrap',
        {
          agent_name: name.trim(),
          requirements_text: objective.trim(),
          ...(targetWorkspace.trim() ? { target_workspace: targetWorkspace.trim() } : {}),
        },
        { signal: controller.signal }
      );
      if (controller.signal.aborted) return;
      setResult(body);
      if (ok && body.outcome === 'registered' && body.agent_name) onRegistered(body.agent_name);
    } catch {
      if (controller.signal.aborted) return;
      setResult({ outcome: 'error', error: 'Network error while setting up the agent.' });
    } finally {
      if (!controller.signal.aborted) setBusy(false);
      if (setupController.current === controller) setupController.current = null;
    }
  }

  async function copySnippet() {
    await navigator.clipboard?.writeText(snippet);
    setCopied(true);
    if (copyTimer.current !== null) window.clearTimeout(copyTimer.current);
    copyTimer.current = window.setTimeout(() => {
      copyTimer.current = null;
      setCopied(false);
    }, 1800);
  }

  return (
    <Card className="border-primary/30 shadow-sm">
      <CardHeader>
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <CardTitle>Quick connect</CardTitle>
            <CardDescription>Instrument any Python agent, HTTP wrapper, or LLM call in minutes.</CardDescription>
          </div>
          <Badge variant="outline">Recommended</Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-5">
        <div className="grid gap-4 md:grid-cols-2">
          <div className="space-y-2">
            <Label htmlFor="quick-agent-name">Agent name</Label>
            <Input
              id="quick-agent-name"
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="my-agent"
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="quick-objective">What should improve?</Label>
            <Textarea
              id="quick-objective"
              value={objective}
              onChange={(event) => setObjective(event.target.value)}
              rows={2}
            />
          </div>
        </div>
        <div className="space-y-2">
          <Label htmlFor="quick-target-workspace">Local project the companion may edit</Label>
          <Input
            id="quick-target-workspace"
            value={targetWorkspace}
            onChange={(event) => setTargetWorkspace(event.target.value)}
            placeholder="/path/to/your/agent/repo"
          />
        </div>
        <div className="rounded-md bg-muted/50 p-4">
          <pre className="overflow-x-auto text-xs leading-5">
            <code>{snippet}</code>
          </pre>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <Button onClick={() => void setup()} disabled={busy || !name.trim() || !objective.trim()}>
            {busy ? 'Setting up experiments, judges, and jobs…' : 'Set up agent'}
          </Button>
          <Button variant="outline" onClick={() => void copySnippet()} disabled={!experimentId}>
            {copied ? 'Copied' : 'Copy configured starter code'}
          </Button>
          <Button variant="outline" onClick={onAdvanced}>
            Use advanced setup
          </Button>
          <span className="text-xs text-muted-foreground">
            The advanced path adds experiment, goals, readiness, and governance configuration.
          </span>
        </div>
        {result && (
          <div
            className={`rounded-md border p-3 text-sm ${
              result.outcome === 'registered' ? 'text-emerald-700 dark:text-emerald-300' : 'text-destructive'
            }`}
          >
            {result.outcome === 'registered' ? (
              <>
                Ready: subject experiment <span className="font-mono">{result.experiment_id}</span>, reviewer experiment{' '}
                <span className="font-mono">{result.reviewer_experiment_id}</span>, judges{' '}
                {(result.authored_judges ?? []).join(', ') || 'none required'}.
              </>
            ) : (
              (result.error ?? 'Setup failed.')
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
