import { useState } from 'react';
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

export function QuickConnectPanel({ onAdvanced }: { onAdvanced: () => void }) {
  const [copied, setCopied] = useState(false);
  const [name, setName] = useState('my-agent');
  const [objective, setObjective] = useState('Improve quality while reducing cost and latency.');

  const snippet = `from ail import improve\n\nagent = improve(\n    name=${JSON.stringify(name || 'my-agent')},\n    run=my_agent,\n    objective=${JSON.stringify(objective)},\n)\nresult = agent.run(task)`;

  async function copySnippet() {
    await navigator.clipboard?.writeText(snippet);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1800);
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
        <div className="rounded-md bg-muted/50 p-4">
          <pre className="overflow-x-auto text-xs leading-5">
            <code>{snippet}</code>
          </pre>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <Button onClick={copySnippet}>{copied ? 'Copied' : 'Copy starter code'}</Button>
          <Button variant="outline" onClick={onAdvanced}>
            Use advanced setup
          </Button>
          <span className="text-xs text-muted-foreground">
            The advanced path adds experiment, goals, readiness, and governance configuration.
          </span>
        </div>
      </CardContent>
    </Card>
  );
}
