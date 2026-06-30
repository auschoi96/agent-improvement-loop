import {
  BarChart,
  DataTable,
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Badge,
} from '@databricks/appkit-ui/react';
import { sql } from '@databricks/appkit-ui/js';
import { useMemo, useState, type ReactNode } from 'react';
import { CorpusKpis } from './components/CorpusKpis';
import { AgentSwitcher, type AgentRow } from './components/AgentSwitcher';
import { VersionComparison } from './components/VersionComparison';

const BRAND_BLUE = '#40d1f5';

function Section({ title, description, children }: { title: string; description?: string; children: ReactNode }) {
  return (
    <section className="space-y-3">
      <div>
        <h2 className="text-xl font-semibold text-foreground">{title}</h2>
        {description && <p className="text-sm text-muted-foreground">{description}</p>}
      </div>
      {children}
    </section>
  );
}

export default function App() {
  const [agent, setAgent] = useState<AgentRow | null>(null);
  const experimentId = agent?.experiment_id ?? '';

  // Memoize the shared :experiment_id binding so the per-agent queries don't
  // refetch on every render (AppKit parameter guidance).
  const expParams = useMemo(() => ({ experiment_id: sql.string(experimentId) }), [experimentId]);

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b px-4 md:px-8 py-4">
        <div className="max-w-7xl mx-auto flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <h1 className="text-2xl font-bold text-foreground">Agent Self-Optimization</h1>
          <span className="text-sm text-muted-foreground">
            Multi-agent observability · L0 deterministic leaderboard + version comparison
          </span>
        </div>
        <p className="max-w-7xl mx-auto mt-1 text-xs text-muted-foreground">
          Every metric is mechanically derived from trace metadata (tokens, timestamps, tool spans) — no model in the
          loop. Dollar figures are <strong>estimates</strong>. One MLflow experiment per agent.
        </p>
        <div className="max-w-7xl mx-auto mt-3">
          <AgentSwitcher value={agent?.agent_name ?? null} onChange={setAgent} />
        </div>
      </header>

      {!agent ? (
        <main className="max-w-7xl mx-auto px-4 md:px-8 py-6">
          <p className="text-muted-foreground">Select an agent to view its metrics.</p>
        </main>
      ) : (
        <main className="max-w-7xl mx-auto px-4 md:px-8 py-6 space-y-10">
          <Section
            title="Baseline vs new version"
            description="Within this agent's experiment, a baseline agent_version vs a newer one — L0 deltas, with readiness honestly gating the trust verdict (never a green improvement the readiness wall has not cleared)."
          >
            <VersionComparison agentName={agent.agent_name} />
          </Section>

          <Section title="Corpus summary" description="Headline L0 metrics across every session in the experiment.">
            <CorpusKpis experimentId={experimentId} />
          </Section>

          <Section
            title="Token heavy tail"
            description="A low median with a long tail — a few enormous sessions hold most of the spend."
          >
            <Card className="shadow-sm">
              <CardHeader>
                <CardTitle>Top sessions by total tokens</CardTitle>
                <CardDescription>Largest 15 sessions; the tail is where token spend lives.</CardDescription>
              </CardHeader>
              <CardContent>
                <BarChart
                  queryKey="session_token_bars"
                  parameters={expParams}
                  xKey="trace"
                  yKey="total_tokens"
                  colors={[BRAND_BLUE]}
                  height={320}
                />
              </CardContent>
            </Card>

            <Card className="shadow-sm">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  High-token sessions
                  <Badge variant="outline">cost = ESTIMATE</Badge>
                </CardTitle>
                <CardDescription>
                  Per-session tokens, tool calls, duration, estimated cost, and strict redundancy rate.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <DataTable
                  queryKey="high_token_sessions"
                  parameters={expParams}
                  filterColumn="model"
                  filterPlaceholder="Filter by model…"
                  pageSize={10}
                />
              </CardContent>
            </Card>
          </Section>

          <Section
            title="Breakdown"
            description="Tokens, estimated cost, and tool calls rolled up by model and by producer."
          >
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card className="shadow-sm">
                <CardHeader>
                  <CardTitle>By model</CardTitle>
                </CardHeader>
                <CardContent>
                  <DataTable queryKey="by_model" parameters={expParams} />
                </CardContent>
              </Card>
              <Card className="shadow-sm">
                <CardHeader>
                  <CardTitle>By producer</CardTitle>
                </CardHeader>
                <CardContent>
                  <DataTable queryKey="by_producer" parameters={expParams} />
                </CardContent>
              </Card>
            </div>
          </Section>

          <Section
            title="Tool-waste diagnosis"
            description="Repeated tool work within a session: re-run shell-setup boilerplate and re-targeted files."
          >
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card className="shadow-sm">
                <CardHeader>
                  <CardTitle>Boilerplate re-runs</CardTitle>
                  <CardDescription>
                    Same normalized shell prologue (cd / env setup) repeated within a trace.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <DataTable queryKey="tool_waste_shell" parameters={expParams} pageSize={10} />
                </CardContent>
              </Card>
              <Card className="shadow-sm">
                <CardHeader>
                  <CardTitle>Repeated file access</CardTitle>
                  <CardDescription>Same file path read or edited repeatedly within a trace.</CardDescription>
                </CardHeader>
                <CardContent>
                  <DataTable queryKey="tool_waste_files" parameters={expParams} pageSize={10} />
                </CardContent>
              </Card>
            </div>
          </Section>
        </main>
      )}
    </div>
  );
}
