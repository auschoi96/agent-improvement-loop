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
import type { ReactNode } from 'react';
import { CorpusKpis } from './components/CorpusKpis';

const EXPERIMENT_ID = '660599403165942';

// Stable empty-params reference shared by every parameter-free query/component.
const EMPTY_PARAMS = {};

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
  return (
    <div className="min-h-screen bg-background">
      <header className="border-b px-4 md:px-8 py-4">
        <div className="max-w-7xl mx-auto flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <h1 className="text-2xl font-bold text-foreground">Agent Self-Optimization</h1>
          <span className="text-sm text-muted-foreground">
            L0 deterministic leaderboard · experiment {EXPERIMENT_ID}
          </span>
        </div>
        <p className="max-w-7xl mx-auto mt-1 text-xs text-muted-foreground">
          Every metric is mechanically derived from trace metadata (tokens, timestamps, tool spans) — no model in the
          loop. Dollar figures are <strong>estimates</strong>.
        </p>
      </header>

      <main className="max-w-7xl mx-auto px-4 md:px-8 py-6 space-y-10">
        <Section title="Corpus summary" description="Headline L0 metrics across every session in the experiment.">
          <CorpusKpis />
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
                parameters={EMPTY_PARAMS}
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
                parameters={EMPTY_PARAMS}
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
                <DataTable queryKey="by_model" parameters={EMPTY_PARAMS} />
              </CardContent>
            </Card>
            <Card className="shadow-sm">
              <CardHeader>
                <CardTitle>By producer</CardTitle>
              </CardHeader>
              <CardContent>
                <DataTable queryKey="by_producer" parameters={EMPTY_PARAMS} />
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
                <DataTable queryKey="tool_waste_shell" parameters={EMPTY_PARAMS} pageSize={10} />
              </CardContent>
            </Card>
            <Card className="shadow-sm">
              <CardHeader>
                <CardTitle>Repeated file access</CardTitle>
                <CardDescription>Same file path read or edited repeatedly within a trace.</CardDescription>
              </CardHeader>
              <CardContent>
                <DataTable queryKey="tool_waste_files" parameters={EMPTY_PARAMS} pageSize={10} />
              </CardContent>
            </Card>
          </div>
        </Section>
      </main>
    </div>
  );
}
