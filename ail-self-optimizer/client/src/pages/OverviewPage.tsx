import { useMemo } from 'react';
import {
  BarChart,
  Badge,
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Skeleton,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from '@databricks/appkit-ui/react';
import { sql } from '@databricks/appkit-ui/js';
import { PageHeader } from '../shell/PageHeader';
import { RequireAgent } from '../shell/RequireAgent';
import { PanelBoundary } from '../shell/PanelBoundary';
import { CorpusKpis } from '../components/CorpusKpis';
import { BRAND_ACCENT } from '../lib/theme';
import { RefreshableAnalyticsQuery } from '../components/RefreshableAnalyticsQuery';
import { StableDataTable } from '../components/StableDataTable';

// Overview — the L0 leaderboard, re-homed from the old single-scroll App. Headline
// KPIs stay pinned at the top; the deeper diagnostics move into tabs so the surface no
// longer dumps everything into one long scroll. Every panel keeps its exact query,
// parameters, and ESTIMATE labeling — only the layout changed.
function OverviewBody({ experimentId }: { experimentId: string }) {
  // Memoize the shared :experiment_id binding so the per-agent queries don't refetch on
  // every render (AppKit parameter guidance).
  const expParams = useMemo(() => ({ experiment_id: sql.string(experimentId) }), [experimentId]);

  return (
    <div className="space-y-6">
      <PanelBoundary title="Corpus summary failed to load">
        <CorpusKpis experimentId={experimentId} />
      </PanelBoundary>

      <Tabs defaultValue="tail" className="space-y-4">
        <TabsList>
          <TabsTrigger value="tail">Token heavy tail</TabsTrigger>
          <TabsTrigger value="breakdown">Breakdown</TabsTrigger>
          <TabsTrigger value="waste">Tool waste</TabsTrigger>
        </TabsList>

        <TabsContent value="tail" className="space-y-6">
          <PanelBoundary title="Token heavy tail failed to load">
            <Card className="shadow-sm">
              <CardHeader>
                <CardTitle>Top sessions by total tokens</CardTitle>
                <CardDescription>Largest 15 sessions; the tail is where token spend lives.</CardDescription>
              </CardHeader>
              <CardContent>
                <RefreshableAnalyticsQuery queryKey="session_token_bars" parameters={expParams}>
                  {({ data, loading, refreshing, error }) => (
                    <div className="space-y-2">
                      <div className="flex justify-end">
                        {refreshing && <Badge variant="outline">Refreshing…</Badge>}
                        {error && data && <Badge variant="outline">Refresh failed; showing prior data</Badge>}
                      </div>
                      {loading ? (
                        <Skeleton className="h-80 w-full" />
                      ) : error && !data ? (
                        <p className="text-sm text-destructive">Error: {error}</p>
                      ) : (
                        <BarChart
                          data={data ?? []}
                          xKey="trace"
                          yKey="total_tokens"
                          colors={[BRAND_ACCENT]}
                          height={320}
                        />
                      )}
                    </div>
                  )}
                </RefreshableAnalyticsQuery>
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
                <StableDataTable
                  queryKey="high_token_sessions"
                  parameters={expParams}
                  filterColumn="model"
                  filterPlaceholder="Filter by model…"
                  pageSize={10}
                />
              </CardContent>
            </Card>
          </PanelBoundary>
        </TabsContent>

        <TabsContent value="breakdown" className="space-y-4">
          <p className="text-sm text-muted-foreground">
            Tokens, estimated cost, and tool calls rolled up by model and by producer.
          </p>
          <PanelBoundary title="Breakdown failed to load">
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              <Card className="shadow-sm">
                <CardHeader>
                  <CardTitle>By model</CardTitle>
                </CardHeader>
                <CardContent>
                  <StableDataTable queryKey="by_model" parameters={expParams} />
                </CardContent>
              </Card>
              <Card className="shadow-sm">
                <CardHeader>
                  <CardTitle>By producer</CardTitle>
                </CardHeader>
                <CardContent>
                  <StableDataTable queryKey="by_producer" parameters={expParams} />
                </CardContent>
              </Card>
            </div>
          </PanelBoundary>
        </TabsContent>

        <TabsContent value="waste" className="space-y-4">
          <p className="text-sm text-muted-foreground">
            Repeated tool work within a session: re-run shell-setup boilerplate and re-targeted files.
          </p>
          <PanelBoundary title="Tool-waste diagnosis failed to load">
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              <Card className="shadow-sm">
                <CardHeader>
                  <CardTitle>Boilerplate re-runs</CardTitle>
                  <CardDescription>
                    Same normalized shell prologue (cd / env setup) repeated within a trace.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <StableDataTable queryKey="tool_waste_shell" parameters={expParams} pageSize={10} />
                </CardContent>
              </Card>
              <Card className="shadow-sm">
                <CardHeader>
                  <CardTitle>Repeated file access</CardTitle>
                  <CardDescription>Same file path read or edited repeatedly within a trace.</CardDescription>
                </CardHeader>
                <CardContent>
                  <StableDataTable queryKey="tool_waste_files" parameters={expParams} pageSize={10} />
                </CardContent>
              </Card>
            </div>
          </PanelBoundary>
        </TabsContent>
      </Tabs>
    </div>
  );
}

export function OverviewPage() {
  return (
    <div className="space-y-6">
      <PageHeader />
      <RequireAgent>{(agent) => <OverviewBody experimentId={agent.experiment_id} />}</RequireAgent>
    </div>
  );
}
