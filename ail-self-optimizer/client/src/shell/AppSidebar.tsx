import { useMemo, type ComponentType } from 'react';
import { Link, useLocation } from 'react-router';
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuBadge,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarRail,
  SidebarSeparator,
  useAnalyticsQuery,
} from '@databricks/appkit-ui/react';
import { sql } from '@databricks/appkit-ui/js';
import {
  Activity,
  BookOpen,
  Bot,
  ClipboardCheck,
  GitBranch,
  GitCompareArrows,
  LayoutDashboard,
  Plus,
  Sparkles,
  Tags,
  type LucideProps,
} from 'lucide-react';
import {
  HELP_NAV,
  PRIMARY_NAV,
  agentSearch,
  isNavItemActive,
  pendingCount,
  type NavItem,
  type NavKey,
} from '../lib/navigation';
import { useAgent } from '../context/agent-context';
import type { ProposedActionRow } from '../lib/approvals';

const ICONS: Record<NavKey, ComponentType<LucideProps>> = {
  overview: LayoutDashboard,
  compare: GitCompareArrows,
  optimize: Sparkles,
  approvals: ClipboardCheck,
  labeling: Tags,
  activity: Activity,
  lineage: GitBranch,
  'add-agent': Plus,
  'how-it-works': BookOpen,
};

// The Approvals pending-count badge. Reuses the SAME proposed_actions query and the
// SAME isPending predicate the queue uses (via pendingCount), so the badge can never
// disagree with the queue. Mounted only when an agent is selected. No new query.
function ApprovalsBadge({ agentName, experimentId }: { agentName: string; experimentId: string }) {
  const params = useMemo(
    () => ({ agent_name: sql.string(agentName), experiment_id: sql.string(experimentId) }),
    [agentName, experimentId]
  );
  const { data } = useAnalyticsQuery('proposed_actions', params);
  const count = pendingCount((data ?? null) as ProposedActionRow[] | null);
  if (count === 0) return null;
  return <SidebarMenuBadge>{count}</SidebarMenuBadge>;
}

function NavMenuItem({
  item,
  pathname,
  search,
  agentName,
  experimentId,
}: {
  item: NavItem;
  pathname: string;
  search: string;
  agentName: string | null;
  experimentId: string | null;
}) {
  const Icon = ICONS[item.key];
  return (
    <SidebarMenuItem>
      <SidebarMenuButton asChild isActive={isNavItemActive(item, pathname)} tooltip={item.label}>
        <Link to={{ pathname: item.path, search }}>
          <Icon />
          <span>{item.label}</span>
        </Link>
      </SidebarMenuButton>
      {item.key === 'approvals' && agentName && experimentId && (
        <ApprovalsBadge agentName={agentName} experimentId={experimentId} />
      )}
    </SidebarMenuItem>
  );
}

export function AppSidebar() {
  const { pathname } = useLocation();
  const { selected } = useAgent();
  const agentName = selected?.agent_name ?? null;
  const experimentId = selected?.experiment_id ?? null;
  const search = agentSearch(agentName);

  return (
    <Sidebar collapsible="icon">
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton asChild size="lg" tooltip="Agent Self-Optimization">
              <Link to={{ pathname: '/overview', search }}>
                <div
                  className="flex aspect-square size-8 items-center justify-center rounded-md text-background"
                  style={{ backgroundColor: 'var(--brand)' }}
                >
                  <Bot className="size-4" />
                </div>
                <div className="flex flex-col gap-0.5 leading-none">
                  <span className="font-semibold">Agent Self-Optimization</span>
                  <span className="text-xs text-muted-foreground">L0 observability</span>
                </div>
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>

      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Agent</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {PRIMARY_NAV.map((item) => (
                <NavMenuItem
                  key={item.key}
                  item={item}
                  pathname={pathname}
                  search={search}
                  agentName={agentName}
                  experimentId={experimentId}
                />
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter>
        <SidebarSeparator />
        <SidebarMenu>
          {HELP_NAV.map((item) => (
            <NavMenuItem
              key={item.key}
              item={item}
              pathname={pathname}
              search={search}
              agentName={agentName}
              experimentId={experimentId}
            />
          ))}
        </SidebarMenu>
      </SidebarFooter>

      <SidebarRail />
    </Sidebar>
  );
}
