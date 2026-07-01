// The app's information architecture as pure data + pure functions, kept free of
// React/JSX so the routing/active-state/breadcrumb logic is unit-testable in the
// node vitest environment (mirrors lib/approvals.ts, lib/lineage.ts). The shell
// components (AppSidebar, TopBar, PageHeader) are thin renderers over this module;
// icons live in the components, not here, so this stays framework-free.
import { isPending, type ProposedActionRow } from './approvals';

export type NavKey = 'overview' | 'compare' | 'approvals' | 'activity' | 'lineage' | 'add-agent' | 'how-it-works';

export interface NavItem {
  key: NavKey;
  /** Sidebar label. */
  label: string;
  /** Route path — the single source of truth for URL-addressable views. */
  path: string;
  /** Page <h1> shown by PageHeader when `showTitle` is true. */
  title: string;
  /** Page subtitle + sidebar tooltip (icon-collapsed mode). */
  description: string;
  /** Agent-scoped pages render an honest "Select an agent" empty when none is chosen. */
  requiresAgent: boolean;
  /** Bare-panel pages render a PageHeader <h1>; flow/help pages let their own Card
   *  header be the hero (avoids a duplicate title), so PageHeader shows breadcrumb only. */
  showTitle: boolean;
}

// Primary IA — grouped by task, one route each. Overview is the default landing.
export const PRIMARY_NAV: NavItem[] = [
  {
    key: 'overview',
    label: 'Overview',
    path: '/overview',
    title: 'Overview',
    description:
      'Deterministic L0 leaderboard for the selected agent — headline metrics, the token heavy tail, and tool-waste diagnostics.',
    requiresAgent: true,
    showTitle: true,
  },
  {
    key: 'compare',
    label: 'Compare',
    path: '/compare',
    title: 'Baseline vs new version',
    description:
      'Within this agent’s experiment, a baseline version vs a newer one — L0 deltas, with readiness honestly gating the trust verdict.',
    requiresAgent: true,
    showTitle: true,
  },
  {
    key: 'approvals',
    label: 'Approvals',
    path: '/approvals',
    title: 'Approval queue',
    description:
      'The human control plane: review the why + the evidence behind a fail-closed wall, then approve or reject the live apply. The app’s only write-path.',
    requiresAgent: true,
    showTitle: true,
  },
  {
    key: 'activity',
    label: 'Activity',
    path: '/activity',
    title: 'Activity',
    description: 'What the framework has actually been doing — real job runs and proposal outcomes. Read-only.',
    requiresAgent: false,
    showTitle: false,
  },
  {
    key: 'lineage',
    label: 'Lineage',
    path: '/lineage',
    title: 'Lineage & audit timeline',
    description:
      'Every registered prompt version newest-first — what changed, the proven held-out delta, and which version is the CHAMPION. The trail that lets a change be reverted.',
    requiresAgent: true,
    showTitle: true,
  },
];

// Help / action items — a distinct onboarding flow and the guided tour. Rendered in
// the sidebar footer, not the primary list.
export const HELP_NAV: NavItem[] = [
  {
    key: 'add-agent',
    label: 'Add agent',
    path: '/add-agent',
    title: 'Add an agent',
    description: 'Register a new agent from a fresh MLflow experiment.',
    requiresAgent: false,
    showTitle: false,
  },
  {
    key: 'how-it-works',
    label: 'How it works',
    path: '/how-it-works',
    title: 'How it works',
    description: 'A guided tour of the self-optimization loop and its readiness gates.',
    requiresAgent: false,
    showTitle: false,
  },
];

export const ALL_NAV: NavItem[] = [...PRIMARY_NAV, ...HELP_NAV];

// The landing route. '/' and any unknown path resolve here.
export const DEFAULT_PATH = '/overview';

const normalizePath = (pathname: string): string => {
  if (!pathname || pathname === '/') return DEFAULT_PATH;
  // Drop a trailing slash (but never reduce '/' to '').
  return pathname.length > 1 && pathname.endsWith('/') ? pathname.slice(0, -1) : pathname;
};

// The nav item a pathname maps to (root → Overview). null for a genuinely unknown path.
export function navItemForPath(pathname: string): NavItem | null {
  const p = normalizePath(pathname);
  return ALL_NAV.find((item) => item.path === p) ?? null;
}

export function navKeyForPath(pathname: string): NavKey | null {
  return navItemForPath(pathname)?.key ?? null;
}

// Active-state predicate for a sidebar item, driven by the current path (SidebarMenuButton isActive).
export function isNavItemActive(item: NavItem, pathname: string): boolean {
  return navKeyForPath(pathname) === item.key;
}

export interface Crumb {
  label: string;
  /** Present for a linkable ancestor crumb; absent for the current page. */
  path?: string;
}

// Breadcrumb trail: the product root (→ Overview) then the current section (current
// page is not a link). An unknown path still yields the root crumb so the header
// never renders empty.
export function breadcrumbsForPath(pathname: string): Crumb[] {
  const root: Crumb = { label: 'Agent Self-Optimization', path: DEFAULT_PATH };
  const item = navItemForPath(pathname);
  if (!item || item.key === 'overview') {
    return [{ label: root.label }, { label: 'Overview' }];
  }
  return [root, { label: item.title }];
}

// The query string that carries the selected agent across navigation, so nav links
// and refresh/share preserve the selection. Empty when no agent is selected.
export function agentSearch(agentName: string | null | undefined): string {
  return agentName ? `?agent=${encodeURIComponent(agentName)}` : '';
}

// The Approvals sidebar badge count — reuses the proposed_actions rows the queue
// already fetches and the same isPending predicate, so the badge can never drift
// from the queue. No new query.
export function pendingCount(rows: readonly ProposedActionRow[] | null | undefined): number {
  if (!rows) return 0;
  return rows.filter(isPending).length;
}
