import { describe, it, expect } from 'vitest';
import {
  ALL_NAV,
  PRIMARY_NAV,
  HELP_NAV,
  DEFAULT_PATH,
  navItemForPath,
  navKeyForPath,
  isNavItemActive,
  breadcrumbsForPath,
  agentSearch,
  pendingCount,
} from './navigation';
import { presentStatus, toneBannerClasses, toneBadgeVariant } from './versionStatus';
import { runTone, outcomeTone, UNTRACKED_OPTIMIZERS } from './jobs';
import type { ProposedActionRow } from './approvals';

describe('IA config', () => {
  it('maps the seven task-grouped primary sections to routes', () => {
    expect(PRIMARY_NAV.map((n) => n.key)).toEqual([
      'overview',
      'compare',
      'optimize',
      'approvals',
      'labeling',
      'activity',
      'lineage',
    ]);
  });

  it('keeps Add agent + How it works as distinct help/action items (footer, not primary)', () => {
    expect(HELP_NAV.map((n) => n.key)).toEqual(['add-agent', 'how-it-works']);
    expect(PRIMARY_NAV.some((n) => n.key === 'add-agent' || n.key === 'how-it-works')).toBe(false);
  });

  it('every nav item has a unique route path', () => {
    const paths = ALL_NAV.map((n) => n.path);
    expect(new Set(paths).size).toBe(paths.length);
  });

  it('scopes the agent-specific sections and leaves cross-agent flows unscoped', () => {
    const requires = (k: string) => ALL_NAV.find((n) => n.key === k)!.requiresAgent;
    // Agent-scoped: overview / compare / optimize / approvals / labeling / lineage need a selected agent.
    expect([
      requires('overview'),
      requires('compare'),
      requires('optimize'),
      requires('approvals'),
      requires('labeling'),
      requires('lineage'),
    ]).toEqual([true, true, true, true, true, true]);
    // Cross-agent / help flows must NOT require an agent (Activity is workspace-wide;
    // Add agent is how you get your first one).
    expect([requires('activity'), requires('add-agent'), requires('how-it-works')]).toEqual([false, false, false]);
  });
});

describe('active-route / nav-state mapping', () => {
  it('resolves each route to its nav key', () => {
    expect(navKeyForPath('/overview')).toBe('overview');
    expect(navKeyForPath('/compare')).toBe('compare');
    expect(navKeyForPath('/optimize')).toBe('optimize');
    expect(navKeyForPath('/approvals')).toBe('approvals');
    expect(navKeyForPath('/labeling')).toBe('labeling');
    expect(navKeyForPath('/activity')).toBe('activity');
    expect(navKeyForPath('/lineage')).toBe('lineage');
    expect(navKeyForPath('/add-agent')).toBe('add-agent');
    expect(navKeyForPath('/how-it-works')).toBe('how-it-works');
  });

  it('treats "/" as Overview (default landing) and tolerates a trailing slash', () => {
    expect(navKeyForPath('/')).toBe('overview');
    expect(navKeyForPath('/compare/')).toBe('compare');
    expect(DEFAULT_PATH).toBe('/overview');
  });

  it('returns null for a genuinely unknown path', () => {
    expect(navKeyForPath('/nope')).toBeNull();
    expect(navItemForPath('/nope')).toBeNull();
  });

  it('isNavItemActive is exclusive — exactly one primary item is active per route', () => {
    const active = PRIMARY_NAV.filter((item) => isNavItemActive(item, '/approvals'));
    expect(active.map((n) => n.key)).toEqual(['approvals']);
  });
});

describe('breadcrumbs', () => {
  it('builds a root → section trail with only the root linkable', () => {
    const crumbs = breadcrumbsForPath('/compare');
    expect(crumbs).toHaveLength(2);
    expect(crumbs[0].path).toBe('/overview');
    expect(crumbs[crumbs.length - 1].path).toBeUndefined();
    expect(crumbs[crumbs.length - 1].label).toMatch(/version/i);
  });

  it('never renders empty, even for an unknown path', () => {
    expect(breadcrumbsForPath('/nope').length).toBeGreaterThan(0);
  });
});

describe('agentSearch (selection carried in the URL)', () => {
  it('encodes the selected agent as a query string', () => {
    expect(agentSearch('claude_code')).toBe('?agent=claude_code');
    expect(agentSearch('a b/c')).toBe('?agent=a%20b%2Fc');
  });

  it('is empty when no agent is selected', () => {
    expect(agentSearch(null)).toBe('');
    expect(agentSearch(undefined)).toBe('');
    expect(agentSearch('')).toBe('');
  });
});

describe('pendingCount (Approvals sidebar badge)', () => {
  const row = (status: string): ProposedActionRow => ({ status }) as ProposedActionRow;

  it('counts only pending proposals, reusing the queue’s isPending predicate', () => {
    expect(pendingCount([row('pending'), row('applied'), row('PENDING'), row('rejected')])).toBe(2);
  });

  it('is 0 for null/empty (no badge shown)', () => {
    expect(pendingCount(null)).toBe(0);
    expect(pendingCount([])).toBe(0);
  });
});

// Regression guard: re-homing the panels into the shell must NOT weaken their honesty.
// These assert the exact tone contracts the re-homed VersionComparison / ActivityJobs
// consume are intact (component-level rendering is additionally covered by
// versionStatus.test.ts and jobs.test.ts).
describe('honesty states survive re-homing', () => {
  it('controlled-proof is amber (caution), never a false green', () => {
    const s = presentStatus('controlled_proof_collecting');
    expect(s.tone).toBe('caution');
    expect(s.tone).not.toBe('positive');
    const banner = toneBannerClasses('caution');
    expect(banner).toMatch(/amber/);
    expect(banner).not.toMatch(/emerald|green/);
    expect(toneBadgeVariant('caution')).not.toBe('default'); // not the "proven" badge styling
  });

  it('a regressed candidate stays red/negative', () => {
    expect(presentStatus('regressed').tone).toBe('negative');
    expect(toneBannerClasses('negative')).toMatch(/destructive/);
  });

  it('only proven_improvement earns the green tone', () => {
    expect(presentStatus('proven_improvement').tone).toBe('positive');
    expect(toneBannerClasses('positive')).toMatch(/emerald/);
  });

  it('Activity run tone is fail-closed — an ambiguous/canceled state is never dressed up', () => {
    expect(runTone({ result_state: 'SUCCESS' })).toBe('success');
    expect(runTone({ result_state: 'FAILED' })).toBe('error');
    expect(runTone({ life_cycle_state: 'RUNNING' })).toBe('active');
    expect(runTone({ result_state: 'CANCELED' })).toBe('neutral');
    expect(runTone({})).toBe('neutral');
  });

  it('proposal outcomes: only applied is a success; rejected/unknown stay neutral', () => {
    expect(outcomeTone('applied')).toBe('success');
    expect(outcomeTone('rejected')).toBe('neutral');
    expect(outcomeTone('superseded')).toBe('neutral');
  });

  it('un-instrumented optimizers keep an explicit "not tracked" state (no fabricated runs)', () => {
    expect(UNTRACKED_OPTIMIZERS.length).toBeGreaterThan(0);
  });
});
