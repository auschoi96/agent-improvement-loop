import { test, expect } from '@playwright/test';
import { writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

// Single-page L0 leaderboard. Assertions use role-based selectors for headings
// and exact text for unique content (Playwright strict mode).

let testArtifactsDir: string;
let consoleLogs: string[] = [];
let consoleErrors: string[] = [];
let pageErrors: string[] = [];
let failedRequests: string[] = [];

test('smoke test - overview loads and renders current sections', async ({ page }) => {
  await page.goto('/');

  // Current app shell + overview headings.
  await expect(page.getByRole('heading', { name: 'Overview' })).toBeVisible();
  await expect(page.getByText('Deterministic L0 leaderboard for the selected agent')).toBeVisible();
  await expect(page.getByText('Token heavy tail', { exact: true })).toBeVisible();
  await expect(page.getByText('Top sessions by total tokens', { exact: true })).toBeVisible();

  const populated = page.getByText('Traces', { exact: true });
  const empty = page.getByText('No corpus data.', { exact: true });
  await expect(populated.or(empty)).toBeVisible({ timeout: 30_000 });
  if (await populated.isVisible()) {
    await expect(page.getByText('Total Tokens', { exact: true }).first()).toBeVisible();
    await expect(page.getByText('Tool Calls', { exact: true }).first()).toBeVisible();
  }
});

test('smoke test - quick connect is the default add-agent path', async ({ page }) => {
  await page.goto('/add-agent');

  await expect(page.getByText('Quick connect', { exact: true })).toBeVisible();
  await expect(page.getByText('Instrument any Python agent, HTTP wrapper, or LLM call in minutes.')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Copy configured starter code' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Use advanced setup' })).toBeVisible();

  await page.getByRole('button', { name: 'Use advanced setup' }).click();
  await expect(page.locator('[data-slot="card-title"]').filter({ hasText: 'Add an agent' })).toBeVisible();
});

test('background agent polling keeps the active overview mounted', async ({ page }) => {
  let agentQueryRequests = 0;
  page.on('request', (request) => {
    if (request.url().includes('/api/analytics/query/agents')) agentQueryRequests += 1;
  });

  // Accelerate only AgentProvider's exact 30-second interval. Other app timers keep
  // their production behavior, while this regression crosses multiple agent polls.
  await page.addInitScript(() => {
    const nativeSetInterval = window.setInterval.bind(window);
    window.setInterval = ((handler, timeout, ...args) =>
      nativeSetInterval(handler, timeout === 30_000 ? 250 : timeout, ...args)) as typeof window.setInterval;
  });

  await page.goto('/overview?agent=claude_code');
  const toolWasteTab = page.getByRole('tab', { name: 'Tool waste' });
  await expect(toolWasteTab).toBeVisible();
  await toolWasteTab.click();
  await expect(toolWasteTab).toHaveAttribute('aria-selected', 'true');

  const tabList = page.getByRole('tablist');
  const originalTabList = await tabList.elementHandle();
  expect(originalTabList).not.toBeNull();

  await expect.poll(() => agentQueryRequests).toBeGreaterThanOrEqual(2);

  await expect(toolWasteTab).toHaveAttribute('aria-selected', 'true');
  expect(await originalTabList!.evaluate((node) => node.isConnected)).toBe(true);
});

test('scheduled refresh retains visible overview data and tab state for 65 seconds', async ({ page }) => {
  test.setTimeout(90_000);
  await page.goto('/overview?agent=claude_code');

  const tracesLabel = page.getByText('Traces', { exact: true });
  await expect(tracesLabel).toBeVisible({ timeout: 30_000 });
  const tracesNode = await tracesLabel.elementHandle();
  expect(tracesNode).not.toBeNull();

  const toolWasteTab = page.getByRole('tab', { name: 'Tool waste' });
  await toolWasteTab.click();
  await expect(toolWasteTab).toHaveAttribute('aria-selected', 'true');

  // Cross the production boundary's 60-second first refresh and leave enough
  // time for the replacement queries to settle. Retained data and local tab state
  // must stay mounted throughout the refresh cycle.
  await page.waitForTimeout(65_000);

  expect(await tracesNode!.evaluate((node) => node.isConnected)).toBe(true);
  await expect(tracesLabel).toBeVisible();
  await expect(toolWasteTab).toHaveAttribute('aria-selected', 'true');
});

test('switching the agent experiment replaces every experiment-scoped view', async ({ page }) => {
  test.setTimeout(180_000);
  await page.setExtraHTTPHeaders({ 'x-forwarded-email': 'smoke-test@databricks.com' });
  const legacyExperiment = '660599403165942';
  const isolatedExperiment = '1301765275062543';
  let selectedExperiment = isolatedExperiment;

  await page.route('**/api/analytics/query/agents', async (route) => {
    const body = `data: ${JSON.stringify({
      type: 'result',
      data: [
        {
          agent_name: 'claude_code',
          experiment_id: selectedExperiment,
          reviewer_experiment_id: '1301765275062544',
          description: 'Claude Code CLI sessions',
        },
      ],
    })}\n\n`;
    await route.fulfill({ status: 200, contentType: 'text/event-stream', body });
  });

  await page.route('**/api/analytics/query/version_comparisons', async (route) => {
    const data =
      selectedExperiment === legacyExperiment
        ? [
            {
              baseline_version: 'v0-baseline-no-skill',
              candidate_version: 'v1-token-efficiency-skill',
              objective_metric: 'total_tokens',
              status: 'READY_TO_PROVE',
              readiness_tier: 'READY_TO_PROVE',
              can_prove_improvement: true,
              trace_count: 100,
              frozen_suite_present: true,
              n_promote: 1,
              n_block: 0,
              n_errored: 0,
              correctness_held: true,
              proof_source: 'frozen_suite',
              headline_metric: 'total_tokens',
              headline_baseline: 100,
              headline_candidate: 80,
              headline_delta_pct: -0.2,
              headline_improved: true,
              reasons: '',
            },
          ]
        : [];
    await route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body: `data: ${JSON.stringify({ type: 'result', data })}\n\n`,
    });
  });

  await page.addInitScript(() => {
    const nativeSetInterval = window.setInterval.bind(window);
    window.setInterval = ((handler, timeout, ...args) => {
      if (timeout === 30_000) {
        (window as Window & { runAgentPoll?: () => void }).runAgentPoll = () => {
          if (typeof handler === 'function') handler(...args);
        };
        return nativeSetInterval(() => undefined, 2_147_483_647);
      }
      return nativeSetInterval(handler, timeout, ...args);
    }) as typeof window.setInterval;
  });

  const pollAgents = async () => {
    await page.evaluate(() => (window as Window & { runAgentPoll?: () => void }).runAgentPoll?.());
  };

  await page.goto('/overview?agent=claude_code');
  const tracesKpi = page.getByText('Traces', { exact: true }).locator('..');
  await expect(tracesKpi.getByText('122', { exact: true })).toBeVisible();

  selectedExperiment = legacyExperiment;
  await pollAgents();
  await expect(tracesKpi.getByText('264', { exact: true })).toBeVisible();

  await page.goto('/compare?agent=claude_code');
  await expect(page.getByText('v0-baseline-no-skill → v1-token-efficiency-skill')).toBeVisible();

  selectedExperiment = isolatedExperiment;
  await pollAgents();
  await expect(page.getByText(/No version comparison published/)).toBeVisible();

  await page.goto('/labeling?agent=claude_code');
  await expect(page.getByText(/correctness judge/).first()).toBeVisible({ timeout: 60_000 });
  await expect(page.getByText(/modularity judge/).first()).toBeVisible();
  await expect(page.getByText(/groundedness judge/).first()).toBeVisible();
  await expect(page.getByText(/token_efficiency judge/).first()).toBeVisible();
  const isolatedTrace = page.getByText(/trace:.*mlflow_traces\.claude_code\//).first();
  await expect(isolatedTrace).toBeVisible();
  const isolatedTraceNode = await isolatedTrace.elementHandle();
  expect(isolatedTraceNode).not.toBeNull();

  selectedExperiment = legacyExperiment;
  await pollAgents();
  await expect(page.getByText(/trace:.*mlflow_traces\.cc\//).first()).toBeVisible({ timeout: 60_000 });
  expect(await isolatedTraceNode!.evaluate((node) => node.isConnected)).toBe(false);

  selectedExperiment = isolatedExperiment;
  await pollAgents();
  await page.goto('/approvals?agent=claude_code');
  await expect(page.getByText('Skill update', { exact: true })).toBeVisible();

  selectedExperiment = legacyExperiment;
  await pollAgents();
  await expect(page.getByText(/No pending proposals/)).toBeVisible();
});

// ── Lifecycle hooks (artifact capture; unchanged from template) ──────────────

test.beforeEach(async ({ page }) => {
  consoleLogs = [];
  consoleErrors = [];
  pageErrors = [];
  failedRequests = [];
  await page.setExtraHTTPHeaders({ 'x-forwarded-email': 'smoke-test@databricks.com' });

  testArtifactsDir = join(process.cwd(), '.smoke-test');
  mkdirSync(testArtifactsDir, { recursive: true });

  page.on('console', (msg) => {
    const type = msg.type();
    const text = msg.text();
    if (!text.trim() || /^%[osd]$/.test(text.trim())) {
      return;
    }
    const location = msg.location();
    const locationStr = location.url ? ` at ${location.url}:${location.lineNumber}:${location.columnNumber}` : '';
    consoleLogs.push(`[${type}] ${text}${locationStr}`);
    if (type === 'error') {
      consoleErrors.push(`${text}${locationStr}`);
    }
  });

  page.on('pageerror', (error) => {
    const errorDetails = `Page error: ${error.message}\nStack: ${error.stack || 'No stack trace available'}`;
    pageErrors.push(errorDetails);
    console.error('Page error detected:', errorDetails);
  });

  page.on('requestfailed', (request) => {
    failedRequests.push(`Failed request: ${request.url()} - ${request.failure()?.errorText}`);
  });
});

test.afterEach(async ({ page }, testInfo) => {
  const testName = testInfo.title.replace(/ /g, '-').toLowerCase();
  const screenshotPath = join(testArtifactsDir, `${testName}-app-screenshot.png`);
  await page.screenshot({ path: screenshotPath, fullPage: true });

  const logsPath = join(testArtifactsDir, `${testName}-console-logs.txt`);
  const allLogs = [
    '=== Console Logs ===',
    ...consoleLogs,
    '\n=== Console Errors (React errors) ===',
    ...consoleErrors,
    '\n=== Page Errors ===',
    ...pageErrors,
    '\n=== Failed Requests ===',
    ...failedRequests,
  ];
  writeFileSync(logsPath, allLogs.join('\n'), 'utf-8');

  console.log(`Screenshot saved to: ${screenshotPath}`);
  console.log(`Console logs saved to: ${logsPath}`);
  if (consoleErrors.length > 0) {
    console.log('Console errors detected:', consoleErrors);
  }
  if (pageErrors.length > 0) {
    console.log('Page errors detected:', pageErrors);
  }
  if (failedRequests.length > 0) {
    console.log('Failed requests detected:', failedRequests);
  }

  await page.close();
});
