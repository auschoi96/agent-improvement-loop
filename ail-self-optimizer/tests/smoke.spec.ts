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

// ── Lifecycle hooks (artifact capture; unchanged from template) ──────────────

test.beforeEach(async ({ page }) => {
  consoleLogs = [];
  consoleErrors = [];
  pageErrors = [];
  failedRequests = [];

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
