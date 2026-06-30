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

test('smoke test - leaderboard loads and renders sections + data', async ({ page }) => {
  await page.goto('/');

  // App + section headings (h1/h2).
  await expect(page.getByRole('heading', { name: 'Agent Self-Optimization' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Baseline vs new version' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Corpus summary' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Token heavy tail' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Tool-waste diagnosis' })).toBeVisible();

  // The Phase-B priority visual renders the REAL Phase-2 result once the
  // version_comparisons query resolves: the -35.4% token headline and the honest
  // (amber, NOT green) controlled-proof status. Doubles as the "data loaded" wait.
  await expect(page.getByText('Controlled proof', { exact: false })).toBeVisible({ timeout: 30000 });
  await expect(page.getByText('-35.4%', { exact: false }).first()).toBeVisible({ timeout: 30000 });

  // Corpus KPI cards render after the corpus_summary query resolves. 'Traces' is
  // a unique KPI label.
  await expect(page.getByText('Traces', { exact: true })).toBeVisible({ timeout: 30000 });
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
