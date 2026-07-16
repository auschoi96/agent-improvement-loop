import { readFileSync } from 'node:fs';
import { defineConfig, devices } from '@playwright/test';

const goalCatalog = JSON.parse(
  readFileSync(new URL('./server/plugins/onboarding/goal-catalog.json', import.meta.url), 'utf8')
) as { thresholds: { quality_min_labels: number } };

export default defineConfig({
  testDir: './tests',
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: 1,
  reporter: 'html',
  expect: {
    timeout: 15_000,
  },
  use: {
    baseURL: `http://localhost:${process.env.DATABRICKS_APP_PORT || process.env.PORT || 8000}`,
    trace: 'on-first-retry',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
  webServer: {
    command: 'npm run dev',
    url: `http://localhost:${process.env.DATABRICKS_APP_PORT || process.env.PORT || 8000}`,
    env: {
      DATABRICKS_CONFIG_PROFILE: process.env.DATABRICKS_CONFIG_PROFILE || 'dais-demo',
      DATABRICKS_WAREHOUSE_ID: process.env.DATABRICKS_WAREHOUSE_ID || '7d1d3dbb3ba65f2a',
      AIL_CATALOG: process.env.AIL_CATALOG || 'austin_choi_omni_agent_catalog',
      AIL_SCHEMA: process.env.AIL_SCHEMA || 'agent_improvement_loop',
      AIL_TRACE_CATALOG: process.env.AIL_TRACE_CATALOG || 'austin_choi_omni_agent_catalog',
      AIL_TRACE_SCHEMA: process.env.AIL_TRACE_SCHEMA || 'mlflow_traces',
      AIL_APPLY_JOB_ID: process.env.AIL_APPLY_JOB_ID || '372176302889362',
      AIL_ONBOARDING_JOB_ID: process.env.AIL_ONBOARDING_JOB_ID || '110543773338252',
      AIL_ONBOARDING_TRANSPORT: 'job',
      AIL_LABELING_TRANSPORT: 'rest',
      AIL_LABEL_FLOOR: process.env.AIL_LABEL_FLOOR || String(goalCatalog.thresholds.quality_min_labels),
    },
    reuseExistingServer: !process.env.CI,
    timeout: 120 * 1000,
  },
});
