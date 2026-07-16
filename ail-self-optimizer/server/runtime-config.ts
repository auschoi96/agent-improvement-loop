const UNRESOLVED_BUNDLE_VALUE = /\$\{var\.[^}]+\}/;
const UC_IDENTIFIER = /^[A-Za-z0-9_-]+$/;

export interface RuntimeConfig {
  warehouseId: string;
  catalog: string;
  schema: string;
  traceCatalog: string;
  traceSchema: string;
  applyJobId: string;
  onboardingJobId: string;
  labelFloor: number;
}

function required(env: NodeJS.ProcessEnv, name: string): string {
  const value = env[name]?.trim() ?? '';
  if (!value) throw new Error(`${name} is required`);
  if (UNRESOLVED_BUNDLE_VALUE.test(value)) {
    throw new Error(`${name} contains an unresolved bundle variable: ${value}`);
  }
  return value;
}

function identifier(env: NodeJS.ProcessEnv, name: string): string {
  const value = required(env, name);
  if (!UC_IDENTIFIER.test(value)) {
    throw new Error(`${name} must be a single Unity Catalog identifier`);
  }
  return value;
}

function jobId(env: NodeJS.ProcessEnv, name: string): string {
  const value = required(env, name);
  if (!/^\d+$/.test(value)) throw new Error(`${name} must be a numeric Databricks job id`);
  return value;
}

export function loadRuntimeConfig(env: NodeJS.ProcessEnv = process.env): RuntimeConfig {
  const rawLabelFloor = required(env, 'AIL_LABEL_FLOOR');
  const labelFloor = Number(rawLabelFloor);
  if (!Number.isSafeInteger(labelFloor) || labelFloor <= 0) {
    throw new Error('AIL_LABEL_FLOOR must be a positive integer');
  }

  return {
    warehouseId: required(env, 'DATABRICKS_WAREHOUSE_ID'),
    catalog: identifier(env, 'AIL_CATALOG'),
    schema: identifier(env, 'AIL_SCHEMA'),
    traceCatalog: identifier(env, 'AIL_TRACE_CATALOG'),
    traceSchema: identifier(env, 'AIL_TRACE_SCHEMA'),
    applyJobId: jobId(env, 'AIL_APPLY_JOB_ID'),
    onboardingJobId: jobId(env, 'AIL_ONBOARDING_JOB_ID'),
    labelFloor,
  };
}
