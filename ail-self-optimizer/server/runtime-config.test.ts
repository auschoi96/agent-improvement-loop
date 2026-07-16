import { describe, expect, it } from 'vitest';
import { loadRuntimeConfig } from './runtime-config';

const VALID = {
  DATABRICKS_WAREHOUSE_ID: 'warehouse-1',
  AIL_CATALOG: 'catalog_name',
  AIL_SCHEMA: 'agent_improvement_loop',
  AIL_TRACE_CATALOG: 'catalog_name',
  AIL_TRACE_SCHEMA: 'mlflow_traces',
  AIL_APPLY_JOB_ID: '123',
  AIL_ONBOARDING_JOB_ID: '456',
  AIL_LABEL_FLOOR: '20',
};

describe('loadRuntimeConfig', () => {
  it('accepts a fully resolved production configuration', () => {
    expect(loadRuntimeConfig(VALID)).toMatchObject({
      catalog: 'catalog_name',
      schema: 'agent_improvement_loop',
      labelFloor: 20,
    });
  });

  it('fails startup when app.yaml leaked a bundle template token', () => {
    expect(() => loadRuntimeConfig({ ...VALID, AIL_TRACE_CATALOG: '${var.catalog}' })).toThrow(
      'unresolved bundle variable'
    );
  });

  it('rejects invalid identifiers and job ids', () => {
    expect(() => loadRuntimeConfig({ ...VALID, AIL_SCHEMA: 'schema.with.dot' })).toThrow('Unity Catalog identifier');
    expect(() => loadRuntimeConfig({ ...VALID, AIL_APPLY_JOB_ID: 'not-a-job' })).toThrow('numeric Databricks job id');
  });
});
