import { describe, expect, it } from 'vitest';
import { spansTableFromAnnotations, traceFreshness } from './traces';

describe('spansTableFromAnnotations', () => {
  it('derives the managed sibling table without changing numeric prefixes', () => {
    expect(
      spansTableFromAnnotations('austin_choi_omni_agent_catalog.mlflow_traces.4408383386333204_otel_annotations')
    ).toBe('austin_choi_omni_agent_catalog.mlflow_traces.4408383386333204_otel_spans');
  });

  it('fails closed when the registry value is blank or not an annotations table', () => {
    expect(spansTableFromAnnotations('')).toBeNull();
    expect(spansTableFromAnnotations('catalog.schema.some_table')).toBeNull();
  });
});

describe('traceFreshness', () => {
  it('reports traces awaiting the L0 publisher', () => {
    expect(traceFreshness(20, 15)).toEqual({ state: 'pending', pending: 5 });
  });

  it('distinguishes a current snapshot from a lagging live source', () => {
    expect(traceFreshness(15, 15)).toEqual({ state: 'current', pending: 0 });
    expect(traceFreshness(14, 15)).toEqual({ state: 'source_behind', pending: 0 });
  });
});
