const ANNOTATIONS_SUFFIX = '_otel_annotations';

/** Derive the sibling MLflow OTEL spans table from the persisted annotations table. */
export function spansTableFromAnnotations(annotationsTable: string | null | undefined): string | null {
  const table = annotationsTable?.trim() ?? '';
  if (!table.endsWith(ANNOTATIONS_SUFFIX)) return null;
  const prefix = table.slice(0, -ANNOTATIONS_SUFFIX.length);
  return prefix ? `${prefix}_otel_spans` : null;
}

export type TraceFreshness =
  | { state: 'current'; pending: 0 }
  | { state: 'pending'; pending: number }
  | { state: 'source_behind'; pending: 0 };

/** Compare the live OTEL population with the atomic L0 snapshot population. */
export function traceFreshness(liveCount: number, snapshotCount: number): TraceFreshness {
  if (liveCount > snapshotCount) return { state: 'pending', pending: liveCount - snapshotCount };
  if (liveCount < snapshotCount) return { state: 'source_behind', pending: 0 };
  return { state: 'current', pending: 0 };
}
