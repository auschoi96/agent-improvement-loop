import { useMemo, useState } from 'react';
import {
  Badge,
  Button,
  Input,
  Skeleton,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  type QueryRegistry,
} from '@databricks/appkit-ui/react';
import { RefreshableAnalyticsQuery, type AppQueryKey } from './RefreshableAnalyticsQuery';

type TableQueryKey = {
  [K in AppQueryKey]: QueryRegistry[K]['result'] extends Array<Record<string, unknown>> ? K : never;
}[AppQueryKey];
type QueryRows<K extends TableQueryKey> = QueryRegistry[K]['result'] & Array<Record<string, unknown>>;
type QueryRow<K extends TableQueryKey> = QueryRows<K>[number];

function display(value: unknown): string {
  if (value == null) return '—';
  if (typeof value === 'boolean') return value ? 'yes' : 'no';
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'bigint') return String(value);
  if (value instanceof Date) return value.toISOString();
  try {
    return JSON.stringify(value) ?? '—';
  } catch {
    return 'unavailable';
  }
}

function StableRowsTable<Row extends Record<string, unknown>>({
  rows,
  loading,
  refreshing,
  error,
  filterColumn,
  filterPlaceholder,
  pageSize,
}: {
  rows: Row[];
  loading: boolean;
  refreshing: boolean;
  error: string | null;
  filterColumn?: Extract<keyof Row, string>;
  filterPlaceholder?: string;
  pageSize: number;
}) {
  const [filter, setFilter] = useState('');
  const [page, setPage] = useState(0);
  const columns = useMemo(() => Object.keys(rows[0] ?? {}) as Array<Extract<keyof Row, string>>, [rows]);
  const filtered = useMemo(() => {
    if (!filterColumn || !filter.trim()) return rows;
    const needle = filter.trim().toLowerCase();
    return rows.filter((row) => display(row[filterColumn]).toLowerCase().includes(needle));
  }, [filter, filterColumn, rows]);
  const pageCount = Math.max(1, Math.ceil(filtered.length / pageSize));

  const currentPage = Math.min(page, pageCount - 1);
  const visible = filtered.slice(currentPage * pageSize, (currentPage + 1) * pageSize);
  const duplicateCounts = new Map<string, number>();
  const keyedVisible = visible.map((row) => {
    const serialized = JSON.stringify(row);
    const occurrence = duplicateCounts.get(serialized) ?? 0;
    duplicateCounts.set(serialized, occurrence + 1);
    return { row, key: `${serialized}#${occurrence}` };
  });

  if (loading) return <Skeleton className="h-56 w-full" />;
  if (error && rows.length === 0) return <p className="text-sm text-destructive">Error: {error}</p>;
  if (rows.length === 0) return <p className="text-sm text-muted-foreground">No rows found.</p>;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        {filterColumn ? (
          <Input
            value={filter}
            onChange={(event) => {
              setFilter(event.target.value);
              setPage(0);
            }}
            placeholder={filterPlaceholder ?? `Filter by ${filterColumn}…`}
            className="max-w-xs"
          />
        ) : (
          <span />
        )}
        {refreshing && <Badge variant="outline">Refreshing…</Badge>}
        {error && rows.length > 0 && <Badge variant="outline">Refresh failed; showing prior data</Badge>}
      </div>
      <div className="overflow-x-auto rounded-md border">
        <Table>
          <TableHeader>
            <TableRow>
              {columns.map((column) => (
                <TableHead key={column}>{column.replaceAll('_', ' ')}</TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {keyedVisible.map(({ row, key }) => (
              <TableRow key={key}>
                {columns.map((column) => (
                  <TableCell key={column}>{display(row[column])}</TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
      {pageCount > 1 && (
        <div className="flex items-center justify-end gap-2 text-sm text-muted-foreground">
          <Button variant="outline" size="sm" disabled={currentPage === 0} onClick={() => setPage(currentPage - 1)}>
            Previous
          </Button>
          <span>
            Page {currentPage + 1} of {pageCount}
          </span>
          <Button
            variant="outline"
            size="sm"
            disabled={currentPage + 1 >= pageCount}
            onClick={() => setPage(currentPage + 1)}
          >
            Next
          </Button>
        </div>
      )}
    </div>
  );
}

export function StableDataTable<K extends TableQueryKey>({
  queryKey,
  parameters,
  filterColumn,
  filterPlaceholder,
  pageSize = 10,
}: {
  queryKey: K;
  parameters: QueryRegistry[K]['parameters'];
  filterColumn?: Extract<keyof QueryRow<K>, string>;
  filterPlaceholder?: string;
  pageSize?: number;
}) {
  return (
    <RefreshableAnalyticsQuery queryKey={queryKey} parameters={parameters}>
      {({ data, loading, refreshing, error }) => (
        <StableRowsTable
          rows={(data ?? []) as QueryRows<K>}
          loading={loading}
          refreshing={refreshing}
          error={error}
          filterColumn={filterColumn}
          filterPlaceholder={filterPlaceholder}
          pageSize={pageSize}
        />
      )}
    </RefreshableAnalyticsQuery>
  );
}
