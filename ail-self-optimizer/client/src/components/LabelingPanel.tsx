import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Card,
  CardContent,
  Badge,
  Button,
  Input,
  Textarea,
  Label,
  Progress,
  RadioGroup,
  RadioGroupItem,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  Skeleton,
} from '@databricks/appkit-ui/react';
import {
  DIMENSIONS_ENDPOINT,
  LABEL_ENDPOINT,
  buildLabelRequest,
  httpErrorMessage,
  labelMessage,
  missingDimensions,
  progressLabel,
  progressRatio,
  type DimensionProgress,
  type DimensionsResponse,
  type LabelInput,
  type LabelResponse,
  type LabelTone,
  type TraceTarget,
} from '../lib/labeling';

const TONE_CLASS: Record<LabelTone, string> = {
  success: 'text-emerald-700 dark:text-emerald-300',
  warning: 'text-amber-700 dark:text-amber-300',
  error: 'text-destructive',
};

// The in-app labeling page (L4). It reads the experiment's REGISTERED judged
// dimensions and each one's label progress toward the readiness floor (all from the
// Python engine — the floor is never hardcoded here), then lets a signed-in user
// grade a trace along one of those dimensions. A label POSTs to the authenticated
// route, which reuses ail.judges.labeling.record_label to write a HUMAN assessment
// named for the judge — the name-match L2's scheduled auto-align pairs by. After a
// write the view remounts (via `reloadKey`) so progress refetches.
export function LabelingPanel({ agentName, experimentId }: { agentName: string; experimentId: string }) {
  const [reloadKey, setReloadKey] = useState(0);
  return (
    <DimensionsView
      key={reloadKey}
      agentName={agentName}
      experimentId={experimentId}
      onLabeled={() => setReloadKey((k) => k + 1)}
    />
  );
}

function DimensionsView({
  agentName,
  experimentId,
  onLabeled,
}: {
  agentName: string;
  experimentId: string;
  onLabeled: () => void;
}) {
  const [state, setState] = useState<{ loading: boolean; error: string | null; data: DimensionsResponse | null }>({
    loading: true,
    error: null,
    data: null,
  });

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();
    async function load() {
      try {
        const res = await fetch(DIMENSIONS_ENDPOINT, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ experiment_id: experimentId }),
          signal: controller.signal,
        });
        const body = (await res.json().catch(() => ({}))) as DimensionsResponse;
        if (cancelled) return;
        if (!res.ok && !body.outcome) {
          setState({ loading: false, error: httpErrorMessage(res.status), data: null });
          return;
        }
        if (body.outcome !== 'dimensions') {
          // The engine failed closed (e.g. it could not determine the registered judges);
          // surface the honest reason rather than inventing dimensions.
          setState({
            loading: false,
            error: body.error ?? body.refused_reason ?? 'The labeling engine returned no dimensions.',
            data: null,
          });
          return;
        }
        setState({ loading: false, error: null, data: body });
      } catch (err) {
        if (!cancelled) {
          setState({ loading: false, error: err instanceof Error ? err.message : 'Network error.', data: null });
        }
      }
    }
    void load();
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [experimentId]);

  if (state.loading) {
    return (
      <div className="space-y-3">
        <Skeleton className="h-20 w-full" />
        <Skeleton className="h-40 w-full" />
      </div>
    );
  }
  if (state.error) {
    return <div className="text-destructive bg-destructive/10 p-3 rounded-md">Error: {state.error}</div>;
  }

  const data = state.data as DimensionsResponse;
  const dimensions = data.dimensions ?? [];
  const traces = data.traces ?? [];
  const order = dimensions.map((d) => d.name);

  if (dimensions.length === 0) {
    return (
      <div className="text-muted-foreground border rounded-md p-4">
        No registered judges for <span className="font-mono">{agentName}</span> yet. Author a judge (ail.judges) so its
        dimension appears here to label.
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {data.summary && <p className="text-sm text-muted-foreground max-w-3xl">{data.summary}</p>}

      <div className="grid gap-3 md:grid-cols-2">
        {dimensions.map((dim) => (
          <ProgressCard key={dim.name} dim={dim} />
        ))}
      </div>

      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-muted-foreground">Traces needing a label</h3>
          {typeof data.scanned === 'number' && (
            <span className="text-xs text-muted-foreground tabular-nums">
              scanned {data.scanned}
              {data.scan_capped ? ' (most recent)' : ''}
            </span>
          )}
        </div>
        {traces.length === 0 ? (
          <div className="text-muted-foreground border rounded-md p-4">
            Every scanned trace is labeled on all dimensions — nothing left in this batch.
          </div>
        ) : (
          <ol className="space-y-3">
            {traces.map((trace) => (
              <li key={trace.trace_id}>
                <TraceRow
                  trace={trace}
                  dimensions={dimensions}
                  order={order}
                  experimentId={experimentId}
                  onLabeled={onLabeled}
                />
              </li>
            ))}
          </ol>
        )}
      </div>
    </div>
  );
}

function ProgressCard({ dim }: { dim: DimensionProgress }) {
  const ratio = progressRatio(dim);
  return (
    <Card className="shadow-sm">
      <CardContent className="p-4 space-y-2">
        <div className="flex items-center justify-between gap-2">
          <span className="font-mono text-sm font-semibold">{dim.name}</span>
          <Badge variant={dim.complete ? 'default' : 'outline'} className="tabular-nums">
            {progressLabel(dim)}
          </Badge>
        </div>
        <Progress value={ratio === null ? 0 : ratio * 100} />
        {/* Rendered VERBATIM from the Python engine — the floor number is never authored here. */}
        <p className="text-xs text-muted-foreground">{dim.summary}</p>
      </CardContent>
    </Card>
  );
}

function TraceRow({
  trace,
  dimensions,
  order,
  experimentId,
  onLabeled,
}: {
  trace: TraceTarget;
  dimensions: DimensionProgress[];
  order: string[];
  experimentId: string;
  onLabeled: () => void;
}) {
  const missing = useMemo(() => missingDimensions(trace, order), [trace, order]);
  const [selected, setSelected] = useState<string>(missing[0] ?? '');
  const [value, setValue] = useState<unknown>('');
  const [rationale, setRationale] = useState('');
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<{ tone: LabelTone; text: string } | null>(null);
  const submitController = useRef<AbortController | null>(null);
  const reloadTimer = useRef<number | null>(null);

  useEffect(
    () => () => {
      submitController.current?.abort();
      if (reloadTimer.current !== null) window.clearTimeout(reloadTimer.current);
    },
    []
  );

  const dim = dimensions.find((d) => d.name === selected);

  // Reset the entered value when the selected dimension changes (its control differs).
  const onSelectDimension = useCallback((name: string) => {
    setSelected(name);
    setValue('');
    setMessage(null);
  }, []);

  async function submit() {
    setMessage(null);
    let request;
    try {
      request = buildLabelRequest(
        { experiment_id: experimentId, trace_id: trace.trace_id },
        selected,
        value,
        rationale
      );
    } catch (err) {
      setMessage({ tone: 'error', text: err instanceof Error ? err.message : String(err) });
      return;
    }
    submitController.current?.abort();
    const controller = new AbortController();
    submitController.current = controller;
    setBusy(true);
    try {
      const res = await fetch(LABEL_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
        signal: controller.signal,
      });
      const body = (await res.json().catch(() => ({}))) as LabelResponse;
      if (!res.ok && !body.outcome) {
        setMessage({ tone: 'error', text: httpErrorMessage(res.status) });
        return;
      }
      const msg = labelMessage(body);
      setMessage(msg);
      if (body.outcome === 'labeled') {
        // Give the reader a beat to see the confirmation, then refetch progress.
        reloadTimer.current = window.setTimeout(onLabeled, 600);
      }
    } catch (err) {
      if (controller.signal.aborted) return;
      setMessage({ tone: 'error', text: err instanceof Error ? err.message : 'Network error.' });
    } finally {
      if (!controller.signal.aborted) setBusy(false);
      if (submitController.current === controller) submitController.current = null;
    }
  }

  return (
    <Card className="shadow-sm">
      <CardContent className="p-4 space-y-3">
        <div className="flex flex-wrap items-center gap-2">
          <span className="font-mono text-xs text-muted-foreground">{trace.trace_id}</span>
          {trace.request_time && <span className="text-xs text-muted-foreground">· {trace.request_time}</span>}
          <span className="ml-auto flex flex-wrap gap-1">
            {order.map((name) => (
              <Badge key={name} variant={trace.labeled[name] ? 'default' : 'outline'} className="text-xs">
                {trace.labeled[name] ? '✓ ' : ''}
                {name}
              </Badge>
            ))}
          </span>
        </div>

        {trace.preview && (
          <pre className="max-h-24 overflow-auto rounded-md bg-muted p-2 text-xs whitespace-pre-wrap">
            {trace.preview}
          </pre>
        )}

        <div className="flex flex-wrap items-end gap-3">
          <div className="space-y-1">
            <Label className="text-xs">Dimension</Label>
            <Select value={selected} onValueChange={onSelectDimension}>
              <SelectTrigger className="w-48">
                <SelectValue placeholder="Pick a dimension" />
              </SelectTrigger>
              <SelectContent>
                {missing.map((name) => (
                  <SelectItem key={name} value={name}>
                    {name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-1">
            <Label className="text-xs">Value</Label>
            <ValueControl input={dim?.input ?? null} value={value} onChange={setValue} />
          </div>
        </div>

        <Textarea
          value={rationale}
          onChange={(e) => setRationale(e.target.value)}
          placeholder="Rationale — name the specific evidence in the trace (optional but recommended)…"
          rows={2}
        />

        <div className="flex flex-wrap items-center gap-2">
          <Button onClick={() => void submit()} disabled={busy || !selected}>
            {busy ? 'Saving…' : 'Save label'}
          </Button>
          {message && <span className={`text-sm ${TONE_CLASS[message.tone]}`}>{message.text}</span>}
        </div>
      </CardContent>
    </Card>
  );
}

// The value control implied by the dimension's L1 label schema (a UI hint from the
// engine). Numeric → a small select over the schema's range; pass/fail → a radio over
// the schema's labels; anything else / unknown → a free-form field. Never invents a
// scale — an unknown schema falls back to free text.
function ValueControl({
  input,
  value,
  onChange,
}: {
  input: LabelInput | null;
  value: unknown;
  onChange: (v: unknown) => void;
}) {
  if (input?.kind === 'numeric') {
    const min = Math.round(input.min ?? 1);
    const max = Math.round(input.max ?? 5);
    if (max - min >= 0 && max - min <= 10) {
      const options = Array.from({ length: max - min + 1 }, (_, i) => min + i);
      return (
        <Select value={value === '' ? '' : String(value)} onValueChange={(v) => onChange(Number(v))}>
          <SelectTrigger className="w-32">
            <SelectValue placeholder={`${min}–${max}`} />
          </SelectTrigger>
          <SelectContent>
            {options.map((n) => (
              <SelectItem key={n} value={String(n)}>
                {n}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      );
    }
    return (
      <Input
        type="number"
        className="w-32"
        value={value === '' ? '' : String(value)}
        min={min}
        max={max}
        onChange={(e) => onChange(e.target.value === '' ? '' : Number(e.target.value))}
      />
    );
  }

  if (input?.kind === 'pass_fail') {
    const positive = input.positive ?? 'pass';
    const negative = input.negative ?? 'fail';
    return (
      <RadioGroup
        className="flex items-center gap-4"
        value={typeof value === 'string' ? value : ''}
        onValueChange={onChange}
      >
        {[positive, negative].map((opt) => (
          <div key={opt} className="flex items-center gap-1.5">
            <RadioGroupItem value={opt} id={`pf-${opt}`} />
            <Label htmlFor={`pf-${opt}`} className="text-sm">
              {opt}
            </Label>
          </div>
        ))}
      </RadioGroup>
    );
  }

  return (
    <Input
      className="w-48"
      value={typeof value === 'string' ? value : ''}
      onChange={(e) => onChange(e.target.value)}
      placeholder="value"
    />
  );
}
