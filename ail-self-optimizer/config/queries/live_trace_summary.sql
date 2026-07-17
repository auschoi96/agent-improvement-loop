-- Live trace freshness from the MLflow-managed OTEL spans Delta table.
--
-- The table is selected from the registry-provided *_otel_annotations mapping
-- after the client derives the sibling *_otel_spans name. IDENTIFIER keeps the
-- table reference parameterized instead of interpolating SQL. AppKit 0.38 uses
-- an empty string while describing parameterized queries, so the CASE supplies
-- a describe-only fallback; runtime binds the selected agent's actual table.
-- @param otel_spans_table STRING
SELECT
  COUNT(DISTINCT trace_id) AS live_trace_count,
  MAX(timestamp_micros(CAST(end_time_unix_nano / 1000 AS BIGINT))) AS latest_trace_end
FROM IDENTIFIER(
  CASE
    WHEN :otel_spans_table = ''
      THEN 'austin_choi_omni_agent_catalog.mlflow_traces.4408383386333204_otel_spans'
    ELSE :otel_spans_table
  END
)
WHERE parent_span_id IS NULL
  AND end_time_unix_nano IS NOT NULL
