CREATE OR REPLACE VIEW `austin_choi_omni_agent_catalog`.`agent_improvement_loop`.`mv_cache_token_churn_dashboard_metric`
WITH METRICS
LANGUAGE YAML
AS $$
version: '1.1'
source: austin_choi_omni_agent_catalog.agent_improvement_loop.l0_session_metrics
comment: 'Auto-generated (Stage 6) from L3/RLM recommendation rank 3 (recurs across 3 trace(s)): Cache token churn dashboard metric. Token-efficiency / tool-call-redundancy metrics over l0_session_metrics.'
dimensions:
- name: Model
  expr: model
  comment: Model that produced the trace.
- name: Producer
  expr: producer
  comment: Agent runtime that produced the trace.
- name: Status
  expr: status
  comment: Trace terminal status (OK/ERROR/...).
- name: Request Time
  expr: request_time
  comment: Trace request time (ISO-8601).
measures:
- name: Trace Count
  expr: COUNT(1)
  comment: Number of traces (the denominator for per-trace efficiency measures).
- name: Total Tokens
  expr: SUM(total_tokens)
  comment: Total tokens consumed across traces.
- name: Tokens per Trace
  expr: SUM(total_tokens) / NULLIF(COUNT(1), 0)
  comment: Average tokens per trace — the headline tokens-per-task efficiency measure.
- name: Cache Tokens
  expr: SUM(cache_total_tokens)
  comment: Total cache read+write tokens — churn here signals re-ingested context.
- name: Redundant Tool Calls
  expr: SUM(redundant_tool_calls)
  comment: Total byte-identical repeated tool calls (re-reads / re-run boilerplate).
- name: Redundant Tool Call Rate
  expr: SUM(redundant_tool_calls) / NULLIF(SUM(total_tool_calls), 0)
  comment: Share of tool calls that were byte-identical repeats (re-aggregatable).
$$
