import { describe, expect, it, vi } from 'vitest';
import {
  dispatchGepaRun,
  fetchGepaOutput,
  gepaRunLabel,
  isGepaSupportedAgent,
  isSuccessfulGepaRun,
  isTerminalGepaRun,
} from './gepa';

const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === 'object' && value !== null;

describe('GEPA adapter and lifecycle guards', () => {
  it('supports only the executable Claude Code adapter', () => {
    expect(isGepaSupportedAgent('claude_code')).toBe(true);
    expect(isGepaSupportedAgent('codex')).toBe(false);
    expect(isGepaSupportedAgent('custom')).toBe(false);
  });

  it('recognizes terminal and clean-success states without guessing', () => {
    const success = { state: { life_cycle_state: 'TERMINATED', result_state: 'SUCCESS' } };
    expect(isTerminalGepaRun(success)).toBe(true);
    expect(isSuccessfulGepaRun(success)).toBe(true);
    expect(gepaRunLabel(success)).toBe('TERMINATED · SUCCESS');
    expect(isTerminalGepaRun({ state: { life_cycle_state: 'RUNNING' } })).toBe(false);
    expect(isSuccessfulGepaRun({ state: { life_cycle_state: 'INTERNAL_ERROR' } })).toBe(false);
  });
});

describe('GEPA job API', () => {
  it('dispatches bounded values as raw job_parameters and returns the run id', async () => {
    let captured: RequestInit | undefined;
    const fetcher: typeof fetch = vi.fn((_input: RequestInfo | URL, init?: RequestInit) => {
      captured = init;
      return Promise.resolve(
        new Response(JSON.stringify({ runId: 123 }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        })
      );
    });
    await expect(
      dispatchGepaRun(
        {
          agentName: 'claude_code',
          experimentId: 'exp-1',
          maxMetricCalls: 6,
          holdoutFraction: 0.4,
          maxTrainTasks: 2,
        },
        fetcher
      )
    ).resolves.toBe(123);

    const rawBody = captured?.body;
    if (typeof rawBody !== 'string') throw new Error('expected a JSON request body');
    const sent: unknown = JSON.parse(rawBody);
    if (!isRecord(sent) || !isRecord(sent.params)) throw new Error('expected params');
    const token = sent.params.idempotency_token;
    if (typeof token !== 'string') throw new Error('expected an idempotency token');
    expect(sent).toMatchObject({
      params: {
        job_parameters: {
          agent_name: 'claude_code',
          experiment_id: 'exp-1',
          max_metric_calls: '6',
          holdout_fraction: '0.4',
          max_train_tasks: '2',
          confirmed_costly_run: 'true',
        },
      },
    });
    expect(token).toMatch(/^gepa-claude_code-/);
  });

  it('surfaces backend errors and retrieves the candidate output endpoint', async () => {
    const failed: typeof fetch = vi.fn(() =>
      Promise.resolve(
        new Response(JSON.stringify({ error: 'not allowed' }), {
          status: 403,
          headers: { 'Content-Type': 'application/json' },
        })
      )
    );
    await expect(fetchGepaOutput(123, failed)).rejects.toThrow('not allowed');

    let requested: string | URL | Request | null = null;
    const ok: typeof fetch = vi.fn((input: string | URL | Request) => {
      requested = input;
      return Promise.resolve(
        new Response(JSON.stringify({ result: null, logs_truncated: false, task_error: null }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        })
      );
    });
    await expect(fetchGepaOutput(123, ok)).resolves.toEqual({
      result: null,
      logs_truncated: false,
      task_error: null,
    });
    expect(requested).toBe('/api/gepa/runs/123/output');
  });
});
