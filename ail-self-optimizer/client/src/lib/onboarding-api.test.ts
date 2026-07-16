import { afterEach, describe, expect, it, vi } from 'vitest';
import { postOnboardingJson } from './onboarding-api';

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

describe('postOnboardingJson', () => {
  afterEach(() => vi.useRealTimers());

  it('recovers when a status poll has a transient network failure', async () => {
    const fetchJson = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ outcome: 'pending', request_id: 'req-1' }))
      .mockRejectedValueOnce(new TypeError('connection reset'))
      .mockResolvedValueOnce(
        jsonResponse({ outcome: 'validated', experiment_id: 'exp-1', name: 'existing', fresh: true })
      );

    const result = await postOnboardingJson<{ outcome: string; experiment_id?: string }>(
      '/api/onboarding/experiment/validate',
      { experiment_id: 'exp-1' },
      { fetchJson, wait: () => Promise.resolve() }
    );

    expect(result).toEqual({
      ok: true,
      status: 200,
      body: { outcome: 'validated', experiment_id: 'exp-1', name: 'existing', fresh: true },
    });
    expect(fetchJson).toHaveBeenCalledTimes(3);
    expect(fetchJson).toHaveBeenNthCalledWith(
      3,
      '/api/onboarding/status',
      expect.objectContaining({ body: JSON.stringify({ request_id: 'req-1' }) })
    );
  });

  it('retries transient server errors but returns authentication failures', async () => {
    const fetchJson = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ outcome: 'pending', request_id: 'req-2' }))
      .mockResolvedValueOnce(jsonResponse({ outcome: 'error', error: 'temporarily unavailable' }, 503))
      .mockResolvedValueOnce(jsonResponse({ outcome: 'error', error: 'sign in' }, 401));

    const result = await postOnboardingJson<{ outcome: string; error?: string }>(
      '/api/onboarding/experiment/validate',
      { experiment_id: 'exp-2' },
      { fetchJson, wait: () => Promise.resolve() }
    );

    expect(result).toEqual({
      ok: false,
      status: 401,
      body: { outcome: 'error', error: 'sign in' },
    });
    expect(fetchJson).toHaveBeenCalledTimes(3);
  });

  it('does not retry a failed initial action request', async () => {
    const fetchJson = vi.fn().mockRejectedValueOnce(new TypeError('offline'));

    await expect(
      postOnboardingJson(
        '/api/onboarding/experiment/validate',
        { experiment_id: 'exp-3' },
        { fetchJson, wait: () => Promise.resolve() }
      )
    ).rejects.toThrow('offline');
    expect(fetchJson).toHaveBeenCalledOnce();
  });

  it('cancels the polling wait without issuing another status request', async () => {
    vi.useFakeTimers();
    const controller = new AbortController();
    const fetchJson = vi.fn().mockResolvedValueOnce(jsonResponse({ outcome: 'pending', request_id: 'req-abort' }));

    const pending = postOnboardingJson(
      '/api/onboarding/experiment/validate',
      {},
      { fetchJson, signal: controller.signal }
    );
    controller.abort();

    await expect(pending).rejects.toMatchObject({ name: 'AbortError' });
    expect(fetchJson).toHaveBeenCalledOnce();
  });
});
