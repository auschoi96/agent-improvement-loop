interface PendingOnboardingResponse {
  outcome?: string;
  request_id?: string;
}

interface OnboardingRequestOptions {
  fetchJson?: (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;
  wait?: (durationMs: number) => Promise<void>;
  now?: () => number;
  signal?: AbortSignal;
}

const STATUS_URL = '/api/onboarding/status';
const POLL_INTERVAL_MS = 3_000;
const POLL_TIMEOUT_MS = 15 * 60_000;

function abortableWait(durationMs: number, signal?: AbortSignal): Promise<void> {
  if (signal?.aborted) return Promise.reject(new DOMException('Request aborted', 'AbortError'));
  return new Promise((resolve, reject) => {
    const onAbort = () => {
      clearTimeout(timer);
      reject(new DOMException('Request aborted', 'AbortError'));
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, durationMs);
    signal?.addEventListener('abort', onAbort, { once: true });
  });
}

function isRetryableStatus(status: number): boolean {
  return status === 408 || status === 425 || status === 429 || status >= 500;
}

async function responseBody<T>(response: Response): Promise<T & PendingOnboardingResponse> {
  return (await response.json().catch(() => ({}))) as T & PendingOnboardingResponse;
}

// The action POST is intentionally sent exactly once. If it starts a Lakeflow
// job, only the idempotent status lookup is retried after transient failures.
export async function postOnboardingJson<T>(
  url: string,
  body: unknown,
  options: OnboardingRequestOptions = {}
): Promise<{ ok: boolean; status: number; body: T }> {
  const fetchJson = options.fetchJson ?? fetch;
  const wait = options.wait ?? ((durationMs) => abortableWait(durationMs, options.signal));
  const now = options.now ?? Date.now;

  let response = await fetchJson(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal: options.signal,
  });
  let parsed = await responseBody<T>(response);

  if (!(response.ok && parsed.outcome === 'pending' && parsed.request_id)) {
    return { ok: response.ok, status: response.status, body: parsed };
  }

  const requestId = parsed.request_id;
  const deadline = now() + POLL_TIMEOUT_MS;

  while (now() < deadline) {
    await wait(POLL_INTERVAL_MS);

    try {
      response = await fetchJson(STATUS_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ request_id: requestId }),
        signal: options.signal,
      });
      parsed = await responseBody<T>(response);
    } catch (error) {
      if (options.signal?.aborted) throw error;
      continue;
    }

    if (isRetryableStatus(response.status)) continue;
    if (response.ok && parsed.outcome === 'pending') continue;

    return { ok: response.ok, status: response.status, body: parsed };
  }

  return {
    ok: false,
    status: 504,
    body: { outcome: 'error', error: 'Onboarding did not finish within 15 minutes.' } as T,
  };
}
