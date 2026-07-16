interface PendingOnboardingResponse {
  outcome?: string;
  request_id?: string;
  run_id?: number;
}

interface OnboardingRequestOptions {
  fetchJson?: (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;
  wait?: (durationMs: number) => Promise<void>;
  now?: () => number;
}

const STATUS_URL = '/api/onboarding/status';
const POLL_INTERVAL_MS = 3_000;
const POLL_TIMEOUT_MS = 15 * 60_000;

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
  const wait = options.wait ?? ((durationMs) => new Promise((resolve) => setTimeout(resolve, durationMs)));
  const now = options.now ?? Date.now;

  let response = await fetchJson(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  let parsed = await responseBody<T>(response);

  if (!(response.ok && parsed.outcome === 'pending' && parsed.request_id && parsed.run_id != null)) {
    return { ok: response.ok, status: response.status, body: parsed };
  }

  const requestId = parsed.request_id;
  const runId = parsed.run_id;
  const deadline = now() + POLL_TIMEOUT_MS;

  while (now() < deadline) {
    await wait(POLL_INTERVAL_MS);

    try {
      response = await fetchJson(STATUS_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ request_id: requestId, run_id: runId }),
      });
      parsed = await responseBody<T>(response);
    } catch {
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
