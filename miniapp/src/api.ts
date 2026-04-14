/**
 * Typed API client for the AUG backend.
 *
 * Auth flow:
 *   1. On init, call initAuth(initDataRaw) with Telegram's signed initData string.
 *   2. The backend verifies the signature and returns a JWT.
 *   3. All subsequent requests carry that JWT as a Bearer token.
 *   4. No secrets are ever baked into the bundle.
 */

const BASE_URL = "";
const TIMEOUT_MS = 15_000;

let _token: string | null = null;

export async function initAuth(initData: string): Promise<void> {
  const response = await fetch(`${BASE_URL}/auth/telegram`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ init_data: initData }),
    signal: AbortSignal.timeout(TIMEOUT_MS),
  });
  if (!response.ok) {
    throw new Error(`Auth failed: ${response.status}`);
  }
  const body = (await response.json()) as { token: string };
  _token = body.token;
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  if (!_token) throw new Error("Not authenticated");
  const response = await fetch(`${BASE_URL}${path}`, {
    ...options,
    signal: AbortSignal.timeout(TIMEOUT_MS),
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${_token}`,
      ...options.headers,
    },
  });
  if (!response.ok) {
    const text = await response.text().catch(() => response.statusText);
    throw new Error(`${response.status} ${text}`);
  }
  return response.json() as Promise<T>;
}

export function getSettings(): Promise<Record<string, unknown>> {
  return request<Record<string, unknown>>("/settings");
}

export function putSettings(
  data: Record<string, unknown>
): Promise<Record<string, unknown>> {
  return request<Record<string, unknown>>("/settings", {
    method: "PUT",
    body: JSON.stringify(data),
  });
}

export function getModels(): Promise<string[]> {
  return request<string[]>("/models");
}
