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
    // Try to extract the backend's detail message
    try {
      const body = await response.json();
      const detail = body?.detail;
      if (typeof detail === "string") throw new Error(detail);
    } catch (e) {
      if (e instanceof Error && e.message !== "") throw e;
    }
    throw new Error(`${response.status} ${response.statusText}`);
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

// ---------------------------------------------------------------------------
// Skills (local — authenticated)
// ---------------------------------------------------------------------------

import type {
  ClawHubListResponse,
  ClawHubSearchResult,
  ClawHubSkillDetail,
  SkillDetail,
  SkillSummary,
} from "./types.ts";

async function requestText(path: string): Promise<string> {
  if (!_token) throw new Error("Not authenticated");
  const response = await fetch(`${BASE_URL}${path}`, {
    signal: AbortSignal.timeout(TIMEOUT_MS),
    headers: { Authorization: `Bearer ${_token}` },
  });
  if (!response.ok) {
    const text = await response.text().catch(() => response.statusText);
    throw new Error(`${response.status} ${text}`);
  }
  return response.text();
}

export function listSkills(): Promise<SkillSummary[]> {
  return request<SkillSummary[]>("/skills");
}

export function getSkillDetail(name: string): Promise<SkillDetail> {
  return request<SkillDetail>(`/skills/${encodeURIComponent(name)}`);
}

export function getSkillFile(name: string, path: string): Promise<string> {
  return requestText(
    `/skills/${encodeURIComponent(name)}/file?path=${encodeURIComponent(path)}`
  );
}

export function updateSkill(
  name: string,
  patch: Partial<{ description: string; body: string; always_on: boolean }>
): Promise<void> {
  return request<void>(`/skills/${encodeURIComponent(name)}`, {
    method: "PUT",
    body: JSON.stringify(patch),
  });
}

export function updateSkillFile(
  name: string,
  path: string,
  content: string
): Promise<void> {
  return request<void>(
    `/skills/${encodeURIComponent(name)}/file?path=${encodeURIComponent(path)}`,
    { method: "PUT", body: JSON.stringify({ content }) }
  );
}

export function deleteSkill(name: string): Promise<void> {
  return request<void>(`/skills/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
}

export function deleteSkillFile(name: string, path: string): Promise<void> {
  return request<void>(
    `/skills/${encodeURIComponent(name)}/file?path=${encodeURIComponent(path)}`,
    { method: "DELETE" }
  );
}

export function installSkill(
  name: string,
  slug: string,
  version?: string
): Promise<void> {
  return request<void>(`/skills/${encodeURIComponent(name)}/install`, {
    method: "POST",
    body: JSON.stringify({ slug, version }),
  });
}

// ---------------------------------------------------------------------------
// ClawHub (direct browser calls — public, no auth)
// ---------------------------------------------------------------------------

const CLAWHUB = "https://clawhub.ai/api/v1";
const CLAWHUB_TIMEOUT = 15_000;

async function clawhubFetch<T>(path: string): Promise<T> {
  const response = await fetch(`${CLAWHUB}${path}`, {
    signal: AbortSignal.timeout(CLAWHUB_TIMEOUT),
  });
  if (!response.ok) throw new Error(`ClawHub ${response.status}`);
  return response.json() as Promise<T>;
}

async function clawhubFetchText(path: string): Promise<string> {
  const response = await fetch(`${CLAWHUB}${path}`, {
    signal: AbortSignal.timeout(CLAWHUB_TIMEOUT),
  });
  if (!response.ok) throw new Error(`ClawHub ${response.status}`);
  return response.text();
}

export function clawhubList(cursor?: string): Promise<ClawHubListResponse> {
  const params = new URLSearchParams({ family: "skill", limit: "20" });
  if (cursor) params.set("cursor", cursor);
  return clawhubFetch<ClawHubListResponse>(`/packages?${params}`);
}

export function clawhubSearch(q: string): Promise<ClawHubSearchResult[]> {
  const params = new URLSearchParams({ q, limit: "20" });
  return clawhubFetch<{ results: ClawHubSearchResult[] }>(
    `/search?${params}`
  ).then((r) => r.results);
}

export function clawhubGetSkill(slug: string): Promise<ClawHubSkillDetail> {
  return clawhubFetch<ClawHubSkillDetail>(`/skills/${encodeURIComponent(slug)}`);
}

export function clawhubGetFile(slug: string, path: string): Promise<string> {
  return clawhubFetchText(
    `/skills/${encodeURIComponent(slug)}/file?path=${encodeURIComponent(path)}`
  );
}
