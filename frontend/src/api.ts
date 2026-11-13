// Set VITE_API_URL in production (e.g. https://your-backend.example.com). Empty = same origin / dev proxy.
const BASE = import.meta.env.VITE_API_URL ?? "";

export type ChatResult = { reply: string; provider: string; latency_ms: number };
export type HistoryRow = { id: number; provider: string; prompt: string; reply: string; latency_ms: number };

async function request<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(BASE + path, {
    method: body ? "POST" : "GET",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail ?? `Request failed (${res.status})`);
  }
  return res.json();
}

export const getProviders = () => request<Record<string, boolean>>("/api/providers");
export const getHistory = () => request<HistoryRow[]>("/api/history?limit=10");
export const chat = (message: string, provider: string, image_b64?: string, image_mime?: string) =>
  request<ChatResult>("/api/chat", { message, provider, image_b64, image_mime });
