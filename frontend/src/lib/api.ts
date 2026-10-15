/** Typed fetch client for the AI TriboPredict backend. */

export const API_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");
const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "";

export class ApiError extends Error {
  status: number;
  detail: unknown;
  constructor(status: number, detail: unknown) {
    super(typeof detail === "string" ? detail : JSON.stringify(detail));
    this.status = status;
    this.detail = detail;
  }
}

function headers(extra?: Record<string, string>): Record<string, string> {
  const h: Record<string, string> = { ...(extra || {}) };
  if (API_KEY) h["X-API-Key"] = API_KEY;
  return h;
}

async function parse<T>(res: Response): Promise<T> {
  const text = await res.text();
  let body: unknown = text;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    /* non-JSON */
  }
  if (!res.ok) {
    const detail = body && typeof body === "object" && "detail" in (body as Record<string, unknown>) ? (body as Record<string, unknown>).detail : body || res.statusText;
    throw new ApiError(res.status, detail);
  }
  return body as T;
}

export async function apiGet<T = unknown>(path: string, params?: Record<string, string | number | boolean | null | undefined>): Promise<T> {
  const url = new URL(API_URL + path);
  if (params) for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null && v !== "") url.searchParams.set(k, String(v));
  const res = await fetch(url.toString(), { headers: headers(), cache: "no-store" });
  return parse<T>(res);
}

export async function apiSend<T = unknown>(method: "POST" | "PUT" | "PATCH" | "DELETE", path: string, body?: unknown): Promise<T> {
  const res = await fetch(API_URL + path, {
    method,
    headers: headers(body !== undefined ? { "Content-Type": "application/json" } : undefined),
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  return parse<T>(res);
}

export const apiPost = <T = unknown>(path: string, body?: unknown) => apiSend<T>("POST", path, body);
export const apiPut = <T = unknown>(path: string, body?: unknown) => apiSend<T>("PUT", path, body);
export const apiPatch = <T = unknown>(path: string, body?: unknown) => apiSend<T>("PATCH", path, body);
export const apiDelete = <T = unknown>(path: string) => apiSend<T>("DELETE", path);

export async function apiUpload<T = unknown>(path: string, form: FormData): Promise<T> {
  const res = await fetch(API_URL + path, { method: "POST", headers: headers(), body: form });
  return parse<T>(res);
}

export function downloadUrl(path: string): string {
  return API_URL + path;
}

export async function apiDownload(path: string, filename: string): Promise<void> {
  const res = await fetch(API_URL + path, { headers: headers() });
  if (!res.ok) throw new ApiError(res.status, await res.text());
  const blob = await res.blob();
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

export function errorMessage(e: unknown): string {
  if (e instanceof ApiError) {
    const d = e.detail as unknown;
    if (typeof d === "string") return d;
    if (d && typeof d === "object" && "message" in (d as Record<string, unknown>)) return String((d as Record<string, unknown>).message);
    if (d && typeof d === "object" && "errors" in (d as Record<string, unknown>)) {
      const errs = (d as { errors: Array<{ loc?: unknown[]; msg?: string }> }).errors;
      return errs.map((x) => `${(x.loc || []).slice(1).join(".")}: ${x.msg}`).join("; ");
    }
    return JSON.stringify(d);
  }
  if (e instanceof Error) return e.message.includes("fetch") ? `Cannot reach the backend at ${API_URL}. Is it running?` : e.message;
  return String(e);
}
