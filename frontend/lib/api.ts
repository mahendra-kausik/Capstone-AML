import type {
  CaseItem,
  DriftPayload,
  HistoryDetail,
  HistoryItem,
  PredictResult,
  UploadResult,
  User,
} from "./types";

function resolveApiBase(): string {
  if (process.env.NEXT_PUBLIC_API_URL) {
    return process.env.NEXT_PUBLIC_API_URL.replace(/\/$/, "");
  }
  if (typeof window !== "undefined") {
    return "/api/v1";
  }
  return "http://127.0.0.1:8000/api/v1";
}

const API_BASE = resolveApiBase();

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem("aml_token");
}

export function logout() {
  if (typeof window === "undefined") return;
  localStorage.removeItem("aml_token");
  localStorage.removeItem("aml_role");
  window.location.href = "/login";
}

async function handleUnauthorized(res: Response) {
  if (res.status === 401 && typeof window !== "undefined") {
    localStorage.removeItem("aml_token");
    localStorage.removeItem("aml_role");
    window.location.href = "/login";
    throw new Error("Session expired — please sign in again.");
  }
}

export async function api<T>(path: string, options: RequestInit = {}): Promise<T> {
  const token = getToken();
  const headers: HeadersInit = {
    "Content-Type": "application/json",
    ...(options.headers || {}),
  };
  if (token) {
    (headers as Record<string, string>)["Authorization"] = `Bearer ${token}`;
  }

  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, { ...options, headers });
  } catch {
    throw new Error(
      "Cannot reach API. Start the backend: cd backend && uvicorn app.main:app --reload --port 8000"
    );
  }

  await handleUnauthorized(res);

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    const detail = err.detail;
    const msg =
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
          ? detail.map((d: { msg?: string }) => d.msg).join(", ")
          : res.statusText;
    throw new Error(msg || "API error");
  }
  return res.json();
}

export async function login(email: string, password: string) {
  return api<{ access_token: string; role: string }>("/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
}

export async function getMe() {
  return api<User>("/auth/me");
}

export async function registerUser(body: {
  email: string;
  password: string;
  role: string;
}) {
  return api<User>("/auth/register", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function getMetrics() {
  return api<Record<string, any>>("/metrics");
}

export async function getDrift() {
  return api<DriftPayload>("/drift");
}

export async function getDriftEvents() {
  return api<
    Array<{
      id: string;
      event_type: string;
      model_name: string;
      window: string;
      metric_value: number;
      is_alert: boolean;
      detected_at: string;
    }>
  >("/drift/events");
}

export async function getHistory(params?: {
  risk_min?: number;
  limit?: number;
  offset?: number;
  model?: string;
  prediction?: string;
}) {
  const q = new URLSearchParams();
  if (params?.risk_min != null) q.set("risk_min", String(params.risk_min));
  if (params?.limit) q.set("limit", String(params.limit));
  if (params?.offset != null) q.set("offset", String(params.offset));
  if (params?.model) q.set("model", params.model);
  if (params?.prediction) q.set("prediction", params.prediction);
  const qs = q.toString();
  return api<{ items: HistoryItem[]; total: number; limit: number; offset: number }>(
    `/history${qs ? `?${qs}` : ""}`
  );
}

export async function getHistoryDetail(predictionId: string) {
  return api<HistoryDetail>(`/history/${predictionId}`);
}

export async function getBatches(params?: { limit?: number; offset?: number }) {
  const q = new URLSearchParams();
  if (params?.limit) q.set("limit", String(params.limit));
  if (params?.offset != null) q.set("offset", String(params.offset));
  const qs = q.toString();
  return api<{ items: Array<Record<string, unknown>>; limit: number; offset: number }>(
    `/batches${qs ? `?${qs}` : ""}`
  );
}

export async function predict(body: Record<string, unknown>) {
  return api<PredictResult>("/predict", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function explain(body: Record<string, unknown>) {
  return api<PredictResult & { method: string; shap_id?: string }>("/explain", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function uploadCsv(file: File, model = "static") {
  const token = getToken();
  const form = new FormData();
  form.append("file", file);
  const base = typeof window !== "undefined" ? "/api/v1" : API_BASE;
  let res: Response;
  try {
    res = await fetch(`${base}/upload?model=${model}`, {
      method: "POST",
      headers: token ? { Authorization: `Bearer ${token}` } : {},
      body: form,
    });
  } catch {
    throw new Error("Cannot reach API — start the backend on port 8000.");
  }
  await handleUnauthorized(res);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(typeof err.detail === "string" ? err.detail : "Upload failed");
  }
  return res.json() as Promise<UploadResult>;
}

export async function getSubgraph(txIds: string[], depth = 1) {
  const q = new URLSearchParams({ tx_ids: txIds.join(","), depth: String(depth) });
  return api<{
    nodes: Array<{ id: string; risk_score?: number; prediction?: string; is_seed?: boolean }>;
    edges: Array<{ source: string; target: string }>;
    truncated: boolean;
  }>(`/graph/subgraph?${q}`);
}

export async function getCases(params?: {
  status?: string;
  limit?: number;
  offset?: number;
}) {
  const q = new URLSearchParams();
  if (params?.status) q.set("status", params.status);
  if (params?.limit) q.set("limit", String(params.limit));
  if (params?.offset != null) q.set("offset", String(params.offset));
  const qs = q.toString();
  return api<{ items: CaseItem[]; total: number }>(`/cases${qs ? `?${qs}` : ""}`);
}

export async function getCase(caseId: string) {
  return api<CaseItem & { notes: Array<{ id: string; content: string; author_email?: string; created_at: string }> }>(
    `/cases/${caseId}`
  );
}

export async function createCase(body: {
  prediction_id: string;
  title?: string;
  priority?: string;
}) {
  return api<CaseItem>("/cases", { method: "POST", body: JSON.stringify(body) });
}

export async function updateCase(
  caseId: string,
  body: { status?: string; priority?: string; title?: string }
) {
  return api<CaseItem>(`/cases/${caseId}`, {
    method: "PATCH",
    body: JSON.stringify(body),
  });
}

export async function addCaseNote(caseId: string, content: string) {
  return api<{ id: string; content: string; created_at: string }>(
    `/cases/${caseId}/notes`,
    { method: "POST", body: JSON.stringify({ content }) }
  );
}

export async function getAuditLogs(params?: { limit?: number }) {
  const q = new URLSearchParams();
  if (params?.limit) q.set("limit", String(params.limit));
  const qs = q.toString();
  return api<{ items: Array<Record<string, unknown>> }>(
    `/admin/audit-logs${qs ? `?${qs}` : ""}`
  );
}

export async function getUsers() {
  return api<User[]>("/admin/users");
}
