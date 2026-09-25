/**
 * VAYU-NET Centralized API Configuration & Client
 * All requests route through API_BASE_URL (configured via VITE_API_BASE_URL).
 */

export const API_BASE_URL =
  (import.meta.env.VITE_API_BASE_URL as string) || "http://localhost:8000";

export function resolveAssetUrl(path: string): string {
  if (!path) return "";
  if (path.startsWith("http://") || path.startsWith("https://")) {
    return path;
  }
  const cleanPath = path.startsWith("/") ? path : `/${path}`;
  return `${API_BASE_URL}${cleanPath}`;
}

async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  const headers = {
    "Content-Type": "application/json",
    Accept: "application/json",
    ...options.headers,
  };

  const response = await fetch(url, { ...options, headers });
  if (!response.ok) {
    const errorBody = await response.text();
    throw new Error(`API Error [${response.status}] ${endpoint}: ${errorBody}`);
  }
  return response.json() as Promise<T>;
}

export const vayuApi = {
  getHealth: () => request<{ status: string; service: string; version: string }>("/health"),

  listCyclones: (shortlistOnly = true) =>
    request<Array<{
      cyclone_id: string;
      storm_name: string;
      year: number;
      split: string;
      peak_category: string;
      max_wind_kt: number;
      samples_count: number;
      is_demo_shortlist: boolean;
    }>>(`/api/cyclones?shortlist_only=${shortlistOnly}`),

  getCycloneDetails: (cycloneId: string) =>
    request<{
      cyclone_id: string;
      storm_name: string;
      year: number;
      split: string;
      peak_category: string;
      max_wind_kt: number;
      samples: Array<{
        sample_id: string;
        t0: string;
        center: [number, number];
        wind_kt: number | null;
        category: string;
      }>;
    }>(`/api/cyclones/${encodeURIComponent(cycloneId)}`),

  getForecast: (cycloneId: string, t0?: string, percentile = "p80", mode = "MODEL_INFERENCE") => {
    const query = new URLSearchParams();
    if (t0) query.set("t0", t0);
    query.set("percentile", percentile);
    query.set("mode", mode);
    return request<any>(`/api/cyclones/${encodeURIComponent(cycloneId)}/forecast?${query.toString()}`);
  },

  getVerification: (cycloneId: string, t0?: string, mode = "MODEL_INFERENCE") => {
    const query = new URLSearchParams();
    if (t0) query.set("t0", t0);
    query.set("mode", mode);
    return request<any>(`/api/cyclones/${encodeURIComponent(cycloneId)}/verification?${query.toString()}`);
  },

  getAnalogs: (cycloneId: string, t0?: string, k = 2) => {
    const query = new URLSearchParams();
    if (t0) query.set("t0", t0);
    query.set("k", String(k));
    return request<any>(`/api/cyclones/${encodeURIComponent(cycloneId)}/analogs?${query.toString()}`);
  },

  getExplainability: (cycloneId: string) =>
    request<any>(`/api/cyclones/${encodeURIComponent(cycloneId)}/explainability`),

  predict: (data: { cyclone_id: string; t0?: string; percentile?: string; mode?: string }) =>
    request<any>("/api/predict", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  runInference: (data: { event_id: string; t0_utc?: string }) =>
    request<any>("/api/inference/run", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  runCurrentInference: (data: { event_id: string; t0_utc?: string }) =>
    request<any>("/api/inference/current", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  getCurrentEvents: () => request<Array<any>>("/api/current/events"),

  getEvents: () => request<Array<any>>("/api/events"),

  getEventDetails: (eventId: string) =>
    request<any>(`/api/events/${encodeURIComponent(eventId)}`),

  getInferenceById: (inferenceId: string) =>
    request<any>(`/api/inference/${encodeURIComponent(inferenceId)}`),
};
