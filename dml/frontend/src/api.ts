import type { EstimateRequest, EstimateResponse, ExportFormat, ExportKind, Q1Config, Row, RunDetail } from "./types";

async function errorDetail(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) {
      return body.detail
        .map((d: { loc?: unknown[]; msg?: string }) => `${(d.loc ?? []).slice(1).join(".")}: ${d.msg ?? ""}`)
        .join("; ");
    }
    return JSON.stringify(body.detail ?? body);
  } catch {
    return res.statusText;
  }
}

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error(await errorDetail(res));
  return res.json() as Promise<T>;
}

async function handleBlob(res: Response): Promise<Blob> {
  if (!res.ok) throw new Error(await errorDetail(res));
  return res.blob();
}

export function getConfig(): Promise<Q1Config> {
  return fetch("/api/q1/config").then(handle<Q1Config>);
}

export function estimate(req: EstimateRequest): Promise<EstimateResponse> {
  return fetch("/api/q1/estimate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  }).then(handle<EstimateResponse>);
}

export function exportFigure(req: EstimateRequest, figure: ExportKind, format: ExportFormat): Promise<Blob> {
  return fetch("/api/q1/export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...req, figure, format }),
  }).then(handleBlob);
}

export function listRuns(): Promise<{ runs: Row[] }> {
  return fetch("/api/runs").then(handle<{ runs: Row[] }>);
}

export function getRun(question: string, runId: string): Promise<RunDetail> {
  return fetch(`/api/runs/${encodeURIComponent(question)}/${encodeURIComponent(runId)}`).then(handle<RunDetail>);
}

export function exportRunFigure(question: string, runId: string, figure: ExportKind, format: ExportFormat): Promise<Blob> {
  return fetch(`/api/runs/${encodeURIComponent(question)}/${encodeURIComponent(runId)}/export/${figure}?format=${format}`).then(handleBlob);
}
