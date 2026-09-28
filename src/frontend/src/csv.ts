import type { Row } from "./types";

function escapeField(v: unknown): string {
  if (v === null || v === undefined) return "";
  const s = String(v);
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function download(filename: string, blob: Blob): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);  // detached anchors are not clicked reliably (Firefox)
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);  // revoking synchronously can cancel the download
}

export function downloadBlob(filename: string, blob: Blob): void {
  download(filename, blob);
}

export function downloadCsv(filename: string, rows: Row[]): void {
  if (!rows.length) return;
  const cols = Object.keys(rows[0]);
  const lines = [cols.join(","), ...rows.map((r) => cols.map((c) => escapeField(r[c])).join(","))];
  download(filename, new Blob([lines.join("\r\n")], { type: "text/csv;charset=utf-8" }));
}
