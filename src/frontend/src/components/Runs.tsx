import { useEffect, useState } from "react";
import { exportRunFigure, getRun, listRuns } from "../api";
import { downloadBlob, downloadCsv } from "../csv";
import type { ExportFormat, ExportKind, Row, RunDetail } from "../types";
import Plot from "./Plot";

function fmt(v: unknown): string {
  if (v === null || v === undefined) return "—";
  return typeof v === "number" ? v.toFixed(4) : String(v);
}

function fmtDate(v: unknown): string {
  const d = new Date(String(v));
  return Number.isNaN(d.getTime()) ? String(v) : d.toLocaleString();
}

function Table({ rows }: { rows: Row[] }) {
  const cols = rows.length ? Object.keys(rows[0]) : [];
  return (
    <table className="table">
      <thead>
        <tr>{cols.map((c) => <th key={c}>{c}</th>)}</tr>
      </thead>
      <tbody>
        {rows.slice(0, 500).map((r, i) => (
          <tr key={i}>{cols.map((c) => <td key={c}>{fmt(r[c])}</td>)}</tr>
        ))}
      </tbody>
    </table>
  );
}

function Detail({ question, runId, onBack }: { question: string; runId: string; onBack: () => void }) {
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [exporting, setExporting] = useState<string | null>(null);

  useEffect(() => {
    getRun(question, runId)
      .then(setDetail)
      .catch((e) => setError(String(e.message)));
  }, [question, runId]);

  const doExport = (kind: ExportKind, format: ExportFormat) => {
    setExporting(`${kind}.${format}`);
    setError(null);
    exportRunFigure(question, runId, kind, format)
      .then((blob) => downloadBlob(`${runId}-${kind}.${format}`, blob))
      .catch((e) => setError(String(e.message)))
      .finally(() => setExporting(null));
  };

  const exportButtons = (kind: ExportKind) => (
    <div className="exports">
      <button className="btn tiny" disabled={!!exporting} onClick={() => doExport(kind, "png")}>PNG</button>
      <button className="btn tiny" disabled={!!exporting} onClick={() => doExport(kind, "svg")}>SVG</button>
    </div>
  );

  if (error && !detail) return <div className="error">{error}</div>;
  if (!detail) return <div className="card">loading…</div>;

  const manifestRows = Object.entries(detail.manifest).map(([k, v]) => ({
    key: k,
    value: typeof v === "object" && v !== null ? JSON.stringify(v) : String(v),
  }));
  const metrics = (detail.manifest.metrics ?? {}) as Row;

  return (
    <div>
      <button className="btn" onClick={onBack}>← all runs</button>
      <div className="card">
        <div className="card-head">
          <h2>{String(detail.manifest.name)}</h2>
          <div className="exports">
            <button className="btn tiny" disabled={!!exporting} onClick={() => doExport("coefficients", "png")}>
              forest PNG
            </button>
            <button className="btn tiny" disabled={!!exporting} onClick={() => doExport("coefficients", "svg")}>
              forest SVG
            </button>
          </div>
        </div>
        <p>
          <span className="badge">{String(detail.manifest.role)}</span>{" "}
          {question} / {runId} · created {fmtDate(detail.manifest.created)}
        </p>
        <h3 style={{ marginTop: 14 }}>metrics</h3>
        <Table rows={Object.entries(metrics).map(([k, v]) => ({ metric: k, value: fmt(v) }))} />
      </div>
      {error && (
        <div className="error">
          {error}
          <button className="btn tiny" onClick={() => setError(null)}>dismiss</button>
        </div>
      )}
      {Object.entries(detail.figures).map(([name, fig]) => (
        <div className="card" key={name}>
          <div className="card-head">
            <h3>{name.replace(/_/g, " ")}</h3>
            {name === "residualized_fit" ? exportButtons("residualized_fit") : null}
          </div>
          <Plot fig={fig} />
        </div>
      ))}
      <div className="card">
        <div className="card-head">
          <h3>Influence (static)</h3>
          {exportButtons("influence")}
        </div>
        <p>Leave-one-meeting-out sensitivity from the saved residuals — export as PNG/SVG for slides.</p>
      </div>
      {Object.entries(detail.tables).map(([name, rows]) => (
        <details className="card" key={name}>
          <summary>{name} ({rows.length} rows)</summary>
          <Table rows={rows} />
          <button className="btn tiny" onClick={() => downloadCsv(`${runId}-${name}.csv`, rows)}>download CSV</button>
        </details>
      ))}
      <details className="card">
        <summary>manifest</summary>
        <Table rows={manifestRows} />
      </details>
    </div>
  );
}

export default function Runs() {
  const [runs, setRuns] = useState<Row[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<{ question: string; runId: string } | null>(null);

  useEffect(() => {
    listRuns()
      .then((r) => setRuns(r.runs))
      .catch((e) => setError(String(e.message)));
  }, []);

  if (selected) {
    return (
      <Detail
        question={selected.question}
        runId={selected.runId}
        onBack={() => setSelected(null)}
      />
    );
  }
  if (error) return <div className="error">{error}</div>;
  if (!runs) return <div className="card">loading…</div>;
  if (!runs.length) return <div className="card placeholder"><strong>No runs saved yet.</strong></div>;

  return (
    <div className="card">
      <table className="table">
        <thead>
          <tr>
            <th>run_id</th><th>name</th><th>role</th><th>created</th>
            <th>theta</th><th>se</th><th>n_meetings</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((r, i) => (
            <tr key={i} className="clickable"
              onClick={() => setSelected({ question: String(r.question), runId: String(r.run_id) })}>
              <td>{String(r.run_id)}</td>
              <td>{String(r.name)}</td>
              <td><span className="badge">{String(r.role)}</span></td>
              <td>{fmtDate(r.created)}</td>
              <td>{fmt(r["m.theta"])}</td>
              <td>{fmt(r["m.se"])}</td>
              <td>{fmt(r["m.n_meetings"])}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
