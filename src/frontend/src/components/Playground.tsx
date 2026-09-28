import { useEffect, useState } from "react";
import { estimate, exportFigure, getConfig } from "../api";
import { downloadBlob, downloadCsv } from "../csv";
import type { EstimateRequest, EstimateResponse, ExportFormat, ExportKind, Q1Config, Row } from "../types";
import Plot from "./Plot";

type Auto = { auto: boolean; value: string };

type Form = {
  outcome: string;
  treatment: string;
  outcome_ticker: string;
  nuisance: string;
  se: string;
  alpha: string;
  min_train: Auto;
  test_size: Auto;
  step: Auto;
  embargo: string;
  hac_lag: string;
  n_boot: string;
  seed: string;
  level: string;
  features: string[];
};

const AUTO: Auto = { auto: true, value: "" };

function fmt(v: unknown, digits = 4): string {
  if (v === null || v === undefined) return "—";
  return typeof v === "number" ? v.toFixed(digits) : String(v);
}

function Table({ rows, cols, emphasize }: {
  rows: Row[]; cols: string[]; emphasize?: (row: Row) => boolean;
}) {
  return (
    <table className="table">
      <thead>
        <tr>{cols.map((c) => <th key={c}>{c}</th>)}</tr>
      </thead>
      <tbody>
        {rows.map((r, i) => (
          <tr key={i} className={emphasize?.(r) ? "row-accent" : undefined}>
            {cols.map((c) => <td key={c}>{fmt(r[c])}</td>)}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

const isPrimaryDml = (r: Row) =>
  String(r.estimator).startsWith("plr_dml") && !String(r.estimator).includes("bootstrap");

export default function Playground() {
  const [config, setConfig] = useState<Q1Config | null>(null);
  const [form, setForm] = useState<Form | null>(null);
  const [result, setResult] = useState<EstimateResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [exporting, setExporting] = useState<string | null>(null);

  useEffect(() => {
    getConfig().then((c) => {
      setConfig(c);
      setForm({
        outcome: c.defaults.outcome, treatment: c.defaults.treatment,
        outcome_ticker: c.defaults.outcome_ticker, nuisance: c.defaults.nuisance,
        se: c.defaults.se, alpha: "", min_train: AUTO, test_size: AUTO, step: AUTO,
        embargo: String(c.defaults.embargo), hac_lag: String(c.defaults.hac_lag),
        n_boot: String(c.defaults.n_boot), seed: String(c.defaults.seed),
        level: String(c.defaults.level),
        features: [...(c.features[c.defaults.outcome] ?? [])],
      });
    }).catch((e) => setError(String(e.message)));
  }, []);

  if (!config || !form) return <div className="card">{error ?? "loading config…"}</div>;

  const featureInfo = config.feature_info[form.outcome] ?? {};
  const set = (patch: Partial<Form>) => setForm({ ...form, ...patch });

  const buildRequest = (): EstimateRequest => ({
    outcome: form.outcome, treatment: form.treatment, outcome_ticker: form.outcome_ticker,
    features: form.features, nuisance: form.nuisance,
    alpha: form.alpha.trim() === "" ? null : Number(form.alpha),
    min_train: form.min_train.auto ? null : Number(form.min_train.value),
    test_size: form.test_size.auto ? null : Number(form.test_size.value),
    step: form.step.auto ? null : Number(form.step.value),
    embargo: Number(form.embargo), se: form.se, hac_lag: Number(form.hac_lag),
    n_boot: Number(form.n_boot), seed: Number(form.seed), level: Number(form.level),
  });

  const submit = () => {
    if (form.features.length === 0) {
      setError("select at least one control feature");
      return;
    }
    setPending(true);
    setError(null);
    estimate(buildRequest())
      .then((r) => setResult(r))
      .catch((e) => setError(String(e.message)))
      .finally(() => setPending(false));
  };

  const doExport = (kind: ExportKind, format: ExportFormat) => {
    if (form.features.length === 0) {
      setError("select at least one control feature");
      return;
    }
    setExporting(`${kind}.${format}`);
    setError(null);
    exportFigure(buildRequest(), kind, format)
      .then((blob) => downloadBlob(`q1-${form.outcome}-${form.treatment}-${kind}.${format}`, blob))
      .catch((e) => setError(String(e.message)))
      .finally(() => setExporting(null));
  };

  const autoInput = (label: string, key: "min_train" | "test_size" | "step") => (
    <div className="field">
      <label>{label}</label>
      <span className="inline-row">
        <input
          type="number"
          disabled={form[key].auto}
          value={form[key].value}
          onChange={(e) => set({ [key]: { auto: false, value: e.target.value } } as Partial<Form>)}
        />
        <label className="inline">
          <input
            type="checkbox"
            checked={form[key].auto}
            onChange={(e) => set({ [key]: { auto: e.target.checked, value: form[key].value } } as Partial<Form>)}
          />
          auto
        </label>
      </span>
    </div>
  );

  const num = (label: string, key: "embargo" | "hac_lag" | "n_boot" | "seed" | "level", disabled = false) => (
    <div className="field">
      <label>{label}</label>
      <input
        type="number"
        disabled={disabled}
        value={form[key]}
        onChange={(e) => set({ [key]: e.target.value } as Partial<Form>)}
      />
    </div>
  );

  return (
    <div className="split">
      <aside className="panel">
        <div className="field">
          <label>outcome</label>
          <select value={form.outcome}
            onChange={(e) => set({ outcome: e.target.value, features: [...(config.features[e.target.value] ?? [])] })}>
            {config.outcomes.map((o) => <option key={o}>{o}</option>)}
          </select>
        </div>
        <div className="field">
          <label>treatment</label>
          <select value={form.treatment} onChange={(e) => set({ treatment: e.target.value })}>
            {config.surprises.map((t) => <option key={t}>{t}</option>)}
          </select>
        </div>
        {form.outcome === "etf_day0" && (
          <div className="field">
            <label>outcome ticker</label>
            <select value={form.outcome_ticker} onChange={(e) => set({ outcome_ticker: e.target.value })}>
              {config.tickers.map((t) => <option key={t}>{t}</option>)}
            </select>
          </div>
        )}
        <div className="field">
          <label>nuisance</label>
          <select value={form.nuisance} onChange={(e) => set({ nuisance: e.target.value })}>
            {config.nuisances.map((n) => <option key={n}>{n}</option>)}
          </select>
        </div>
        <div className="field">
          <label>alpha (blank = learner default)</label>
          <input type="number" value={form.alpha} onChange={(e) => set({ alpha: e.target.value })} />
        </div>
        {autoInput("min_train", "min_train")}
        {autoInput("test_size", "test_size")}
        {autoInput("step", "step")}
        {num("embargo", "embargo")}
        <div className="field">
          <label>se</label>
          <select value={form.se} onChange={(e) => set({ se: e.target.value })}>
            {config.se_types.map((s) => <option key={s}>{s}</option>)}
          </select>
        </div>
        {num("hac_lag", "hac_lag", form.se !== "HAC")}
        {num("n_boot", "n_boot")}
        {num("seed", "seed")}
        {num("level", "level")}
        <hr className="divider" />
        <div className="field">
          <label>
            controls
            <span className="links">
              <button type="button" className="btn tiny" onClick={() => set({ features: Object.keys(featureInfo) })}>
                Select all
              </button>
              <button type="button" className="btn tiny" onClick={() => set({ features: [] })}>
                Clear
              </button>
            </span>
          </label>
          {Object.keys(featureInfo).map((f) => (
            <label key={f} className="check" title={featureInfo[f]}>
              <input
                type="checkbox"
                checked={form.features.includes(f)}
                onChange={(e) => set({
                  features: e.target.checked ? [...form.features, f] : form.features.filter((x) => x !== f),
                })}
              />
              {f}
            </label>
          ))}
        </div>
        <button className="btn primary" disabled={pending} onClick={submit}>
          {pending ? "estimating…" : "Run"}
        </button>
        {error && (
          <div className="error">
            {error}
            <button className="btn tiny" onClick={() => setError(null)}>dismiss</button>
          </div>
        )}
      </aside>

      <main className={pending ? "results pending" : "results"}>
        {!result && !pending && (
          <div className="card placeholder">
            <strong>Set the knobs and press Run.</strong>
            Estimates, charts and slide-ready exports appear here.
          </div>
        )}
        {!result && pending && <div className="card placeholder"><strong>estimating…</strong></div>}
        {result && (
          <>
            <div className="card">
              <div className="card-head">
                <h2>{form.outcome} · {form.treatment} · {form.nuisance}</h2>
                <span className="badge">{String(result.metrics.se_type)}</span>
              </div>
              <div className="stats">
                <div className="stat">
                  <div className="stat-label">theta</div>
                  <div className="stat-value">{fmt(result.metrics.theta, 3)}</div>
                  <div className="stat-sub">% per 10bp surprise</div>
                </div>
                <div className="stat">
                  <div className="stat-label">CI</div>
                  <div className="stat-value">[{fmt(result.metrics.ci_low, 2)}, {fmt(result.metrics.ci_high, 2)}]</div>
                  <div className="stat-sub">se {fmt(result.metrics.se)}</div>
                </div>
                <div className="stat">
                  <div className="stat-label">meetings</div>
                  <div className="stat-value">{String(result.metrics.n_meetings)}</div>
                  <div className="stat-sub">scored · {String(result.metrics.n_meetings_discarded)} discarded</div>
                </div>
                <div className="stat">
                  <div className="stat-label">r² y / s</div>
                  <div className="stat-value">{fmt(result.metrics.r2_y, 2)} / {fmt(result.metrics.r2_s, 2)}</div>
                  <div className="stat-sub">out-of-fold nuisances</div>
                </div>
                <div className="stat">
                  <div className="stat-label">sd s resid</div>
                  <div className="stat-value">{fmt(result.metrics.sd_s_resid)}</div>
                  <div className="stat-sub">residual treatment variation</div>
                </div>
              </div>
            </div>

            <div className="card">
              <div className="card-head">
                <h3>Estimates</h3>
                <div className="exports">
                  <button className="btn tiny"
                    onClick={() => downloadCsv("q1-estimates.csv", result.estimates)}>CSV</button>
                  <span className="sep" />
                  <button className="btn tiny" disabled={!!exporting}
                    onClick={() => doExport("coefficients", "png")}>forest PNG</button>
                  <button className="btn tiny" disabled={!!exporting}
                    onClick={() => doExport("coefficients", "svg")}>forest SVG</button>
                </div>
              </div>
              <Table rows={result.estimates} emphasize={isPrimaryDml}
                cols={["estimator", "theta", "se", "ci_low", "ci_high", "se_type", "n_meetings"]} />
              <Plot fig={result.figures.coefficients} />
              <p>“forest” exports a static seaborn plot of these estimates with CIs, for slides.</p>
            </div>

            <div className="card">
              <div className="card-head">
                <h3>Residualized fit</h3>
                <div className="exports">
                  <button className="btn tiny" disabled={!!exporting}
                    onClick={() => doExport("residualized_fit", "png")}>PNG</button>
                  <button className="btn tiny" disabled={!!exporting}
                    onClick={() => doExport("residualized_fit", "svg")}>SVG</button>
                </div>
              </div>
              <Plot fig={result.figures.residualized_fit} />
            </div>

            <div className="card">
              <div className="card-head">
                <h3>Influence</h3>
                <div className="exports">
                  <button className="btn tiny" disabled={!!exporting}
                    onClick={() => doExport("influence", "png")}>PNG</button>
                  <button className="btn tiny" disabled={!!exporting}
                    onClick={() => doExport("influence", "svg")}>SVG</button>
                  <span className="sep" />
                  <button className="btn tiny"
                    onClick={() => downloadCsv("q1-influence.csv", result.influence)}>CSV</button>
                </div>
              </div>
              <Plot fig={result.figures.influence} />
            </div>

            <details className="card">
              <summary>folds</summary>
              <Table rows={result.folds} cols={["fold", "n_train", "train_start", "train_end", "n_test", "test_start", "test_end"]} />
              <button className="btn tiny" onClick={() => downloadCsv("q1-folds.csv", result.folds)}>folds.csv</button>
            </details>
            <details className="card">
              <summary>audit</summary>
              <Table rows={result.audit} cols={["item", "value"]} />
              <button className="btn tiny" onClick={() => downloadCsv("q1-audit.csv", result.audit)}>audit.csv</button>
            </details>
            <div className="card">
              <div className="card-head">
                <h3>Data</h3>
                <div className="exports">
                  <button className="btn tiny"
                    onClick={() => downloadCsv("q1-residuals.csv", result.residuals)}>residuals.csv</button>
                </div>
              </div>
              <p>Per-meeting residuals ({result.residuals.length} rows) — the full estimation output.</p>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
