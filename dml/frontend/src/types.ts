export type Row = Record<string, unknown>;

export interface PlotFigure {
  data: unknown[];
  layout: Record<string, unknown>;
}

export interface EstimateRequest {
  outcome: string;
  treatment: string;
  outcome_ticker: string;
  features: string[];
  nuisance: string;
  alpha: number | null;
  min_train: number | null;
  test_size: number | null;
  step: number | null;
  embargo: number;
  se: string;
  hac_lag: number;
  n_boot: number;
  seed: number;
  level: number;
}

export interface EstimateResponse {
  params: Row;
  metrics: Row;
  estimates: Row[];
  audit: Row[];
  folds: Row[];
  residuals: Row[];
  influence: Row[];
  figures: Record<string, PlotFigure>;
}

export interface Q1Config {
  surprises: string[];
  outcomes: string[];
  nuisances: string[];
  se_types: string[];
  defaults: EstimateRequest;
  features: Record<string, string[]>;
  feature_info: Record<string, Record<string, string>>;
  tickers: string[];
}

export interface RunDetail {
  manifest: Row;
  tables: Record<string, Row[]>;
  figures: Record<string, PlotFigure>;
}

export type ExportKind = "residualized_fit" | "influence" | "coefficients";
export type ExportFormat = "png" | "svg";
