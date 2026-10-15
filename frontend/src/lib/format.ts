export function fmt(v: unknown, digits = 4): string {
  if (v === null || v === undefined || v === "") return "–";
  if (typeof v === "number") {
    if (!Number.isFinite(v)) return "–";
    if (Math.abs(v) >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 1 });
    return v.toFixed(digits).replace(/\.?0+$/, (m) => (m.startsWith(".") ? "" : m)) || "0";
  }
  if (typeof v === "boolean") return v ? "yes" : "no";
  return String(v);
}

export function fmtNum(v: number | null | undefined, digits = 4): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "–";
  return v.toFixed(digits);
}

export function fmtPct(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "–";
  return `${v.toFixed(digits)} %`;
}

export function fmtDate(iso: string | null | undefined): string {
  if (!iso) return "–";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
}

export const SERIES_COLORS = ["#2f8cff", "#26c6f9", "#9b7dff", "#2ad18d", "#ff9f43", "#ff5c7a", "#f5b942", "#7fe3ff"];

export const MODEL_LABELS: Record<string, string> = {
  linear: "Linear Regression",
  polynomial: "Polynomial Regression",
  random_forest: "Random Forest",
  svr: "SVR",
  xgboost: "XGBoost",
  mlp: "Neural Network (MLP)",
};

export const FEATURE_LABELS: Record<string, string> = {
  material: "Material",
  composition: "Composition",
  pore_size_class: "Pore size class",
  material_pair: "Material pair",
  lubricant: "Lubricant",
  normal_load_n: "Normal load (N)",
  sliding_speed_m_s: "Sliding speed (m/s)",
  temperature_c: "Temperature (°C)",
  time_s: "Sliding time (s)",
  sliding_distance_m: "Sliding distance (m)",
  rpm: "Rotational speed (rpm)",
  track_diameter_mm: "Track diameter (mm)",
  friction_force_n: "Friction force (N)",
  cof: "COF (–)",
  wear_volume_mm3: "Wear volume (mm³)",
  wear_mass_mg: "Wear mass (mg)",
  wear_rate: "Wear rate",
  test_run_id: "Test run",
};

export const featureLabel = (f: string) => FEATURE_LABELS[f] || f;
