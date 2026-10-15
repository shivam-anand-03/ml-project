"use client";

import { Alert, Field, Panel } from "@/components/ui";
import { Bars, ChartBox } from "@/components/charts";
import { fmtNum } from "@/lib/format";
import { PredictSchema } from "@/lib/hooks";

export interface TargetResult {
  target: string; available: boolean; reason?: string; value: number | null; unit: string; inputs_used?: Record<string, unknown>;
  interval?: { low: number; high: number; level: number; method: string; test_coverage: number | null } | null;
  model?: { id: number; algorithm: string; version_label: string; status: string } | null;
  warnings: string[]; in_domain: boolean | null; extrapolation?: string;
  explanation?: { base_value: number; prediction: number; contributions: { feature: string; label: string; value: unknown; unit: string; contribution: number }[]; method: string; error?: string } | null;
}

export interface PredictResult { prediction_id?: number; created_at?: string; inputs: Record<string, unknown>; cof: TargetResult; wear_rate: TargetResult; warnings: string[]; in_domain: boolean; note: string }

/** Builds initial form values at the centre of each numeric range / first categorical level. */
export function initialValues(schema: PredictSchema): Record<string, string> {
  const init: Record<string, string> = {};
  for (const inp of schema.inputs) init[inp.name] = inp.dtype === "number" ? String(Number((((inp.min ?? 0) + (inp.max ?? 0)) / 2).toFixed(2))) : Object.keys(inp.levels || {})[0] || "";
  return init;
}

export function InputForm({ schema, values, onChange }: { schema: PredictSchema; values: Record<string, string>; onChange: (v: Record<string, string>) => void }) {
  return (
    <div className="space-y-3">
      {schema.inputs.map((inp) => (
        <Field key={inp.name} label={`${inp.label}${inp.unit ? ` (${inp.unit})` : ""}`} hint={inp.dtype === "number" ? `Experimental range ${fmtNum(inp.min, 3)} – ${fmtNum(inp.max, 3)} ${inp.unit}` : `${Object.keys(inp.levels || {}).length} level(s) in the data`}>
          {inp.dtype === "number" ? (
            <input className="input" type="number" step="any" value={values[inp.name] ?? ""} onChange={(e) => onChange({ ...values, [inp.name]: e.target.value })} />
          ) : (
            <select className="input" value={values[inp.name] ?? ""} onChange={(e) => onChange({ ...values, [inp.name]: e.target.value })}>
              <option value="">— select —</option>
              {Object.entries(inp.levels || {}).map(([lv, n]) => <option key={lv} value={lv}>{lv} ({n} rows)</option>)}
            </select>
          )}
        </Field>
      ))}
      {Object.values(schema.fixed_conditions).map((c) => (
        <Field key={c.name} label={`${c.label}${c.unit ? ` (${c.unit})` : ""}`} hint={c.value === null ? c.note : "Fixed in all experiments — the model has no evidence for other values"}>
          <input className="input" value={c.value === null ? (c.levels ? c.levels.join(", ") : `${fmtNum(c.min, 3)} – ${fmtNum(c.max, 3)}`) : String(c.value)} disabled />
        </Field>
      ))}
      {schema.not_recorded.map((n) => (
        <Field key={n.name} label={`${n.label}${n.unit ? ` (${n.unit})` : ""}`} hint="Not recorded in the training dataset — cannot be used as an input"><input className="input" value="not recorded" disabled /></Field>
      ))}
    </div>
  );
}

export function toInputs(schema: PredictSchema, values: Record<string, string>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const inp of schema.inputs) {
    const v = values[inp.name];
    if (v === undefined || v === "") continue;
    out[inp.name] = inp.dtype === "number" ? Number(v) : v;
  }
  return out;
}

export function ExplanationPanel({ t }: { t: TargetResult }) {
  if (!t.available) return null;
  const e = t.explanation;
  if (!e || e.error) return <Panel title="Contributing factors"><Alert tone="info">{e?.error || "No explanation computed."}</Alert></Panel>;
  const data = e.contributions.map((c) => ({ name: `${c.label} = ${typeof c.value === "number" ? fmtNum(c.value, 2) : c.value}${c.unit ? " " + c.unit : ""}`, contribution: c.contribution }));
  return (
    <ChartBox title="Contributing factors (SHAP, local explanation)" subtitle={`Base value ${fmtNum(e.base_value)} + contributions = prediction ${fmtNum(e.prediction)}. ${e.method}. Describes the model, not physical causation.`} height={60 + 40 * data.length}>
      <Bars data={data} xKey="name" bars={[{ key: "contribution", name: "SHAP contribution" }]} horizontal />
    </ChartBox>
  );
}
