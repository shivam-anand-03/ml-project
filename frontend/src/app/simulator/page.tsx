"use client";

import { useMutation } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { apiPost, errorMessage } from "@/lib/api";
import { Alert, Badge, Button, DataTable, EmptyState, ErrorBox, Field, KeyValue, Loading, PageHeader, Panel } from "@/components/ui";
import { Bars, ChartBox, MultiLine } from "@/components/charts";
import { featureLabel, fmtNum, MODEL_LABELS, SERIES_COLORS } from "@/lib/format";
import { PredictSchema, usePredictSchema } from "@/lib/hooks";

interface SimResult {
  target: string; unit: string; model: { id: number; algorithm: string; version_label: string };
  baseline: { inputs: Record<string, unknown>; value: number; warnings: string[] };
  sensitivity: { feature: string; label: string; unit: string; low_input: number; high_input: number; low_value: number; high_value: number; delta: number; slope_per_unit: number }[];
  comparisons: Record<string, { level: string; value: number; delta_vs_baseline: number }[]>;
  sweep?: { feature: string; label: string; unit: string; by: string | null; series: { level: string | null; points: { x: number; y: number; extrapolated: boolean }[] }[]; experimental_range: [number, number] };
  note: string;
}

export default function SimulatorPage() {
  const schema = usePredictSchema();
  if (schema.isLoading) return <Loading />;
  if (schema.isError) return <ErrorBox error={schema.error} />;
  const s = schema.data!;
  const targets = Object.entries(s.targets).filter(([, t]) => t.available).map(([k]) => k);
  if (!targets.length) return <><PageHeader title="What-If Simulator" /><EmptyState title="Model not trained">Train a model first; the simulator recalculates predictions with the trained model.</EmptyState></>;
  return <Simulator schema={s} targets={targets} />;
}

function Simulator({ schema, targets }: { schema: PredictSchema; targets: string[] }) {
  const [target, setTarget] = useState(targets[0]);
  const numeric = schema.inputs.filter((i) => i.dtype === "number");
  const categorical = schema.inputs.filter((i) => i.dtype === "category");
  const [vals, setVals] = useState<Record<string, string>>({});
  const [baseline, setBaseline] = useState<Record<string, string> | null>(null);
  const [sweepFeature, setSweepFeature] = useState(numeric[0]?.name || "");
  const [sweepBy, setSweepBy] = useState(categorical[0]?.name || "");
  const [scenarios, setScenarios] = useState<Record<string, string>[]>([]);
  useEffect(() => {
    const init: Record<string, string> = {};
    for (const i of schema.inputs) init[i.name] = i.dtype === "number" ? String(Number((((i.min ?? 0) + (i.max ?? 0)) / 2).toFixed(2))) : Object.keys(i.levels || {})[0] || "";
    setVals(init);
    setBaseline(init);
  }, [schema]);
  const toInputs = (v: Record<string, string>) => Object.fromEntries(schema.inputs.map((i) => [i.name, i.dtype === "number" ? Number(v[i.name]) : v[i.name]]));
  const sim = useMutation({ mutationFn: (v: Record<string, string>) => apiPost<SimResult>("/api/simulate", { target, base_inputs: toInputs(v), sweep: sweepFeature ? { feature: sweepFeature, steps: 60, by: sweepBy || null } : null }) });
  const base = useMutation({ mutationFn: (v: Record<string, string>) => apiPost<SimResult>("/api/simulate", { target, base_inputs: toInputs(v) }) });
  const cmp = useMutation({ mutationFn: (sc: Record<string, string>[]) => apiPost<{ unit: string; scenarios: { index: number; ok: boolean; inputs: Record<string, unknown>; value?: number; delta_vs_first?: number; pct_vs_first?: number | null; warnings?: string[]; error?: string }[] }>("/api/simulate/compare", { target, scenarios: sc.map(toInputs) }) });
  // debounce recalculation when inputs change
  const key = JSON.stringify({ vals, sweepFeature, sweepBy, target });
  useEffect(() => { if (!Object.keys(vals).length) return; const t = setTimeout(() => sim.mutate(vals), 250); return () => clearTimeout(t); }, [key]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { if (baseline) base.mutate(baseline); }, [baseline, target]); // eslint-disable-line react-hooks/exhaustive-deps
  const r = sim.data, b = base.data;
  const unitLabel = target === "cof" ? "COF (–)" : `Wear rate (${r?.unit || ""})`;
  const delta = r && b ? r.baseline.value - b.baseline.value : null;
  const sweepSeries = useMemo(() => (r?.sweep ? r.sweep.series.map((s, i) => ({ name: s.level ?? "prediction", color: SERIES_COLORS[i % SERIES_COLORS.length], points: s.points.map((p) => ({ x: p.x, y: p.y })) })) : []), [r]);

  return (
    <>
      <PageHeader title="What-If Simulator" subtitle="Move the controls and the trained model recalculates. Everything shown here is a model prediction — not an experimentally measured outcome. Inputs are limited to the experimental domain (±10 % of the training range)." actions={targets.length > 1 && <select className="input" value={target} onChange={(e) => setTarget(e.target.value)}>{targets.map((t) => <option key={t} value={t}>{t}</option>)}</select>} />
      <div className="grid gap-4 xl:grid-cols-[360px_1fr]">
        <div className="space-y-4">
          <Panel title="Operating conditions" glow actions={<Button size="sm" onClick={() => setBaseline({ ...vals })}>Set as baseline</Button>}>
            <div className="space-y-4">
              {numeric.map((i) => {
                const lo = i.min ?? 0, hi = i.max ?? 1, rng = hi - lo || 1, margin = 0.1 * rng;
                return (
                  <Field key={i.name} label={`${i.label} (${i.unit})`} hint={`experimental range ${fmtNum(lo, 2)} – ${fmtNum(hi, 2)} ${i.unit}`}>
                    <div className="flex items-center gap-3">
                      <input type="range" min={lo - margin} max={hi + margin} step={rng / 200} value={Number(vals[i.name] ?? lo)} onChange={(e) => setVals({ ...vals, [i.name]: e.target.value })} />
                      <input className="input" style={{ width: 110 }} type="number" step="any" value={vals[i.name] ?? ""} onChange={(e) => setVals({ ...vals, [i.name]: e.target.value })} />
                    </div>
                  </Field>
                );
              })}
              {categorical.map((i) => (
                <Field key={i.name} label={i.label}>
                  <select className="input" value={vals[i.name] ?? ""} onChange={(e) => setVals({ ...vals, [i.name]: e.target.value })}>{Object.keys(i.levels || {}).map((lv) => <option key={lv}>{lv}</option>)}</select>
                </Field>
              ))}
              {Object.values(schema.fixed_conditions).map((c) => (
                <Field key={c.name} label={`${c.label}${c.unit ? ` (${c.unit})` : ""}`} hint={c.value === null ? c.note : "fixed in all experiments — not adjustable"}>
                  <input className="input" disabled value={c.value === null ? (c.levels ? c.levels.join(", ") : `${fmtNum(c.min, 3)} – ${fmtNum(c.max, 3)}`) : String(c.value)} />
                </Field>
              ))}
              {schema.not_recorded.map((n) => <Field key={n.name} label={`${n.label}${n.unit ? ` (${n.unit})` : ""}`} hint="not recorded in the dataset — no control possible"><input className="input" disabled value="not recorded" /></Field>)}
            </div>
            <Button className="mt-4 w-full" onClick={() => setScenarios([...scenarios, { ...vals }])}>Add current conditions to comparison</Button>
          </Panel>
          <Panel title="Sweep settings">
            <Field label="Sweep variable"><select className="input" value={sweepFeature} onChange={(e) => setSweepFeature(e.target.value)}><option value="">none</option>{numeric.map((i) => <option key={i.name} value={i.name}>{i.label}</option>)}</select></Field>
            <Field label="One curve per" className="mt-2"><select className="input" value={sweepBy} onChange={(e) => setSweepBy(e.target.value)}><option value="">baseline only</option>{categorical.map((i) => <option key={i.name} value={i.name}>{i.label}</option>)}</select></Field>
          </Panel>
        </div>
        <div className="space-y-4">
          {sim.isError && <Alert tone="danger" title="Prediction refused">{errorMessage(sim.error)}</Alert>}
          {r && (
            <>
              {r.baseline.warnings.length > 0 && <Alert tone="warn">{r.baseline.warnings.join(" ")}</Alert>}
              <div className="grid gap-4 md:grid-cols-4">
                <Panel title={`Predicted ${target === "cof" ? "COF" : "wear rate"}`} glow><div className="text-3xl font-bold text-[var(--predicted)]">{fmtNum(r.baseline.value)}</div><Badge tone="predicted" className="mt-2">model prediction</Badge></Panel>
                <Panel title="Baseline"><div className="text-2xl font-bold text-white">{b ? fmtNum(b.baseline.value) : "–"}</div><div className="mt-1 text-xs text-[var(--muted)]">{b ? Object.entries(b.baseline.inputs).map(([k, v]) => `${featureLabel(k)}: ${typeof v === "number" ? fmtNum(v, 2) : v}`).join(" · ") : ""}</div></Panel>
                <Panel title="Change vs baseline"><div className={`text-2xl font-bold ${delta != null && delta > 0 ? "text-[var(--danger)]" : "text-[var(--ok)]"}`}>{delta == null ? "–" : `${delta >= 0 ? "+" : ""}${fmtNum(delta)}`}</div><div className="mt-1 text-xs text-[var(--muted)]">{delta != null && b ? `${((100 * delta) / Math.abs(b.baseline.value || 1)).toFixed(1)} %` : ""}</div></Panel>
                <Panel title={target === "cof" ? "Wear rate" : "COF"}><div className="text-lg font-bold text-[var(--muted)]">{targets.includes(target === "cof" ? "wear_rate" : "cof") ? "switch target above" : "unavailable"}</div><div className="mt-1 text-xs text-[var(--muted)]">{targets.includes("wear_rate") ? "" : "no wear measurements in the training data"}</div></Panel>
              </div>
              <div className="grid gap-4 lg:grid-cols-2">
                <ChartBox title="Input sensitivity" subtitle="Change in the prediction when each numeric input moves ±10 % of its experimental range around the current point" height={220}>
                  {r.sensitivity.length ? <Bars data={r.sensitivity.map((s) => ({ name: `${s.label} (${fmtNum(s.low_input, 1)} → ${fmtNum(s.high_input, 1)} ${s.unit})`, delta: s.delta }))} xKey="name" bars={[{ key: "delta", name: "Δ prediction" }]} horizontal /> : <div className="text-sm text-[var(--muted)]">No numeric inputs.</div>}
                </ChartBox>
                {Object.entries(r.comparisons).map(([f, levels]) => (
                  <ChartBox key={f} title={`Comparison between ${featureLabel(f).toLowerCase()} levels at the current conditions`} subtitle="Predicted values; the experimental data decide the ranking — nothing is assumed" height={220}>
                    <Bars data={levels.map((l) => ({ level: l.level, value: l.value }))} xKey="level" bars={[{ key: "value", name: unitLabel }]} />
                  </ChartBox>
                ))}
              </div>
              {r.sweep && (
                <ChartBox title={`Predicted ${unitLabel} across ${r.sweep.label.toLowerCase()}`} subtitle={`Experimental range ${fmtNum(r.sweep.experimental_range[0], 2)} – ${fmtNum(r.sweep.experimental_range[1], 2)} ${r.sweep.unit}; points beyond it are extrapolations`} height={340}>
                  <MultiLine xLabel={`${r.sweep.label} (${r.sweep.unit})`} yLabel={unitLabel} series={sweepSeries} />
                </ChartBox>
              )}
            </>
          )}
          <Panel title="Side-by-side scenario comparison" subtitle="Add conditions from the panel on the left; the first scenario is the reference" actions={<><Button size="sm" disabled={scenarios.length < 1} onClick={() => cmp.mutate(scenarios)} loading={cmp.isPending}>Compare</Button><Button size="sm" onClick={() => setScenarios([])}>Clear</Button></>}>
            {scenarios.length === 0 && <div className="text-sm text-[var(--muted)]">No scenarios added yet.</div>}
            {cmp.isError && <Alert tone="danger">{errorMessage(cmp.error)}</Alert>}
            {cmp.data && (
              <DataTable columns={["#", ...schema.inputs.map((i) => i.label), "predicted", "Δ vs first", "% vs first", "status"]} rows={cmp.data.scenarios.map((s) => ({ "#": s.index + 1, ...Object.fromEntries(schema.inputs.map((i) => [i.label, s.inputs?.[i.name]])), predicted: s.ok ? s.value : null, "Δ vs first": s.ok ? s.delta_vs_first : null, "% vs first": s.ok && s.pct_vs_first != null ? `${s.pct_vs_first.toFixed(1)} %` : null, status: s.ok ? (s.warnings?.length ? s.warnings.join(" ") : "in domain") : s.error }))} />
            )}
            {!cmp.data && scenarios.length > 0 && <KeyValue items={scenarios.map((s, i) => [`Scenario ${i + 1}`, Object.entries(s).map(([k, v]) => `${featureLabel(k)}=${v}`).join(", ")])} />}
          </Panel>
          {r && <div className="text-xs text-[var(--muted)]">{r.note} Model: {MODEL_LABELS[r.model.algorithm] || r.model.algorithm} ({r.model.version_label}).</div>}
        </div>
      </div>
    </>
  );
}
