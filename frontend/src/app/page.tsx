"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { useState } from "react";
import { apiGet } from "@/lib/api";
import { Alert, Badge, Button, EmptyState, ErrorBox, Field, KeyValue, Loading, PageHeader, Panel, Stat } from "@/components/ui";
import { fmt, fmtDate, fmtNum, MODEL_LABELS } from "@/lib/format";
import { useDatasets, ModelSummary } from "@/lib/hooks";
import { ChartBox, MultiLine } from "@/components/charts";
import { SERIES_COLORS } from "@/lib/format";

interface Overview {
  n_datasets: number; n_versions: number; n_models: number; n_training_runs: number; n_predictions: number; n_validations: number; n_records_total: number;
  dataset: { id: number; name: string; source_type: string } | null;
  version: { id: number; version_number: number; kind: string; record_count: number; usable_record_count: number; feature_summary: { n_runs: number; n_materials: number } } | null;
  quality?: { score: number | null; summary: Record<string, unknown> | null; flags: Record<string, number> | null };
  data: { n_rows: number; n_usable: number; n_runs: number; materials: string[]; lubricants: string[]; material_pairs: string[]; ranges: Record<string, { min: number | null; max: number | null; unit: string; label: string; varies: boolean; not_recorded?: boolean }>; targets: Record<string, { available: boolean; n: number; mean: number | null }> } | null;
  filtered: { n_rows: number; n_runs: number; cof_mean: number | null; cof_min: number | null; cof_max: number | null; wear_rate_mean: number | null; materials: string[] } | null;
  models: Record<string, { label: string; active: ModelSummary | null; is_production: boolean; reason: string | null }>;
  recent_predictions: { id: number; inputs: Record<string, unknown>; predicted_cof: number | null; predicted_wear_rate: number | null; in_domain: boolean; source: string; created_at: string }[];
  recent_validations: { id: number; experiment_id: string; measured_cof: number | null; predicted_cof: number | null; cof_abs_error: number | null; status: string }[];
  recent_runs: { id: number; target: string; status: string; best_algorithm: string | null; created_at: string }[];
}

export default function OverviewPage() {
  const datasets = useDatasets();
  const [versionId, setVersionId] = useState<number | "">("");
  const [filters, setFilters] = useState({ material: "", lubricant: "", load_min: "", load_max: "", speed_min: "", speed_max: "", temp_min: "", temp_max: "" });
  const q = useQuery({
    queryKey: ["overview", versionId, filters],
    queryFn: () => apiGet<Overview>("/api/dashboard/overview", { dataset_version_id: versionId || undefined, ...filters }),
  });
  const vid = q.data?.version?.id;
  const curves = useQuery({ queryKey: ["overview-curves", vid], queryFn: () => apiGet<{ available: boolean; groups: { group: string; n_runs: number; points: { t: number; mean: number }[] }[] }>(`/api/analysis/${vid}/target-vs-time`, { bins: 60 }), enabled: !!vid });

  if (q.isLoading) return <Loading label="Computing dashboard statistics from stored data…" />;
  if (q.isError) return <ErrorBox error={q.error} />;
  const d = q.data!;
  if (!d.dataset) {
    return (
      <>
        <PageHeader title="Overview" subtitle="Engineering dashboard — every number is computed from the uploaded experimental data." />
        <EmptyState title="No dataset yet" action={<Link href="/datasets" className="btn btn-primary">Go to Datasets</Link>}>
          Upload your Excel workbooks or import the bundled Kolev 2023 pin-on-disk dataset to get started. Model training and predictions require real data.
        </EmptyState>
      </>
    );
  }
  const cof = d.models.cof, wear = d.models.wear_rate;
  const r = d.data?.ranges || {};
  const rangeText = (k: string) => {
    const x = r[k];
    if (!x || x.not_recorded) return "not recorded";
    if (!x.varies) return `${fmt(x.min)} ${x.unit} (fixed)`;
    return `${fmt(x.min)} – ${fmt(x.max)} ${x.unit}`;
  };
  const versionOptions = datasets.data?.flatMap((ds) => (ds.latest_version_id ? [{ id: ds.latest_version_id, label: `${ds.name} — latest (v${ds.latest_version_number}, ${ds.latest_kind})` }] : [])) || [];

  return (
    <>
      <PageHeader title="Overview" subtitle={<>Dataset <span className="text-white">{d.dataset.name}</span> · version {d.version?.version_number} ({d.version?.kind}). All statistics are computed from the stored experimental records.</>}
        actions={<Link href="/predict" className="btn btn-primary">Open AI Predictions</Link>} />

      <div className="mb-6 grid grid-cols-2 gap-3 md:grid-cols-4 xl:grid-cols-6">
        <Stat label="Experimental records" value={d.version?.record_count ?? 0} hint={`${d.n_records_total} rows across all versions`} />
        <Stat label="Usable after preprocessing" value={d.version?.usable_record_count ?? 0} hint={d.quality?.score != null ? `quality score ${d.quality.score}` : undefined} />
        <Stat label="Test runs" value={d.data?.n_runs ?? 0} hint={`${d.data?.materials.length ?? 0} material(s)`} />
        <Stat label="Lubricants" value={d.data?.lubricants.length ?? 0} hint={d.data?.lubricants.join(", ") || "not recorded"} />
        <Stat label="Material pairs" value={d.data?.material_pairs.length || "–"} hint={d.data?.material_pairs.join(", ") || "not recorded"} />
        <Stat label="Predictions logged" value={d.n_predictions} hint={`${d.n_validations} validation record(s)`} />
      </div>

      <div className="mb-6 grid gap-4 lg:grid-cols-3">
        <Panel title="Experimental ranges" subtitle="Operating window covered by the measured data">
          <KeyValue items={[["Normal load", rangeText("normal_load_n")], ["Sliding speed", rangeText("sliding_speed_m_s")], ["Temperature", rangeText("temperature_c")], ["Sliding time", rangeText("time_s")], ["Sliding distance", rangeText("sliding_distance_m")], ["Materials", d.data?.materials.join(", ") || "–"]]} />
        </Panel>
        <ModelCard title="Best validated COF model" m={cof} />
        <ModelCard title="Best validated wear-rate model" m={wear} />
      </div>

      <div className="mb-6 grid gap-4 lg:grid-cols-3">
        <Panel title="Filters" subtitle="Narrow the statistics below to a subset of the measured data" className="lg:col-span-1">
          <div className="grid grid-cols-2 gap-3">
            <Field label="Dataset version" className="col-span-2">
              <select className="input" value={versionId} onChange={(e) => setVersionId(e.target.value ? Number(e.target.value) : "")}>
                <option value="">Latest processed</option>
                {versionOptions.map((o) => <option key={o.id} value={o.id}>{o.label}</option>)}
              </select>
            </Field>
            <Field label="Material" className="col-span-2">
              <select className="input" value={filters.material} onChange={(e) => setFilters({ ...filters, material: e.target.value })}>
                <option value="">All</option>
                {d.data?.materials.map((m) => <option key={m} value={m}>{m}</option>)}
              </select>
            </Field>
            <Field label="Lubricant" className="col-span-2">
              <select className="input" value={filters.lubricant} onChange={(e) => setFilters({ ...filters, lubricant: e.target.value })}>
                <option value="">All</option>
                {d.data?.lubricants.map((m) => <option key={m} value={m}>{m}</option>)}
              </select>
            </Field>
            <Field label="Load min (N)"><input className="input" type="number" value={filters.load_min} onChange={(e) => setFilters({ ...filters, load_min: e.target.value })} /></Field>
            <Field label="Load max (N)"><input className="input" type="number" value={filters.load_max} onChange={(e) => setFilters({ ...filters, load_max: e.target.value })} /></Field>
            <Field label="Speed min (m/s)"><input className="input" type="number" step="0.01" value={filters.speed_min} onChange={(e) => setFilters({ ...filters, speed_min: e.target.value })} /></Field>
            <Field label="Speed max (m/s)"><input className="input" type="number" step="0.01" value={filters.speed_max} onChange={(e) => setFilters({ ...filters, speed_max: e.target.value })} /></Field>
            <Field label="Temp min (°C)"><input className="input" type="number" value={filters.temp_min} onChange={(e) => setFilters({ ...filters, temp_min: e.target.value })} disabled={r.temperature_c?.not_recorded} /></Field>
            <Field label="Temp max (°C)"><input className="input" type="number" value={filters.temp_max} onChange={(e) => setFilters({ ...filters, temp_max: e.target.value })} disabled={r.temperature_c?.not_recorded} /></Field>
          </div>
          <div className="mt-3 flex gap-2">
            <Button size="sm" onClick={() => setFilters({ material: "", lubricant: "", load_min: "", load_max: "", speed_min: "", speed_max: "", temp_min: "", temp_max: "" })}>Reset</Button>
          </div>
          {d.filtered && (
            <div className="mt-4">
              <KeyValue items={[["Rows", d.filtered.n_rows], ["Runs", d.filtered.n_runs], ["Measured COF mean", fmtNum(d.filtered.cof_mean)], ["Measured COF range", `${fmtNum(d.filtered.cof_min, 3)} – ${fmtNum(d.filtered.cof_max, 3)}`], ["Measured wear rate mean", d.filtered.wear_rate_mean == null ? "unavailable" : fmtNum(d.filtered.wear_rate_mean)]]} />
            </div>
          )}
        </Panel>
        <div className="lg:col-span-2">
          <ChartBox title="Measured COF vs sliding time" subtitle="Mean across replicate runs per material (measured data, not predictions)" height={340}>
            {curves.data?.available ? (
              <MultiLine xLabel="Sliding time (s)" yLabel="COF (–)" series={curves.data.groups.map((g, i) => ({ name: `${g.group} (${g.n_runs} runs)`, color: SERIES_COLORS[i % SERIES_COLORS.length], points: g.points.map((p) => ({ x: p.t, y: p.mean })) }))} />
            ) : (
              <div className="flex h-full items-center justify-center text-sm text-[var(--muted)]">{curves.isLoading ? "Loading…" : "No time series in this dataset"}</div>
            )}
          </ChartBox>
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <Panel title="Data quality indicators">
          {d.quality?.summary ? (
            <KeyValue items={[["Completeness", `${fmt(d.quality.summary.completeness_pct)} %`], ["Usable share", `${fmt(d.quality.summary.usable_pct)} %`], ["COF available", d.quality.summary.has_cof ? <Badge tone="ok">yes</Badge> : <Badge tone="danger">no</Badge>], ["Wear rate available", d.quality.summary.has_wear_rate ? <Badge tone="ok">yes</Badge> : <Badge tone="warn">no</Badge>], ...Object.entries(d.quality.flags || {}).slice(0, 6).map(([k, v]) => [k, v] as [string, number])]} />
          ) : <span className="text-sm text-[var(--muted)]">No quality report.</span>}
        </Panel>
        <Panel title="Recent predictions" subtitle="Model outputs (not measurements)">
          {d.recent_predictions.length === 0 && <div className="text-sm text-[var(--muted)]">No predictions yet.</div>}
          <ul className="space-y-2 text-sm">
            {d.recent_predictions.map((p) => (
              <li key={p.id} className="flex items-center justify-between gap-2 border-b border-[var(--border)]/60 pb-2">
                <span className="truncate text-[var(--muted)]">{Object.entries(p.inputs).map(([k, v]) => `${k}=${v}`).join(", ")}</span>
                <span className="shrink-0 font-semibold text-[var(--predicted)]">COF {fmtNum(p.predicted_cof)}</span>
              </li>
            ))}
          </ul>
        </Panel>
        <Panel title="Recent experiments & training">
          <div className="label mb-1">Validation records</div>
          {d.recent_validations.length === 0 && <div className="mb-3 text-sm text-[var(--muted)]">None yet.</div>}
          <ul className="mb-4 space-y-1 text-sm">
            {d.recent_validations.map((v) => (
              <li key={v.id} className="flex justify-between gap-2"><span>{v.experiment_id}</span><span className="text-[var(--muted)]">measured {fmtNum(v.measured_cof)} · predicted {fmtNum(v.predicted_cof)} · <Badge tone={v.status === "approved" ? "ok" : v.status === "rejected" ? "danger" : "warn"}>{v.status}</Badge></span></li>
            ))}
          </ul>
          <div className="label mb-1">Training runs</div>
          <ul className="space-y-1 text-sm">
            {d.recent_runs.map((rn) => (
              <li key={rn.id} className="flex justify-between gap-2"><span>Run #{rn.id} · {rn.target}</span><span className="text-[var(--muted)]">{rn.best_algorithm ? MODEL_LABELS[rn.best_algorithm] || rn.best_algorithm : "–"} · <Badge tone={rn.status === "completed" ? "ok" : rn.status === "failed" ? "danger" : "warn"}>{rn.status}</Badge></span></li>
            ))}
          </ul>
        </Panel>
      </div>
    </>
  );
}

function ModelCard({ title, m }: { title: string; m: { label: string; active: ModelSummary | null; is_production: boolean; reason: string | null } }) {
  if (!m.active) {
    return (
      <Panel title={title}>
        <Alert tone="warn" title="Model not trained">{m.reason}</Alert>
      </Panel>
    );
  }
  const a = m.active;
  const t = a.metrics.test || {}, v = a.metrics.validation || {};
  return (
    <Panel title={title} actions={<Badge tone={m.is_production ? "ok" : "accent"}>{m.is_production ? "production" : "latest best (candidate)"}</Badge>}>
      <div className="mb-2 text-lg font-bold text-white">{MODEL_LABELS[a.algorithm] || a.algorithm} <span className="text-sm font-normal text-[var(--muted)]">{a.version_label}</span></div>
      <KeyValue items={[["Validation R² (CV)", fmtNum(v.r2)], ["Validation RMSE", fmtNum(v.rmse, 5)], ["Test R² (held-out)", fmtNum(t.r2)], ["Test MAE", fmtNum(t.mae, 5)], ["Split", a.split?.strategy || "–"], ["Trained", fmtDate(a.trained_at)]]} />
    </Panel>
  );
}
