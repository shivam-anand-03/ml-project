"use client";

import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { apiGet } from "@/lib/api";
import { Alert, Badge, DataTable, EmptyState, ErrorBox, Field, KeyValue, Loading, PageHeader, Panel, Tabs } from "@/components/ui";
import { Bars, ChartBox, Heatmap, Histogram, MultiLine, ScatterXY } from "@/components/charts";
import { featureLabel, fmt, fmtNum, SERIES_COLORS } from "@/lib/format";
import { useDatasets, VersionSummary } from "@/lib/hooks";

type Tab = "time" | "features" | "materials" | "correlation" | "distributions" | "trends";

export default function AnalysisPage() {
  const datasets = useDatasets();
  const [dsId, setDsId] = useState<number | null>(null);
  const [vid, setVid] = useState<number | null>(null);
  const [target, setTarget] = useState<"cof" | "wear_rate">("cof");
  const [tab, setTab] = useState<Tab>("time");
  useEffect(() => { if (dsId === null && datasets.data?.length) setDsId(datasets.data[0].id); }, [datasets.data, dsId]);
  const versions = useQuery({ queryKey: ["versions", dsId], queryFn: () => apiGet<VersionSummary[]>(`/api/datasets/${dsId}/versions`), enabled: dsId !== null });
  useEffect(() => { if (versions.data?.length) setVid(versions.data[versions.data.length - 1].id); }, [versions.data]);
  const version = versions.data?.find((v) => v.id === vid);
  const targetAvailable = version?.feature_summary?.targets?.[target]?.available;

  if (datasets.isLoading) return <Loading />;
  if (!datasets.data?.length) return <EmptyState title="No dataset">Upload or import a dataset first.</EmptyState>;
  return (
    <>
      <PageHeader title="Experimental Analysis" subtitle="Scientific visualisations of the measured observations. Everything on this page is measured data; model predictions are never mixed in." />
      <Panel className="mb-4">
        <div className="grid gap-3 md:grid-cols-3">
          <Field label="Dataset"><select className="input" value={dsId ?? ""} onChange={(e) => { setDsId(Number(e.target.value)); setVid(null); }}>{datasets.data.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}</select></Field>
          <Field label="Version"><select className="input" value={vid ?? ""} onChange={(e) => setVid(Number(e.target.value))}>{versions.data?.map((v) => <option key={v.id} value={v.id}>v{v.version_number} · {v.kind} · {v.usable_record_count} usable rows</option>)}</select></Field>
          <Field label="Target"><select className="input" value={target} onChange={(e) => setTarget(e.target.value as "cof" | "wear_rate")}><option value="cof">Coefficient of friction</option><option value="wear_rate">Wear rate</option></select></Field>
        </div>
      </Panel>
      {version && !targetAvailable && <Alert tone="warn" title={`${target === "cof" ? "COF" : "Wear rate"} is not available in this dataset version`} className="mb-4">No measurements of this target exist in the uploaded files, so no chart can be drawn for it. {target === "wear_rate" && "The Kolev 2023 dataset contains friction force only — no wear volume, mass or rate."}</Alert>}
      {vid && targetAvailable && (
        <>
          <div className="mb-4"><Tabs tabs={[{ key: "time", label: "vs time" }, { key: "features", label: "vs load / speed / temperature" }, { key: "materials", label: "Material & lubricant comparison" }, { key: "correlation", label: "Correlation heatmap" }, { key: "distributions", label: "Distributions" }, { key: "trends", label: "Trends & interactions" }]} value={tab} onChange={setTab} /></div>
          {tab === "time" && <TimeTab vid={vid} target={target} />}
          {tab === "features" && <FeaturesTab vid={vid} target={target} />}
          {tab === "materials" && <MaterialsTab vid={vid} target={target} />}
          {tab === "correlation" && <CorrelationTab vid={vid} target={target} />}
          {tab === "distributions" && <DistributionsTab vid={vid} target={target} />}
          {tab === "trends" && <TrendsTab vid={vid} target={target} />}
        </>
      )}
    </>
  );
}

const TL = { cof: "COF (–)", wear_rate: "Wear rate" };

function TimeTab({ vid, target }: { vid: number; target: string }) {
  const [groupBy, setGroupBy] = useState("material");
  const q = useQuery({ queryKey: ["tvt", vid, target, groupBy], queryFn: () => apiGet<{ available: boolean; reason?: string; groups: { group: string; n_runs: number; points: { t: number; mean: number; std: number | null; min: number; max: number }[] }[]; runs: { test_run_id: string; group: string; points: { t: number; y: number }[] }[] }>(`/api/analysis/${vid}/target-vs-time`, { target, group_by: groupBy }) });
  if (q.isLoading) return <Loading />;
  if (q.isError) return <ErrorBox error={q.error} />;
  const d = q.data!;
  if (!d.available) return <Alert tone="warn">{d.reason}</Alert>;
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3"><span className="text-sm text-[var(--muted)]">Group by</span><select className="input" style={{ width: 200 }} value={groupBy} onChange={(e) => setGroupBy(e.target.value)}>{["material", "composition", "pore_size_class", "lubricant", "material_pair"].map((g) => <option key={g} value={g}>{featureLabel(g)}</option>)}</select></div>
      <ChartBox title={`Measured ${TL[target as keyof typeof TL]} vs sliding time — mean ± SD across replicate runs`} subtitle="Shaded band = ±1 standard deviation between replicate runs at the same time" height={380}>
        <MultiLine xLabel="Sliding time (s)" yLabel={TL[target as keyof typeof TL]} series={d.groups.map((g, i) => ({ name: `${g.group} (${g.n_runs} runs)`, color: SERIES_COLORS[i % SERIES_COLORS.length], points: g.points.map((p) => ({ x: p.t, y: p.mean })) }))}
          bandSeries={d.groups.map((g, i) => ({ name: g.group, color: SERIES_COLORS[i % SERIES_COLORS.length], points: g.points.map((p) => ({ x: p.t, lo: p.mean - (p.std || 0), hi: p.mean + (p.std || 0) })) }))} />
      </ChartBox>
      <ChartBox title="Individual test runs" subtitle="Each line is one measured run (down-sampled for display)" height={380}>
        <MultiLine xLabel="Sliding time (s)" yLabel={TL[target as keyof typeof TL]} series={d.runs.map((r) => ({ name: `${r.test_run_id} (${r.group})`, color: SERIES_COLORS[d.groups.findIndex((g) => g.group === r.group) % SERIES_COLORS.length], points: r.points.map((p) => ({ x: p.t, y: p.y })) }))} />
      </ChartBox>
    </div>
  );
}

function FeaturesTab({ vid, target }: { vid: number; target: string }) {
  const feats = ["normal_load_n", "sliding_speed_m_s", "temperature_c", "sliding_distance_m", "time_s"];
  return (
    <div className="grid gap-4 xl:grid-cols-2">
      {feats.map((f) => <FeatureChart key={f} vid={vid} target={target} feature={f} />)}
    </div>
  );
}

function FeatureChart({ vid, target, feature }: { vid: number; target: string; feature: string }) {
  const q = useQuery({ queryKey: ["tvf", vid, target, feature], queryFn: () => apiGet<{ available: boolean; varied: boolean; reason?: string; fixed_value?: number; feature_unit: string; label: string; points?: { x: number; y: number; group: string | null }[]; trend?: { group: string; points: { x: number; mean: number }[] }[]; spearman?: { rho: number; p_value: number; n: number; caveat: string } }>(`/api/analysis/${vid}/target-vs-feature`, { target, feature }) });
  if (q.isLoading) return <Loading />;
  if (q.isError) return <ErrorBox error={q.error} />;
  const d = q.data!;
  const title = `${TL[target as keyof typeof TL]} vs ${d.label.toLowerCase()}`;
  if (!d.available || !d.varied) {
    return (
      <Panel title={title}>
        <Alert tone="info" title={!d.available ? "Not recorded" : `Held constant at ${fmt(d.fixed_value)} ${d.feature_unit}`}>{d.reason}</Alert>
      </Panel>
    );
  }
  return (
    <ChartBox title={title} subtitle={d.spearman ? `Spearman ρ = ${d.spearman.rho.toFixed(3)} (n = ${d.spearman.n}); ${d.spearman.caveat}` : undefined} height={320}>
      <ScatterXY points={d.points!} xLabel={`${d.label} (${d.feature_unit})`} yLabel={TL[target as keyof typeof TL]} groups />
    </ChartBox>
  );
}

function MaterialsTab({ vid, target }: { vid: number; target: string }) {
  const [groupBy, setGroupBy] = useState("material");
  const q = useQuery({ queryKey: ["mc", vid, target, groupBy], queryFn: () => apiGet<{ available: boolean; reason?: string; groups: { group: string; n_rows: number; n_runs: number; mean: number; std: number; min: number; max: number; steady_mean: number | null; steady_std_between_runs: number | null; runs: { test_run_id: string; mean: number; steady_mean: number | null; max: number }[]; attributes: Record<string, unknown> }[]; test: { name: string; statistic: number; p_value: number; interpretation: string; caveat: string; n_runs: number } | null; steady_state_definition: string }>(`/api/analysis/${vid}/material-comparison`, { target, group_by: groupBy }) });
  const lub = useQuery({ queryKey: ["tvf", vid, target, "lubricant"], queryFn: () => apiGet<{ available: boolean; varied: boolean; reason?: string; levels?: { level: string; n: number; mean: number; std: number }[] }>(`/api/analysis/${vid}/target-vs-feature`, { target, feature: "lubricant" }) });
  if (q.isLoading) return <Loading />;
  if (q.isError) return <ErrorBox error={q.error} />;
  const d = q.data!;
  if (!d.available) return <Alert tone="warn">{d.reason}</Alert>;
  const rows = d.groups.map((g) => ({ group: g.group, mean: g.mean, steady: g.steady_mean ?? undefined, std_between_runs: g.steady_std_between_runs ?? undefined, n_runs: g.n_runs }));
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3"><span className="text-sm text-[var(--muted)]">Compare by</span><select className="input" style={{ width: 200 }} value={groupBy} onChange={(e) => setGroupBy(e.target.value)}>{["material", "composition", "pore_size_class", "lubricant", "material_pair"].map((g) => <option key={g} value={g}>{featureLabel(g)}</option>)}</select></div>
      <div className="grid gap-4 lg:grid-cols-2">
        <ChartBox title={`Measured ${TL[target as keyof typeof TL]} by ${featureLabel(groupBy).toLowerCase()}`} subtitle={`Mean over the whole run vs steady-state mean (${d.steady_state_definition})`} height={300}>
          <Bars data={rows} xKey="group" bars={[{ key: "mean", name: "mean (whole run)" }, { key: "steady", name: "steady-state mean" }]} />
        </ChartBox>
        <Panel title="Between-group test" subtitle="Computed on per-run steady-state means">
          {d.test ? (
            <>
              <KeyValue items={[["Test", d.test.name], ["Statistic", fmtNum(d.test.statistic, 3)], ["p-value", d.test.p_value < 0.001 ? "< 0.001" : fmtNum(d.test.p_value, 4)], ["Runs used", d.test.n_runs]]} />
              <Alert tone={d.test.p_value < 0.05 ? "ok" : "warn"} className="mt-3">{d.test.interpretation} {d.test.caveat}</Alert>
            </>
          ) : <Alert tone="info">Not enough replicate runs for a between-group test; differences cannot be called significant.</Alert>}
        </Panel>
      </div>
      <Panel title="Per-group statistics (measured)">
        <DataTable columns={["group", "n_runs", "n_rows", "mean", "std", "min", "max", "steady_mean", "steady_std_between_runs"]} rows={d.groups as unknown as Record<string, unknown>[]} maxHeight={260} />
      </Panel>
      <Panel title="Replicate runs" subtitle="Run-to-run spread within each group">
        <DataTable columns={["group", "test_run_id", "mean", "steady_mean", "max"]} rows={d.groups.flatMap((g) => g.runs.map((r) => ({ group: g.group, ...r })))} maxHeight={300} />
      </Panel>
      <Panel title="Lubricant-wise performance">
        {lub.data?.available && lub.data.varied ? (
          <DataTable columns={["level", "n", "mean", "std"]} rows={lub.data.levels as unknown as Record<string, unknown>[]} />
        ) : <Alert tone="info">{lub.data?.reason || "Lubricant was not varied in this dataset — no lubricant comparison is possible."}</Alert>}
      </Panel>
    </div>
  );
}

function CorrelationTab({ vid, target }: { vid: number; target: string }) {
  const q = useQuery({ queryKey: ["corr", vid, target], queryFn: () => apiGet<{ available: boolean; reason?: string; features: string[]; labels: string[]; matrix: (number | null)[][]; constant: string[]; caveat: string }>(`/api/analysis/${vid}/correlation`, { target }) });
  if (q.isLoading) return <Loading />;
  if (q.isError) return <ErrorBox error={q.error} />;
  const d = q.data!;
  if (!d.available) return <Alert tone="warn">{d.reason}</Alert>;
  return (
    <Panel title="Spearman correlation heatmap" subtitle={d.caveat}>
      <Heatmap labels={d.labels} matrix={d.matrix} />
      {d.constant.length > 0 && <div className="mt-3 text-sm text-[var(--muted)]">Excluded (constant in this dataset): {d.constant.map(featureLabel).join(", ")}</div>}
    </Panel>
  );
}

function DistributionsTab({ vid, target }: { vid: number; target: string }) {
  const q = useQuery({ queryKey: ["dist2", vid, target], queryFn: () => apiGet<{ features: { feature: string; label: string; unit: string; constant: boolean; value?: number; bins?: { x0: number; x1: number; count: number }[]; mean?: number; std?: number }[]; by_material: { material: string; bins: { x0: number; x1: number; count: number }[] }[] }>(`/api/analysis/${vid}/distributions`, { target }) });
  if (q.isLoading) return <Loading />;
  if (q.isError) return <ErrorBox error={q.error} />;
  const d = q.data!;
  return (
    <div className="grid gap-4 md:grid-cols-2">
      {d.features.filter((f) => !f.constant && f.bins).map((f, i) => (
        <ChartBox key={f.feature} title={`${f.label} (${f.unit})`} subtitle={`mean ${fmt(f.mean)} · SD ${fmt(f.std)}`} height={220}><Histogram bins={f.bins!} color={SERIES_COLORS[i % SERIES_COLORS.length]} /></ChartBox>
      ))}
      {d.by_material.map((m, i) => (
        <ChartBox key={m.material} title={`${TL[target as keyof typeof TL]} distribution — ${m.material}`} height={220}><Histogram bins={m.bins} color={SERIES_COLORS[i % SERIES_COLORS.length]} /></ChartBox>
      ))}
      {d.features.filter((f) => f.constant).length > 0 && <Panel title="Constant in this dataset"><KeyValue items={d.features.filter((f) => f.constant).map((f) => [f.label, `${fmt(f.value)} ${f.unit}`])} /></Panel>}
    </div>
  );
}

function TrendsTab({ vid, target }: { vid: number; target: string }) {
  const q = useQuery({ queryKey: ["trends", vid, target], queryFn: () => apiGet<{ numeric: { feature: string; label: string; status: string; statement: string; spearman_rho?: number; p_value?: number; per_run?: { run: string; rho: number | null }[] }[]; categorical: { feature: string; label: string; status: string; statement: string; test?: { p_value: number } | null; groups?: { group: string; steady_mean: number | null; n_runs: number }[] }[]; disclaimer: string }>(`/api/analysis/${vid}/trends`, { target }) });
  if (q.isLoading) return <Loading />;
  if (q.isError) return <ErrorBox error={q.error} />;
  const d = q.data!;
  const tone = (s: string) => (s === "varied" ? "ok" : s === "constant" ? "warn" : "muted");
  return (
    <div className="space-y-4">
      <Alert tone="info">{d.disclaimer}</Alert>
      <Panel title="Operating parameters">
        <ul className="space-y-2 text-sm">
          {d.numeric.map((n) => <li key={n.feature} className="flex gap-3"><Badge tone={tone(n.status)}>{n.status.replace("_", " ")}</Badge><span><span className="font-medium text-white">{n.label}:</span> <span className="text-[var(--muted)]">{n.statement}</span></span></li>)}
        </ul>
      </Panel>
      <Panel title="Categorical factors">
        <ul className="space-y-2 text-sm">
          {d.categorical.map((c) => <li key={c.feature} className="flex gap-3"><Badge tone={tone(c.status)}>{c.status.replace("_", " ")}</Badge><span><span className="font-medium text-white">{c.label}:</span> <span className="text-[var(--muted)]">{c.statement}</span>{c.groups && <span className="text-[var(--muted)]"> Steady-state means: {c.groups.map((g) => `${g.group} ${fmtNum(g.steady_mean, 3)} (${g.n_runs} runs)`).join("; ")}.</span>}</span></li>)}
        </ul>
      </Panel>
    </div>
  );
}
