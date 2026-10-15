"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { Download, Trash2, Upload } from "lucide-react";
import { apiDelete, apiDownload, apiGet, apiPost, apiUpload, errorMessage } from "@/lib/api";
import { Alert, Badge, Button, ConfirmDialog, DataTable, EmptyState, ErrorBox, Field, KeyValue, Loading, PageHeader, Panel, Spinner, Tabs } from "@/components/ui";
import { ChartBox, Heatmap, Histogram } from "@/components/charts";
import { featureLabel, fmt, fmtDate } from "@/lib/format";
import { useDataset, useDatasets, FileView, SheetInspection, VersionSummary } from "@/lib/hooks";

type Tab = "files" | "mapping" | "preview" | "quality" | "versions";

interface FeatureDef { name: string; label: string; unit: string; dtype: string; role: string; description: string }

export default function DatasetsPage() {
  const qc = useQueryClient();
  const datasets = useDatasets();
  const [selected, setSelected] = useState<number | null>(null);
  const [tab, setTab] = useState<Tab>("files");
  const [msg, setMsg] = useState<{ tone: "ok" | "danger" | "warn"; text: string } | null>(null);
  useEffect(() => { if (selected === null && datasets.data?.length) setSelected(datasets.data[0].id); }, [datasets.data, selected]);
  const dataset = useDataset(selected);

  const importBundled = useMutation({
    mutationFn: () => apiPost<{ dataset: { id: number } }>("/api/datasets/import-bundled"),
    onSuccess: (r) => { qc.invalidateQueries({ queryKey: ["datasets"] }); setSelected(r.dataset.id); setTab("versions"); setMsg({ tone: "ok", text: "Bundled Kolev 2023 dataset imported: raw and processed versions created." }); },
    onError: (e) => setMsg({ tone: "danger", text: errorMessage(e) }),
  });
  const [del, setDel] = useState(false);
  const deleteDs = useMutation({
    mutationFn: () => apiDelete(`/api/datasets/${selected}`),
    onSuccess: () => { setDel(false); setSelected(null); qc.invalidateQueries({ queryKey: ["datasets"] }); setMsg({ tone: "ok", text: "Dataset deleted." }); },
    onError: (e) => { setDel(false); setMsg({ tone: "danger", text: errorMessage(e) }); },
  });

  return (
    <>
      <PageHeader title="Datasets" subtitle="Upload Excel / CSV / tribometer exports, inspect worksheets, map columns to standard features, validate, preprocess and version your experimental data. Originals are never overwritten."
        actions={<Button variant="primary" onClick={() => importBundled.mutate()} loading={importBundled.isPending}>Import bundled Kolev 2023 dataset</Button>} />
      {msg && <Alert tone={msg.tone} className="mb-4">{msg.text}</Alert>}
      <div className="grid gap-4 lg:grid-cols-[300px_1fr]">
        <div className="space-y-4">
          <UploadPanel onDone={(id) => { qc.invalidateQueries({ queryKey: ["datasets"] }); setSelected(id); setTab("mapping"); setMsg({ tone: "ok", text: "Files uploaded and inspected. Review the column mapping, then ingest." }); }} onError={(t) => setMsg({ tone: "danger", text: t })} />
          <Panel title="Your datasets">
            {datasets.isLoading && <Spinner />}
            {datasets.isError && <ErrorBox error={datasets.error} />}
            {datasets.data?.length === 0 && <div className="text-sm text-[var(--muted)]">No datasets yet.</div>}
            <ul className="space-y-1">
              {datasets.data?.map((d) => (
                <li key={d.id}>
                  <button onClick={() => setSelected(d.id)} className={`w-full rounded-lg border px-3 py-2 text-left text-sm ${selected === d.id ? "border-[var(--accent)] bg-[var(--accent-soft)]" : "border-[var(--border)] hover:border-[var(--border-strong)]"}`}>
                    <div className="truncate font-medium text-white">{d.name}</div>
                    <div className="text-xs text-[var(--muted)]">{d.n_files} file(s) · {d.n_versions} version(s) · {d.record_count} rows · <Badge tone={d.source_type === "bundled" ? "accent" : "muted"}>{d.source_type}</Badge></div>
                  </button>
                </li>
              ))}
            </ul>
          </Panel>
        </div>
        <div className="min-w-0">
          {selected === null && <EmptyState title="Select or upload a dataset">Upload your workbooks on the left, or import the bundled dataset.</EmptyState>}
          {selected !== null && dataset.isLoading && <Loading />}
          {selected !== null && dataset.isError && <ErrorBox error={dataset.error} />}
          {dataset.data && (
            <>
              <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
                <div>
                  <div className="text-lg font-semibold text-white">{dataset.data.name}</div>
                  <div className="text-xs text-[var(--muted)]">{dataset.data.description}</div>
                </div>
                <div className="flex gap-2">
                  <Tabs tabs={[{ key: "files", label: "Files & sheets" }, { key: "mapping", label: "Column mapping" }, { key: "preview", label: "Preview" }, { key: "quality", label: "Quality" }, { key: "versions", label: "Versions & preprocessing" }]} value={tab} onChange={setTab} />
                  <Button variant="danger" size="sm" onClick={() => setDel(true)} title="Delete dataset"><Trash2 size={14} /></Button>
                </div>
              </div>
              {tab === "files" && <FilesTab files={dataset.data.files} />}
              {tab === "mapping" && <MappingTab datasetId={dataset.data.id} files={dataset.data.files} onIngested={() => { qc.invalidateQueries({ queryKey: ["dataset", selected] }); qc.invalidateQueries({ queryKey: ["datasets"] }); setTab("versions"); setMsg({ tone: "ok", text: "Raw dataset version created. Apply preprocessing to derive COF / sliding distance and run data validation." }); }} onError={(t) => setMsg({ tone: "danger", text: t })} />}
              {tab === "preview" && <PreviewTab datasetId={dataset.data.id} versions={dataset.data.versions} />}
              {tab === "quality" && <QualityTab datasetId={dataset.data.id} versions={dataset.data.versions} />}
              {tab === "versions" && <VersionsTab datasetId={dataset.data.id} versions={dataset.data.versions} onChanged={() => { qc.invalidateQueries({ queryKey: ["dataset", selected] }); qc.invalidateQueries({ queryKey: ["datasets"] }); }} onMsg={setMsg} />}
            </>
          )}
        </div>
      </div>
      <ConfirmDialog open={del} onClose={() => setDel(false)} onConfirm={() => deleteDs.mutate()} title="Delete dataset?" danger confirmLabel="Delete" loading={deleteDs.isPending}>
        This removes the uploaded files, all versions and records. Datasets with trained models cannot be deleted.
      </ConfirmDialog>
    </>
  );
}

function UploadPanel({ onDone, onError }: { onDone: (id: number) => void; onError: (t: string) => void }) {
  const [name, setName] = useState("");
  const [desc, setDesc] = useState("");
  const [files, setFiles] = useState<FileList | null>(null);
  const up = useMutation({
    mutationFn: async () => {
      const fd = new FormData();
      fd.append("name", name || (files && files.length ? files[0].name.replace(/\.[^.]+$/, "") : "Dataset"));
      if (desc) fd.append("description", desc);
      Array.from(files || []).forEach((f) => fd.append("files", f));
      return apiUpload<{ dataset: { id: number }; errors: { filename: string; error: string }[] }>("/api/datasets/upload", fd);
    },
    onSuccess: (r) => { if (r.errors.length) onError(r.errors.map((e) => `${e.filename}: ${e.error}`).join("; ")); onDone(r.dataset.id); setFiles(null); setName(""); setDesc(""); },
    onError: (e) => onError(errorMessage(e)),
  });
  return (
    <Panel title="Upload experimental data" subtitle=".xlsx, .xls, .csv or tribometer .dwf — select several files at once (e.g. one workbook per run)">
      <div className="space-y-3">
        <Field label="Dataset name"><input className="input" value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Pin-on-disk campaign March" /></Field>
        <Field label="Description (optional)"><input className="input" value={desc} onChange={(e) => setDesc(e.target.value)} /></Field>
        <Field label="Files"><input className="input" type="file" multiple accept=".xlsx,.xls,.xlsm,.csv,.txt,.dwf" onChange={(e) => setFiles(e.target.files)} /></Field>
        <Button variant="primary" className="w-full" disabled={!files || files.length === 0} loading={up.isPending} onClick={() => up.mutate()}><Upload size={16} /> Upload & inspect</Button>
      </div>
    </Panel>
  );
}

function FilesTab({ files }: { files: FileView[] }) {
  if (!files.length) return <EmptyState title="No files">Upload files to inspect their worksheets.</EmptyState>;
  return (
    <div className="space-y-3">
      {files.map((f) => (
        <Panel key={f.id} title={f.filename} subtitle={`${f.file_type} · ${(f.size_bytes / 1024).toFixed(1)} kB · uploaded ${fmtDate(f.uploaded_at)}`}>
          {f.inspection.errors?.length > 0 && <Alert tone="danger" className="mb-3">{f.inspection.errors.join("; ")}</Alert>}
          {f.inspection.sheets.map((s) => (
            <div key={s.sheet_name} className="mb-3 rounded-xl border border-[var(--border)] p-3">
              <div className="mb-2 flex flex-wrap items-center gap-2 text-sm">
                <span className="font-semibold text-white">Sheet “{s.sheet_name}”</span>
                <Badge>{s.n_rows} rows × {s.n_cols} cols</Badge>
                {s.duplicate_rows > 0 && <Badge tone="warn">{s.duplicate_rows} duplicate rows</Badge>}
                {s.is_empty && <Badge tone="danger">empty</Badge>}
                {Object.keys(s.parsed_run || {}).length > 0 && <Badge tone="accent">run metadata parsed from name</Badge>}
              </div>
              {Object.keys(s.parsed_run || {}).length > 0 && (
                <div className="mb-2 text-xs text-[var(--muted)]">{Object.entries(s.parsed_run).map(([k, v]) => `${featureLabel(k)}: ${v}`).join(" · ")}</div>
              )}
              <DataTable columns={["name", "dtype", "non_null", "null_count", "n_unique", "min", "max", "mean", "sample_values"]} rows={s.columns as unknown as Record<string, unknown>[]} maxHeight={220} />
            </div>
          ))}
        </Panel>
      ))}
    </div>
  );
}

function MappingTab({ datasetId, files, onIngested, onError }: { datasetId: number; files: FileView[]; onIngested: () => void; onError: (t: string) => void }) {
  const dict = useQuery({ queryKey: ["feature-dictionary"], queryFn: () => apiGet<FeatureDef[]>("/api/datasets/feature-dictionary") });
  type Spec = { file_id: number; sheet_name: string; include: boolean; mapping: Record<string, string>; constants: Record<string, string>; units: Record<string, string> };
  const initial = useMemo<Spec[]>(() => files.flatMap((f) => f.inspection.sheets.filter((s) => !s.is_empty).map((s: SheetInspection) => ({
    file_id: f.id, sheet_name: s.sheet_name, include: true,
    mapping: Object.fromEntries(Object.entries(s.suggested_mapping || {}).map(([c, v]) => [c, v.feature || ""])),
    constants: Object.fromEntries(Object.entries(s.parsed_run || {}).filter(([k]) => ["material", "test_run_id", "normal_load_n", "rpm", "track_diameter_mm", "sliding_speed_m_s"].includes(k)).map(([k, v]) => [k, String(v)])),
    units: {},
  }))), [files]);
  const [specs, setSpecs] = useState<Spec[]>(initial);
  useEffect(() => setSpecs(initial), [initial]);
  const [notes, setNotes] = useState("");
  const [globalConst, setGlobalConst] = useState<Record<string, string>>({});
  const ingest = useMutation({
    mutationFn: () => apiPost(`/api/datasets/${datasetId}/ingest`, {
      notes,
      sheets: specs.filter((s) => s.include).map((s) => ({ file_id: s.file_id, sheet_name: s.sheet_name, mapping: Object.fromEntries(Object.entries(s.mapping).map(([c, f]) => [c, f || null])), constants: { ...Object.fromEntries(Object.entries(globalConst).filter(([, v]) => v !== "")), ...Object.fromEntries(Object.entries(s.constants).filter(([, v]) => v !== "")) }, units: Object.fromEntries(Object.entries(s.units).filter(([, v]) => v !== "")) })),
    }),
    onSuccess: onInGested,
    onError: (e) => onError(errorMessage(e)),
  });
  function onInGested() { onIngested(); }
  if (!files.length) return <EmptyState title="No files to map">Upload files first.</EmptyState>;
  const feats = dict.data || [];
  const mappedFeatures = new Set(specs.flatMap((s) => Object.values(s.mapping).concat(Object.keys(s.constants).filter((k) => s.constants[k] !== ""))).concat(Object.keys(globalConst).filter((k) => globalConst[k] !== "")));
  const canForm = mappedFeatures.has("cof") || (mappedFeatures.has("friction_force_n") && mappedFeatures.has("normal_load_n")) || mappedFeatures.has("wear_rate");
  const update = (i: number, patch: Partial<Spec>) => setSpecs((prev) => prev.map((s, j) => (j === i ? { ...s, ...patch } : s)));
  const CONST_FEATURES = ["material", "test_run_id", "lubricant", "material_pair", "normal_load_n", "sliding_speed_m_s", "temperature_c", "rpm", "track_diameter_mm"];
  return (
    <div className="space-y-4">
      <Alert tone="info" title="How mapping works">
        Each source column is mapped to a standardized feature. Values that are constant for a whole sheet (e.g. normal load = 50 N, material code) are entered as constants. A target must be available: a <b>cof</b> column, or <b>friction_force_n</b> together with <b>normal_load_n</b> (COF is then derived as FF / load), or a <b>wear_rate</b> column. Nothing is invented for missing fields.
      </Alert>
      <Panel title="Constants applied to every sheet" subtitle="Leave blank when not applicable. Per-sheet constants below override these.">
        <div className="grid grid-cols-2 gap-3 md:grid-cols-3">
          {["lubricant", "material_pair", "normal_load_n", "sliding_speed_m_s", "temperature_c"].map((k) => (
            <Field key={k} label={featureLabel(k)}><input className="input" value={globalConst[k] || ""} onChange={(e) => setGlobalConst({ ...globalConst, [k]: e.target.value })} placeholder={k === "lubricant" ? "e.g. dry" : ""} /></Field>
          ))}
        </div>
      </Panel>
      {specs.map((s, i) => {
        const file = files.find((f) => f.id === s.file_id)!;
        const sheet = file.inspection.sheets.find((x) => x.sheet_name === s.sheet_name)!;
        return (
          <Panel key={`${s.file_id}-${s.sheet_name}`} title={<span><input type="checkbox" checked={s.include} onChange={(e) => update(i, { include: e.target.checked })} className="mr-2" />{file.filename} — {s.sheet_name}</span>} subtitle={`${sheet.n_rows} rows`}>
            <div className="grid gap-4 md:grid-cols-2">
              <div>
                <div className="label mb-2">Source columns → standard features</div>
                {sheet.columns.map((c) => (
                  <div key={c.name} className="mb-2 grid grid-cols-[1fr_1fr_90px] items-center gap-2 text-sm">
                    <div className="truncate" title={c.name}>{c.name} <span className="text-xs text-[var(--muted)]">({c.dtype})</span></div>
                    <select className="input" value={s.mapping[c.name] || ""} onChange={(e) => update(i, { mapping: { ...s.mapping, [c.name]: e.target.value } })}>
                      <option value="">— ignore —</option>
                      {feats.map((f) => <option key={f.name} value={f.name}>{f.label} {f.unit ? `(${f.unit})` : ""}</option>)}
                    </select>
                    <input className="input" placeholder="unit" value={s.units[s.mapping[c.name]] || ""} disabled={!s.mapping[c.name] || feats.find((f) => f.name === s.mapping[c.name])?.dtype !== "number"} onChange={(e) => update(i, { units: { ...s.units, [s.mapping[c.name]]: e.target.value } })} title="Declared source unit (e.g. mm/s, kgf) — converted to the standard unit" />
                  </div>
                ))}
              </div>
              <div>
                <div className="label mb-2">Sheet constants (from the sheet name where parsed)</div>
                <div className="grid grid-cols-2 gap-2">
                  {CONST_FEATURES.map((k) => (
                    <Field key={k} label={featureLabel(k)}><input className="input" value={s.constants[k] || ""} onChange={(e) => update(i, { constants: { ...s.constants, [k]: e.target.value } })} /></Field>
                  ))}
                </div>
              </div>
            </div>
          </Panel>
        );
      })}
      <Panel>
        <Field label="Version notes (optional)"><input className="input" value={notes} onChange={(e) => setNotes(e.target.value)} /></Field>
        {!canForm && <Alert tone="warn" className="mt-3" title="No target can be formed yet">Map a COF column, or friction force + normal load, or a wear-rate column.</Alert>}
        <div className="mt-3 flex gap-2">
          <Button variant="primary" disabled={!canForm || specs.every((s) => !s.include)} loading={ingest.isPending} onClick={() => ingest.mutate()}>Create raw dataset version</Button>
        </div>
      </Panel>
    </div>
  );
}

function VersionSelect({ versions, value, onChange }: { versions: VersionSummary[]; value: number | null; onChange: (v: number) => void }) {
  return (
    <select className="input max-w-md" value={value ?? ""} onChange={(e) => onChange(Number(e.target.value))}>
      {versions.map((v) => <option key={v.id} value={v.id}>v{v.version_number} · {v.kind} · {v.record_count} rows ({v.usable_record_count} usable)</option>)}
    </select>
  );
}

function PreviewTab({ datasetId, versions }: { datasetId: number; versions: VersionSummary[] }) {
  const [vid, setVid] = useState<number | null>(versions[versions.length - 1]?.id ?? null);
  const [material, setMaterial] = useState("");
  const [offset, setOffset] = useState(0);
  const q = useQuery({ queryKey: ["preview", datasetId, vid, material, offset], queryFn: () => apiGet<{ total: number; columns: string[]; rows: Record<string, unknown>[] }>(`/api/datasets/${datasetId}/preview`, { version_id: vid, material, limit: 100, offset }), enabled: vid !== null });
  if (!versions.length) return <EmptyState title="No versions yet">Create a raw version from the Column mapping tab.</EmptyState>;
  const version = versions.find((v) => v.id === vid);
  const materials = version?.feature_summary?.inputs?.material?.levels || [];
  return (
    <Panel title="Record preview" subtitle="First 100 records of the selected version (paged)" actions={<>
      <VersionSelect versions={versions} value={vid} onChange={(v) => { setVid(v); setOffset(0); }} />
      <select className="input" style={{ width: 160 }} value={material} onChange={(e) => { setMaterial(e.target.value); setOffset(0); }}><option value="">All materials</option>{materials.map((m) => <option key={m} value={m}>{m}</option>)}</select>
    </>}>
      {q.isLoading && <Spinner />}
      {q.isError && <ErrorBox error={q.error} />}
      {q.data && (
        <>
          <DataTable columns={q.data.columns.filter((c) => !["id"].includes(c))} rows={q.data.rows} maxHeight={480} />
          <div className="mt-3 flex items-center justify-between text-sm text-[var(--muted)]">
            <span>{q.data.total} records · showing {offset + 1}–{Math.min(offset + 100, q.data.total)}</span>
            <div className="flex gap-2">
              <Button size="sm" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 100))}>Previous</Button>
              <Button size="sm" disabled={offset + 100 >= q.data.total} onClick={() => setOffset(offset + 100)}>Next</Button>
            </div>
          </div>
        </>
      )}
    </Panel>
  );
}

interface QualityReport {
  n_rows: number; usable_rows: number; unusable_rows?: number; duplicate_rows: number; score: number; flags: Record<string, number>;
  summary?: Record<string, unknown>; columns: { name: string; label: string; unit: string; dtype: string; non_null: number; missing: number; missing_pct: number; min?: number; max?: number; mean?: number; std?: number; n_unique?: number; constant?: boolean; invalid?: number }[];
  per_material: Record<string, unknown>[]; per_run: Record<string, unknown>[]; correlations: { features?: string[]; matrix?: (number | null)[][] };
}

function QualityTab({ datasetId, versions }: { datasetId: number; versions: VersionSummary[] }) {
  const [vid, setVid] = useState<number | null>(versions[versions.length - 1]?.id ?? null);
  const q = useQuery({ queryKey: ["quality", datasetId, vid], queryFn: () => apiGet<{ quality_report: QualityReport; cleaning_log: { step: string; description: string; rows_affected: number }[] }>(`/api/datasets/${datasetId}/quality`, { version_id: vid }), enabled: vid !== null });
  const dist = useQuery({ queryKey: ["dist", vid], queryFn: () => apiGet<{ features: { feature: string; label: string; unit: string; constant: boolean; value?: number; bins?: { x0: number; x1: number; count: number }[] }[] }>(`/api/analysis/${vid}/distributions`), enabled: vid !== null, retry: false });
  if (!versions.length) return <EmptyState title="No versions yet" />;
  const r = q.data?.quality_report;
  return (
    <div className="space-y-4">
      <Panel title="Data-quality report" actions={<>
        <VersionSelect versions={versions} value={vid} onChange={setVid} />
        <Button size="sm" onClick={() => apiDownload(`/api/datasets/${datasetId}/quality/report.md?version_id=${vid}`, `quality_report_v${vid}.md`)}><Download size={14} /> Download report</Button>
      </>}>
        {q.isLoading && <Spinner />}
        {q.isError && <ErrorBox error={q.error} />}
        {r && (
          <>
            <div className="mb-4 grid grid-cols-2 gap-3 md:grid-cols-5">
              {[["Rows", r.n_rows], ["Usable", r.usable_rows], ["Duplicates", r.duplicate_rows], ["Quality score", r.score], ["Wear rate", (r.summary as Record<string, unknown>)?.has_wear_rate ? "available" : "not in dataset"]].map(([k, v]) => (
                <div key={String(k)} className="rounded-xl border border-[var(--border)] p-3"><div className="label">{k}</div><div className="mt-1 text-xl font-bold text-white">{String(v)}</div></div>
              ))}
            </div>
            <div className="label mb-2">Columns</div>
            <DataTable columns={["label", "unit", "non_null", "missing", "missing_pct", "min", "max", "mean", "std", "n_unique", "constant", "invalid"]} rows={r.columns as unknown as Record<string, unknown>[]} maxHeight={300} />
            <div className="mt-4 grid gap-4 md:grid-cols-2">
              <div>
                <div className="label mb-2">Flags (rows)</div>
                <KeyValue items={Object.entries(r.flags || {}).map(([k, v]) => [k, v])} />
                {Object.keys(r.flags || {}).length === 0 && <div className="text-sm text-[var(--muted)]">No flags (raw version).</div>}
              </div>
              <div>
                <div className="label mb-2">Cleaning log</div>
                <ol className="list-decimal space-y-1 pl-5 text-sm text-[var(--muted)]">{(q.data?.cleaning_log || []).map((e, i) => <li key={i}><span className="text-white">{e.step}</span>: {e.description} <Badge>{e.rows_affected} rows</Badge></li>)}</ol>
                {(q.data?.cleaning_log || []).length === 0 && <div className="text-sm text-[var(--muted)]">No preprocessing applied to this version.</div>}
              </div>
            </div>
            {r.per_run?.length > 0 && <><div className="label mt-4 mb-2">Per test run</div><DataTable columns={Object.keys(r.per_run[0])} rows={r.per_run} maxHeight={260} /></>}
            {r.correlations?.features && (
              <div className="mt-4"><div className="label mb-2">Correlation matrix (Pearson, varying numeric columns)</div><Heatmap labels={r.correlations.features.map(featureLabel)} matrix={r.correlations.matrix!} /></div>
            )}
          </>
        )}
      </Panel>
      {dist.data && (
        <div className="grid gap-4 md:grid-cols-2">
          {dist.data.features.filter((f) => !f.constant && f.bins).map((f) => (
            <ChartBox key={f.feature} title={`${f.label} distribution`} subtitle={`measured, ${f.unit}`} height={220}><Histogram bins={f.bins!} xLabel={f.unit} /></ChartBox>
          ))}
          {dist.data.features.filter((f) => f.constant).length > 0 && (
            <Panel title="Constant columns"><KeyValue items={dist.data.features.filter((f) => f.constant).map((f) => [f.label, `${fmt(f.value)} ${f.unit}`])} /></Panel>
          )}
        </div>
      )}
    </div>
  );
}

function VersionsTab({ datasetId, versions, onChanged, onMsg }: { datasetId: number; versions: VersionSummary[]; onChanged: () => void; onMsg: (m: { tone: "ok" | "danger" | "warn"; text: string }) => void }) {
  const opts = useQuery({ queryKey: ["preprocess-options"], queryFn: () => apiGet<Record<string, unknown>>("/api/datasets/preprocess-options") });
  const [parent, setParent] = useState<number | null>(versions[versions.length - 1]?.id ?? null);
  const [options, setOptions] = useState<Record<string, unknown>>({});
  const [notes, setNotes] = useState("");
  useEffect(() => { if (opts.data) setOptions(opts.data); }, [opts.data]);
  const run = useMutation({
    mutationFn: () => apiPost(`/api/datasets/${datasetId}/preprocess`, { version_id: parent, options, notes }),
    onSuccess: () => { onChanged(); onMsg({ tone: "ok", text: "Processed version created." }); },
    onError: (e) => onMsg({ tone: "danger", text: errorMessage(e) }),
  });
  const models = useQuery({ queryKey: ["models"], queryFn: () => apiGet<{ id: number; dataset_version_id: number; algorithm: string; n_train: number; n_test: number; status: string }[]>("/api/models") });
  if (!versions.length) return <EmptyState title="No versions yet">Create a raw version from the Column mapping tab.</EmptyState>;
  const boolKeys = ["drop_duplicates", "derive_cof", "derive_sliding_distance", "derive_speed_from_rpm", "derive_material_attributes", "flag_run_in", "exclude_run_in", "exclude_outliers", "drop_rows_missing_target"];
  return (
    <div className="space-y-4">
      <Panel title="Dataset versions" subtitle="Raw → processed → augmented (with approved validation records). Each version keeps its own records; the raw version is never changed.">
        <div className="overflow-auto">
          <table className="table">
            <thead><tr><th>Version</th><th>Kind</th><th>Parent</th><th>Rows</th><th>Usable</th><th>Quality</th><th>Targets</th><th>Models trained on it</th><th>Created</th><th>Notes</th><th>Export</th></tr></thead>
            <tbody>
              {versions.map((v) => {
                const used = (models.data || []).filter((m) => m.dataset_version_id === v.id);
                return (
                  <tr key={v.id}>
                    <td>v{v.version_number} <span className="text-xs text-[var(--muted)]">(id {v.id})</span></td>
                    <td><Badge tone={v.kind === "raw" ? "muted" : v.kind === "processed" ? "accent" : "ok"}>{v.kind}</Badge></td>
                    <td>{v.parent_version_id ?? "–"}</td><td>{v.record_count}</td><td>{v.usable_record_count}</td><td>{v.quality_score ?? "–"}</td>
                    <td>{Object.entries(v.feature_summary?.targets || {}).filter(([, t]) => t.available).map(([k]) => k).join(", ") || "none"}</td>
                    <td>{used.length ? used.map((m) => `${m.algorithm} (#${m.id}, ${m.n_train}+${m.n_test} rows)`).join("; ") : "–"}</td>
                    <td>{fmtDate(v.created_at)}</td><td className="max-w-[240px] truncate" title={v.notes || ""}>{v.notes}</td>
                    <td className="flex gap-1">
                      <Button size="sm" onClick={() => apiDownload(`/api/datasets/${datasetId}/export?version_id=${v.id}&format=csv`, `dataset_${datasetId}_v${v.version_number}.csv`)}>CSV</Button>
                      <Button size="sm" onClick={() => apiDownload(`/api/datasets/${datasetId}/export?version_id=${v.id}&format=xlsx`, `dataset_${datasetId}_v${v.version_number}.xlsx`)}>XLSX</Button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Panel>
      <Panel title="Apply preprocessing → new processed version" subtitle="Documented operations; every step is written to the cleaning log.">
        <div className="grid gap-3 md:grid-cols-2">
          <Field label="Parent version" className="md:col-span-2"><VersionSelect versions={versions} value={parent} onChange={setParent} /></Field>
          <Field label="Negative friction force" hint="Sensor sign noise near zero: take absolute value (as in the published processed files), drop, or keep">
            <select className="input" value={String(options.negative_friction ?? "abs")} onChange={(e) => setOptions({ ...options, negative_friction: e.target.value })}><option value="abs">absolute value</option><option value="drop">mark unusable</option><option value="keep">keep</option></select>
          </Field>
          <Field label="Run-in threshold (fraction of steady-state friction)"><input className="input" type="number" step="0.01" value={String(options.run_in_fraction ?? 0.05)} onChange={(e) => setOptions({ ...options, run_in_fraction: Number(e.target.value) })} /></Field>
          <Field label="Outlier robust z-score (0 disables)"><input className="input" type="number" step="0.5" value={String(options.outlier_zscore ?? 4)} onChange={(e) => setOptions({ ...options, outlier_zscore: Number(e.target.value) })} /></Field>
          <Field label="Notes"><input className="input" value={notes} onChange={(e) => setNotes(e.target.value)} /></Field>
          <div className="grid grid-cols-2 gap-2 md:col-span-2">
            {boolKeys.map((k) => (
              <label key={k} className="flex items-center gap-2 text-sm text-[var(--muted)]"><input type="checkbox" checked={Boolean(options[k])} onChange={(e) => setOptions({ ...options, [k]: e.target.checked })} />{k.replace(/_/g, " ")}</label>
            ))}
          </div>
        </div>
        <Button variant="primary" className="mt-4" loading={run.isPending} onClick={() => run.mutate()}>Create processed version</Button>
      </Panel>
    </div>
  );
}
