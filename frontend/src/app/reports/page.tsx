"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { Download, Trash2 } from "lucide-react";
import { apiDelete, apiDownload, apiGet, apiPost, errorMessage } from "@/lib/api";
import { Alert, Badge, Button, ErrorBox, Field, Loading, PageHeader, Panel } from "@/components/ui";
import { fmtDate, MODEL_LABELS } from "@/lib/format";
import { useDatasets, useModels, VersionSummary } from "@/lib/hooks";

interface ReportView { id: number; kind: string; title: string; params: Record<string, unknown>; dataset_version_id: number | null; created_at: string; file_exists: boolean; filename: string }

export default function ReportsPage() {
  const qc = useQueryClient();
  const datasets = useDatasets();
  const models = useModels();
  const reports = useQuery({ queryKey: ["reports"], queryFn: () => apiGet<ReportView[]>("/api/reports") });
  const [dsId, setDsId] = useState<number | null>(null);
  const versions = useQuery({ queryKey: ["versions", dsId], queryFn: () => apiGet<VersionSummary[]>(`/api/datasets/${dsId}/versions`), enabled: dsId !== null });
  const [form, setForm] = useState({ format: "pdf", title: "", dataset_version_id: "" as string | number, model_version_ids: [] as number[], include_predictions: 20, include_validation: true, include_explainability: true });
  const [msg, setMsg] = useState<{ tone: "ok" | "danger"; text: string } | null>(null);
  useEffect(() => { if (dsId === null && datasets.data?.length) setDsId(datasets.data[0].id); }, [datasets.data, dsId]);
  useEffect(() => { if (versions.data?.length) setForm((f) => ({ ...f, dataset_version_id: versions.data![versions.data!.length - 1].id })); }, [versions.data]);
  const gen = useMutation({
    mutationFn: () => apiPost<ReportView>("/api/reports/generate", { ...form, title: form.title || null, dataset_version_id: form.dataset_version_id || null, model_version_ids: form.model_version_ids.length ? form.model_version_ids : null }),
    onSuccess: (r) => { qc.invalidateQueries({ queryKey: ["reports"] }); setMsg({ tone: "ok", text: `Report "${r.title}" generated.` }); apiDownload(`/api/reports/${r.id}/download`, r.filename); },
    onError: (e) => setMsg({ tone: "danger", text: errorMessage(e) }),
  });
  const del = useMutation({ mutationFn: (id: number) => apiDelete(`/api/reports/${id}`), onSuccess: () => qc.invalidateQueries({ queryKey: ["reports"] }) });
  return (
    <>
      <PageHeader title="Reports" subtitle="Generate a PDF or Excel report with project information, dataset description, cleaning summary, parameter ranges, methodology, model comparison, metrics, predictions, explainability, validation results and limitations — all from real stored results." />
      {msg && <Alert tone={msg.tone} className="mb-4">{msg.text}</Alert>}
      <div className="grid gap-4 lg:grid-cols-[400px_1fr]">
        <Panel title="Generate report">
          <div className="space-y-3">
            <Field label="Format"><select className="input" value={form.format} onChange={(e) => setForm({ ...form, format: e.target.value })}><option value="pdf">PDF</option><option value="xlsx">Excel (.xlsx)</option></select></Field>
            <Field label="Title"><input className="input" value={form.title} onChange={(e) => setForm({ ...form, title: e.target.value })} placeholder="AI TriboPredict — Model and Prediction Report" /></Field>
            <Field label="Dataset"><select className="input" value={dsId ?? ""} onChange={(e) => setDsId(Number(e.target.value))}>{datasets.data?.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}</select></Field>
            <Field label="Dataset version"><select className="input" value={form.dataset_version_id} onChange={(e) => setForm({ ...form, dataset_version_id: Number(e.target.value) })}>{versions.data?.map((v) => <option key={v.id} value={v.id}>v{v.version_number} · {v.kind}</option>)}</select></Field>
            <Field label="Models to feature" hint="Leave empty to use the active (production or best) model per target; its whole training run is used for the comparison table">
              <div className="max-h-40 space-y-1 overflow-auto rounded-lg border border-[var(--border)] p-2">
                {models.data?.map((m) => <label key={m.id} className="flex items-center gap-2 text-sm text-[var(--muted)]"><input type="checkbox" checked={form.model_version_ids.includes(m.id)} onChange={(e) => setForm({ ...form, model_version_ids: e.target.checked ? [...form.model_version_ids, m.id] : form.model_version_ids.filter((x) => x !== m.id) })} />#{m.id} {MODEL_LABELS[m.algorithm] || m.algorithm} · {m.target} · {m.status}</label>)}
                {models.data?.length === 0 && <div className="text-xs text-[var(--muted)]">No models trained yet — the report will state "Model not trained".</div>}
              </div>
            </Field>
            <Field label="Number of recent predictions to include"><input className="input" type="number" min={0} max={500} value={form.include_predictions} onChange={(e) => setForm({ ...form, include_predictions: Number(e.target.value) })} /></Field>
            <label className="flex items-center gap-2 text-sm text-[var(--muted)]"><input type="checkbox" checked={form.include_validation} onChange={(e) => setForm({ ...form, include_validation: e.target.checked })} />Include experimental validation results</label>
            <label className="flex items-center gap-2 text-sm text-[var(--muted)]"><input type="checkbox" checked={form.include_explainability} onChange={(e) => setForm({ ...form, include_explainability: e.target.checked })} />Include explainability plots</label>
            <Button variant="primary" className="w-full" loading={gen.isPending} onClick={() => gen.mutate()}>Generate & download</Button>
          </div>
        </Panel>
        <Panel title="Report history">
          {reports.isLoading && <Loading />}
          {reports.isError && <ErrorBox error={reports.error} />}
          {reports.data?.length === 0 && <div className="text-sm text-[var(--muted)]">No reports generated yet.</div>}
          {reports.data && reports.data.length > 0 && (
            <table className="table"><thead><tr><th>ID</th><th>Title</th><th>Format</th><th>Dataset version</th><th>Models</th><th>Created</th><th></th></tr></thead>
              <tbody>{reports.data.map((r) => <tr key={r.id}><td>{r.id}</td><td className="text-white">{r.title}</td><td><Badge tone="accent">{r.kind}</Badge></td><td>{r.dataset_version_id}</td><td>{((r.params.model_version_ids as number[]) || []).join(", ") || "–"}</td><td>{fmtDate(r.created_at)}</td><td className="flex gap-1"><Button size="sm" disabled={!r.file_exists} onClick={() => apiDownload(`/api/reports/${r.id}/download`, r.filename)}><Download size={14} /></Button><Button size="sm" variant="danger" onClick={() => del.mutate(r.id)}><Trash2 size={14} /></Button></td></tr>)}</tbody>
            </table>
          )}
        </Panel>
      </div>
    </>
  );
}
