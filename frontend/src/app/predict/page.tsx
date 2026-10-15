"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { Download } from "lucide-react";
import { apiDownload, apiGet, apiPost, apiUpload, errorMessage } from "@/lib/api";
import { Alert, Badge, Button, DataTable, EmptyState, ErrorBox, Field, KeyValue, Loading, PageHeader, Panel, Spinner, Tabs } from "@/components/ui";
import { featureLabel, fmtDate, fmtNum, MODEL_LABELS } from "@/lib/format";
import { PredictSchema, usePredictSchema } from "@/lib/hooks";
import { ExplanationPanel, InputForm, PredictResult, TargetResult, toInputs } from "@/components/predict-form";


export default function PredictPage() {
  const schema = usePredictSchema();
  const [tab, setTab] = useState<"single" | "batch" | "history">("single");
  return (
    <>
      <PageHeader title="AI TriboPredict" subtitle="Enter supported operating conditions and obtain COF / wear-rate predictions from the active trained model, with uncertainty, domain checks and SHAP explanations. Outputs are model predictions, not measurements." />
      <div className="mb-4"><Tabs tabs={[{ key: "single", label: "Single prediction" }, { key: "batch", label: "Batch predictions" }, { key: "history", label: "Prediction history" }]} value={tab} onChange={setTab} /></div>
      {schema.isLoading && <Loading label="Loading model schema…" />}
      {schema.isError && <ErrorBox error={schema.error} />}
      {schema.data && tab === "single" && <SingleTab schema={schema.data} />}
      {schema.data && tab === "batch" && <BatchTab schema={schema.data} />}
      {tab === "history" && <HistoryTab />}
    </>
  );
}


function SingleTab({ schema }: { schema: PredictSchema }) {
  const qc = useQueryClient();
  const [values, setValues] = useState<Record<string, string>>({});
  useEffect(() => {
    const init: Record<string, string> = {};
    for (const inp of schema.inputs) init[inp.name] = inp.dtype === "number" ? String(Number(((inp.min ?? 0) + (inp.max ?? 0)) / 2).toFixed(2)) : Object.keys(inp.levels || {})[0] || "";
    setValues(init);
  }, [schema]);
  const predict = useMutation({ mutationFn: () => apiPost<PredictResult>("/api/predict", { inputs: toInputs(schema, values), explain: true, save: true }), onSuccess: () => qc.invalidateQueries({ queryKey: ["history"] }) });
  const anyModel = Object.values(schema.targets).some((t) => t.available);
  if (!anyModel) return <EmptyState title="Model not trained">No trained model is available. Train a COF (or wear-rate) model first.</EmptyState>;
  const r = predict.data;
  return (
    <div className="grid gap-4 lg:grid-cols-[380px_1fr]">
      <Panel title="Inputs" subtitle="Dropdowns and ranges are derived from the training data of the active model" glow>
        <InputForm schema={schema} values={values} onChange={setValues} />
        <Button variant="primary" className="mt-4 w-full" loading={predict.isPending} onClick={() => predict.mutate()}>Predict</Button>
        <div className="mt-4 text-xs text-[var(--muted)]">
          {Object.entries(schema.targets).map(([t, info]) => <div key={t}>{t === "cof" ? "COF" : "Wear rate"}: {info.available ? <>{MODEL_LABELS[info.algorithm!] || info.algorithm} <Badge tone={info.status === "production" ? "ok" : "accent"}>{info.status}</Badge> {info.version_label}</> : <Badge tone="warn">unavailable — {info.reason}</Badge>}</div>)}
        </div>
      </Panel>
      <div className="space-y-4">
        {predict.isError && <Alert tone="danger" title="Prediction refused">{errorMessage(predict.error)}</Alert>}
        {!r && !predict.isError && <EmptyState title="Predictions">Set the operating conditions and click Predict.</EmptyState>}
        {r && (
          <>
            {r.warnings.length > 0 && <Alert tone="warn" title={r.in_domain ? "Notes" : "Outside the training-data distribution"}>{r.warnings.map((w, i) => <div key={i}>• {w}</div>)}</Alert>}
            <div className="grid gap-4 md:grid-cols-3">
              <TargetCard title="Predicted COF" t={r.cof} />
              <TargetCard title="Predicted wear rate" t={r.wear_rate} />
              <Panel title="Confidence">
                {r.cof.available && r.cof.interval ? (
                  <KeyValue items={[["Interval level", `${Math.round((r.cof.interval.level || 0) * 100)} %`], ["COF interval", `${fmtNum(r.cof.interval.low)} – ${fmtNum(r.cof.interval.high)}`], ["Empirical test coverage", r.cof.interval.test_coverage == null ? "–" : `${(100 * r.cof.interval.test_coverage).toFixed(1)} %`], ["Domain", r.in_domain ? <Badge key="d" tone="ok">inside experimental domain</Badge> : <Badge key="d" tone="warn">{r.cof.extrapolation} extrapolation</Badge>]]} />
                ) : <div className="text-sm text-[var(--muted)]">No valid interval method available for this model.</div>}
                <div className="mt-2 text-[11px] text-[var(--muted-2)]">{r.cof.interval?.method}</div>
              </Panel>
            </div>
            <ExplanationPanel t={r.cof} />
            <Panel title="Record" actions={r.prediction_id && <><Button size="sm" onClick={() => apiDownload(`/api/predictions/${r.prediction_id}/export?format=json`, `prediction_${r.prediction_id}.json`)}><Download size={14} /> JSON</Button><Button size="sm" onClick={() => apiDownload(`/api/predictions/${r.prediction_id}/export?format=csv`, `prediction_${r.prediction_id}.csv`)}><Download size={14} /> CSV</Button></>}>
              <KeyValue items={[["Prediction ID", r.prediction_id ?? "not saved"], ["Saved at", fmtDate(r.created_at)], ["COF model version", r.cof.model ? `${r.cof.model.version_label} (#${r.cof.model.id}, ${r.cof.model.status})` : "–"], ["Inputs used", JSON.stringify(r.cof.inputs_used ?? r.inputs)]]} />
              <div className="mt-2 text-xs text-[var(--predicted)]">{r.note}</div>
            </Panel>
          </>
        )}
      </div>
    </div>
  );
}

function TargetCard({ title, t }: { title: string; t: TargetResult }) {
  if (!t.available) {
    return <Panel title={title}><div className="text-xl font-bold text-[var(--muted)]">Unavailable</div><div className="mt-1 text-xs text-[var(--muted)]">{t.reason}</div></Panel>;
  }
  return (
    <Panel title={title} glow>
      <div className="text-3xl font-bold text-[var(--predicted)]">{fmtNum(t.value)} <span className="text-base font-medium text-[var(--muted)]">{t.unit}</span></div>
      {t.interval && <div className="mt-1 text-xs text-[var(--muted)]">{Math.round((t.interval.level || 0) * 100)} % interval {fmtNum(t.interval.low)} – {fmtNum(t.interval.high)}</div>}
      <div className="mt-2"><Badge tone="predicted">model prediction</Badge> {t.model && <span className="ml-1 text-xs text-[var(--muted)]">{MODEL_LABELS[t.model.algorithm] || t.model.algorithm}</span>}</div>
    </Panel>
  );
}


function BatchTab({ schema }: { schema: PredictSchema }) {
  const [rows, setRows] = useState<Record<string, string>[]>([]);
  const [text, setText] = useState("");
  const [result, setResult] = useState<{ n: number; n_ok: number; results: ({ index: number; ok: boolean; inputs: Record<string, unknown>; error?: string } & Partial<PredictResult>)[] } | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const inputNames = schema.inputs.map((i) => i.name);
  const run = useMutation({ mutationFn: (payload: Record<string, unknown>[]) => apiPost<typeof result>("/api/predict/batch", { rows: payload, save: true }), onSuccess: (r) => { setResult(r); setErr(null); }, onError: (e) => setErr(errorMessage(e)) });
  const upload = useMutation({ mutationFn: (f: File) => { const fd = new FormData(); fd.append("file", f); return apiUpload<typeof result>("/api/predict/batch/upload", fd); }, onSuccess: (r) => { setResult(r); setErr(null); }, onError: (e) => setErr(errorMessage(e)) });
  const addRow = () => setRows([...rows, Object.fromEntries(inputNames.map((n) => [n, ""]))]);
  const parseText = () => {
    const lines = text.trim().split(/\n+/);
    if (lines.length < 2) return [];
    const header = lines[0].split(/[,;\t]/).map((s) => s.trim());
    return lines.slice(1).map((l) => Object.fromEntries(l.split(/[,;\t]/).map((v, i) => [header[i], v.trim()])));
  };
  const toPayload = (rs: Record<string, string>[]) => rs.map((r) => toInputs(schema, r));
  const resultRows = (result?.results || []).map((r) => ({ row: r.index + 1, ...Object.fromEntries(Object.entries(r.inputs || {}).map(([k, v]) => [featureLabel(k), v])), predicted_cof: r.ok ? r.cof?.value : null, cof_low: r.ok ? r.cof?.interval?.low : null, cof_high: r.ok ? r.cof?.interval?.high : null, predicted_wear_rate: r.ok ? (r.wear_rate?.available ? r.wear_rate.value : "unavailable") : null, in_domain: r.ok ? r.in_domain : null, status: r.ok ? (r.warnings?.length ? r.warnings.join(" ") : "ok") : r.error }));
  return (
    <div className="space-y-4">
      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Enter conditions" subtitle={`Columns: ${inputNames.join(", ")}`} actions={<Button size="sm" onClick={addRow}>Add row</Button>}>
          {rows.length > 0 && (
            <div className="overflow-auto"><table className="table"><thead><tr>{inputNames.map((n) => <th key={n}>{featureLabel(n)}</th>)}<th /></tr></thead>
              <tbody>{rows.map((r, i) => <tr key={i}>{inputNames.map((n) => { const inp = schema.inputs.find((x) => x.name === n)!; return <td key={n}>{inp.dtype === "number" ? <input className="input" type="number" step="any" value={r[n]} onChange={(e) => setRows(rows.map((x, j) => (j === i ? { ...x, [n]: e.target.value } : x)))} /> : <select className="input" value={r[n]} onChange={(e) => setRows(rows.map((x, j) => (j === i ? { ...x, [n]: e.target.value } : x)))}><option value="">—</option>{Object.keys(inp.levels || {}).map((lv) => <option key={lv}>{lv}</option>)}</select>}</td>; })}<td><Button size="sm" onClick={() => setRows(rows.filter((_, j) => j !== i))}>✕</Button></td></tr>)}</tbody></table></div>
          )}
          <Button variant="primary" className="mt-3" disabled={!rows.length} loading={run.isPending} onClick={() => run.mutate(toPayload(rows))}>Predict {rows.length} row(s)</Button>
        </Panel>
        <Panel title="Or paste / upload a table" subtitle="CSV text with a header row, or upload a .csv / .xlsx file whose columns use the standard feature names">
          <textarea className="input h-32 font-mono text-xs" value={text} onChange={(e) => setText(e.target.value)} placeholder={`${inputNames.join(",")}\n${schema.inputs.map((i) => (i.dtype === "number" ? fmtNum(i.mean, 1) : Object.keys(i.levels || {})[0])).join(",")}`} />
          <div className="mt-3 flex flex-wrap gap-2">
            <Button variant="primary" disabled={!text.trim()} loading={run.isPending} onClick={() => run.mutate(toPayload(parseText()))}>Predict pasted rows</Button>
            <label className="btn"><input type="file" accept=".csv,.xlsx,.xls" className="hidden" onChange={(e) => e.target.files?.[0] && upload.mutate(e.target.files[0])} />Upload file</label>
          </div>
        </Panel>
      </div>
      {err && <Alert tone="danger">{err}</Alert>}
      {result && (
        <Panel title={`Results — ${result.n_ok} of ${result.n} rows predicted`} subtitle="Rows that fail validation are reported individually; no fake values are substituted." actions={<Button size="sm" onClick={() => downloadCsv(resultRows)}><Download size={14} /> Export CSV</Button>}>
          <DataTable columns={Object.keys(resultRows[0] || {})} rows={resultRows} />
        </Panel>
      )}
    </div>
  );
}

function downloadCsv(rows: Record<string, unknown>[]) {
  if (!rows.length) return;
  const cols = Object.keys(rows[0]);
  const csv = [cols.join(","), ...rows.map((r) => cols.map((c) => JSON.stringify(r[c] ?? "")).join(","))].join("\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
  a.download = "batch_predictions.csv";
  a.click();
}

function HistoryTab() {
  const q = useQuery({ queryKey: ["history"], queryFn: () => apiGet<{ total: number; items: { id: number; inputs: Record<string, unknown>; predicted_cof: number | null; cof_interval: { low: number; high: number } | null; predicted_wear_rate: number | null; in_domain: boolean; warnings: string[]; source: string; cof_model_version_id: number | null; created_at: string }[] }>("/api/predictions/history", { limit: 200 }) });
  if (q.isLoading) return <Loading />;
  if (q.isError) return <ErrorBox error={q.error} />;
  const rows = q.data!.items.map((p) => ({ id: p.id, created: fmtDate(p.created_at), source: p.source, inputs: Object.entries(p.inputs).map(([k, v]) => `${k}=${v}`).join(", "), predicted_cof: p.predicted_cof, interval: p.cof_interval ? `${fmtNum(p.cof_interval.low)} – ${fmtNum(p.cof_interval.high)}` : "–", predicted_wear_rate: p.predicted_wear_rate ?? "unavailable", in_domain: p.in_domain, model: p.cof_model_version_id, warnings: p.warnings.join(" ") }));
  return (
    <Panel title={`Prediction history (${q.data!.total})`} subtitle="Every saved prediction with its model version — model outputs, not measurements">
      {rows.length === 0 ? <div className="text-sm text-[var(--muted)]">No predictions saved yet.</div> : <DataTable columns={Object.keys(rows[0])} rows={rows} maxHeight={560} />}
    </Panel>
  );
}
