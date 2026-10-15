"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import Link from "next/link";
import { apiGet, apiPatch, apiPost, errorMessage } from "@/lib/api";
import { Alert, Badge, Button, ConfirmDialog, EmptyState, ErrorBox, Field, KeyValue, Loading, PageHeader, Panel } from "@/components/ui";
import { ChartBox, ScatterXY } from "@/components/charts";
import { featureLabel, fmtDate, fmtNum, fmtPct, MODEL_LABELS } from "@/lib/format";
import { ModelSummary, useDatasets, usePredictSchema, VersionSummary } from "@/lib/hooks";
import { InputForm, toInputs } from "@/components/predict-form";

interface ValidationRecord { id: number; experiment_id: string; prediction_id: number | null; model_version_id: number | null; inputs: Record<string, unknown>; measured_cof: number | null; measured_wear_rate: number | null; predicted_cof: number | null; predicted_wear_rate: number | null; cof_abs_error: number | null; cof_pct_error: number | null; status: string; notes: string | null; added_to_version_id: number | null; created_at: string }

export default function ValidationPage() {
  const qc = useQueryClient();
  const schema = usePredictSchema();
  const records = useQuery({ queryKey: ["validation"], queryFn: () => apiGet<ValidationRecord[]>("/api/validation") });
  const summary = useQuery({ queryKey: ["validation-summary"], queryFn: () => apiGet<{ n_records: number; n_pending: number; n_approved: number; n_rejected: number; cof: { n: number; mae: number | null; rmse: number | null; mape: number | null; max_abs_error: number | null }; by_model: Record<string, { algorithm: string; version_label: string; n: number; mae: number; rmse: number }> }>("/api/validation/summary") });
  const datasets = useDatasets();
  const [msg, setMsg] = useState<{ tone: "ok" | "danger"; text: string } | null>(null);
  const refresh = () => { qc.invalidateQueries({ queryKey: ["validation"] }); qc.invalidateQueries({ queryKey: ["validation-summary"] }); };
  const update = useMutation({ mutationFn: (p: { id: number; status: string }) => apiPatch(`/api/validation/${p.id}`, { status: p.status }), onSuccess: refresh, onError: (e) => setMsg({ tone: "danger", text: errorMessage(e) }) });

  return (
    <>
      <PageHeader title="Experimental Validation" subtitle="Close the loop: predict → test the condition on the tribometer → enter the measured result → compare → approve → add to a new dataset version → retrain → compare models → promote explicitly. ML complements experiments; it does not replace them." />
      {msg && <Alert tone={msg.tone} className="mb-4">{msg.text}</Alert>}
      <div className="grid gap-4 xl:grid-cols-[380px_1fr]">
        <div className="space-y-4">
          {schema.data && <NewRecordForm schema={schema.data} onDone={() => { refresh(); setMsg({ tone: "ok", text: "Validation record saved and compared with the model prediction." }); }} onError={(t) => setMsg({ tone: "danger", text: t })} />}
          {schema.isLoading && <Loading />}
          <Panel title="Validation summary" subtitle="Prediction errors on measured experiments">
            {summary.data ? (
              <>
                <KeyValue items={[["Records", summary.data.n_records], ["Pending / approved / rejected", `${summary.data.n_pending} / ${summary.data.n_approved} / ${summary.data.n_rejected}`], ["COF MAE", fmtNum(summary.data.cof.mae, 5)], ["COF RMSE", fmtNum(summary.data.cof.rmse, 5)], ["COF MAPE", fmtPct(summary.data.cof.mape)], ["Max abs error", fmtNum(summary.data.cof.max_abs_error, 5)]]} />
                {Object.entries(summary.data.by_model).length > 0 && <div className="mt-3"><div className="label mb-1">By model version</div><KeyValue items={Object.entries(summary.data.by_model).map(([id, m]) => [`#${id} ${MODEL_LABELS[m.algorithm] || m.algorithm}`, `MAE ${fmtNum(m.mae, 5)} (n = ${m.n})`])} /></div>}
              </>
            ) : <Loading />}
          </Panel>
        </div>
        <div className="space-y-4">
          <Panel title="Validation records" subtitle="Measured vs predicted; approve records to make them eligible for a new dataset version">
            {records.isLoading && <Loading />}
            {records.isError && <ErrorBox error={records.error} />}
            {records.data?.length === 0 && <div className="text-sm text-[var(--muted)]">No validation records yet.</div>}
            {records.data && records.data.length > 0 && (
              <div className="overflow-auto">
                <table className="table">
                  <thead><tr><th>Experiment</th><th>Conditions</th><th><span className="text-[var(--measured)]">Measured COF</span></th><th><span className="text-[var(--predicted)]">Predicted COF</span></th><th>Abs error</th><th>% error</th><th>Wear (meas / pred)</th><th>Model</th><th>Status</th><th>Added to version</th><th>Date</th><th></th></tr></thead>
                  <tbody>
                    {records.data.map((r) => (
                      <tr key={r.id}>
                        <td className="font-medium text-white">{r.experiment_id}</td>
                        <td className="max-w-[220px] truncate" title={JSON.stringify(r.inputs)}>{Object.entries(r.inputs).map(([k, v]) => `${featureLabel(k)}=${typeof v === "number" ? fmtNum(v, 2) : v}`).join(", ")}</td>
                        <td>{fmtNum(r.measured_cof)}</td><td>{fmtNum(r.predicted_cof)}</td><td>{fmtNum(r.cof_abs_error, 5)}</td><td>{fmtPct(r.cof_pct_error)}</td>
                        <td>{r.measured_wear_rate ?? "–"} / {r.predicted_wear_rate ?? "unavailable"}</td><td>{r.model_version_id ?? "–"}</td>
                        <td><Badge tone={r.status === "approved" ? "ok" : r.status === "rejected" ? "danger" : "warn"}>{r.status}</Badge></td>
                        <td>{r.added_to_version_id ?? "–"}</td><td>{fmtDate(r.created_at)}</td>
                        <td className="flex gap-1">{r.status !== "approved" && !r.added_to_version_id && <Button size="sm" variant="primary" onClick={() => update.mutate({ id: r.id, status: "approved" })}>Approve</Button>}{r.status !== "rejected" && !r.added_to_version_id && <Button size="sm" onClick={() => update.mutate({ id: r.id, status: "rejected" })}>Reject</Button>}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Panel>
          {records.data && records.data.filter((r) => r.measured_cof != null && r.predicted_cof != null).length > 0 && (
            <ChartBox title="Measured vs predicted COF (validation experiments)" subtitle="dashed = perfect agreement" height={280}>
              <ScatterXY points={records.data.filter((r) => r.measured_cof != null && r.predicted_cof != null).map((r) => ({ x: r.measured_cof!, y: r.predicted_cof!, group: r.status }))} xLabel="Measured COF" yLabel="Predicted COF" identity groups />
            </ChartBox>
          )}
          {records.data && datasets.data && <AugmentPanel records={records.data} datasets={datasets.data} onDone={(t) => { refresh(); qc.invalidateQueries({ queryKey: ["datasets"] }); setMsg({ tone: "ok", text: t }); }} onError={(t) => setMsg({ tone: "danger", text: t })} />}
          <RetrainComparePanel />
        </div>
      </div>
    </>
  );
}

function NewRecordForm({ schema, onDone, onError }: { schema: Parameters<typeof InputForm>[0]["schema"]; onDone: () => void; onError: (t: string) => void }) {
  const [values, setValues] = useState<Record<string, string>>({});
  const [exp, setExp] = useState("");
  const [cof, setCof] = useState("");
  const [wear, setWear] = useState("");
  const [notes, setNotes] = useState("");
  const [predId, setPredId] = useState("");
  useEffect(() => { const init: Record<string, string> = {}; for (const inp of schema.inputs) init[inp.name] = inp.dtype === "number" ? String(Number((((inp.min ?? 0) + (inp.max ?? 0)) / 2).toFixed(2))) : Object.keys(inp.levels || {})[0] || ""; setValues(init); }, [schema]);
  const create = useMutation({
    mutationFn: () => apiPost("/api/validation", { experiment_id: exp, inputs: toInputs(schema, values), measured_cof: cof === "" ? null : Number(cof), measured_wear_rate: wear === "" ? null : Number(wear), prediction_id: predId ? Number(predId) : null, notes: notes || null }),
    onSuccess: () => { onDone(); setExp(""); setCof(""); setWear(""); setNotes(""); setPredId(""); },
    onError: (e) => onError(errorMessage(e)),
  });
  const anyModel = Object.values(schema.targets).some((t) => t.available);
  if (!anyModel) return <EmptyState title="Model not trained">A trained model is needed to compare measurements with predictions.</EmptyState>;
  return (
    <Panel title="Record an experimental result" subtitle="Steps 2–5 of the loop: select the tested condition, enter the experiment ID and the measured values; the prediction and error are computed and stored." glow>
      <div className="space-y-3">
        <Field label="Experiment identifier"><input className="input" value={exp} onChange={(e) => setExp(e.target.value)} placeholder="e.g. POD-2026-031" /></Field>
        <Field label="Link to an existing prediction ID (optional)" hint="Leave blank to generate the prediction for the entered conditions now"><input className="input" type="number" value={predId} onChange={(e) => setPredId(e.target.value)} /></Field>
        <InputForm schema={schema} values={values} onChange={setValues} />
        <Field label="Measured COF (–)"><input className="input" type="number" step="any" min={0} max={2} value={cof} onChange={(e) => setCof(e.target.value)} /></Field>
        <Field label="Measured wear rate (optional)" hint="Enter only if a wear quantity was measured with a documented definition and unit"><input className="input" type="number" step="any" min={0} value={wear} onChange={(e) => setWear(e.target.value)} /></Field>
        <Field label="Notes"><input className="input" value={notes} onChange={(e) => setNotes(e.target.value)} /></Field>
        <Button variant="primary" className="w-full" disabled={!exp || (cof === "" && wear === "")} loading={create.isPending} onClick={() => create.mutate()}>Save & compare with prediction</Button>
      </div>
    </Panel>
  );
}

function AugmentPanel({ records, datasets, onDone, onError }: { records: ValidationRecord[]; datasets: { id: number; name: string; latest_version_id: number | null }[]; onDone: (t: string) => void; onError: (t: string) => void }) {
  const eligible = records.filter((r) => r.status === "approved" && !r.added_to_version_id);
  const [dsId, setDsId] = useState<number | null>(datasets[0]?.id ?? null);
  const versions = useQuery({ queryKey: ["versions", dsId], queryFn: () => apiGet<VersionSummary[]>(`/api/datasets/${dsId}/versions`), enabled: dsId !== null });
  const [vid, setVid] = useState<number | null>(null);
  useEffect(() => { if (versions.data?.length) setVid(versions.data[versions.data.length - 1].id); }, [versions.data]);
  const [sel, setSel] = useState<number[]>([]);
  useEffect(() => setSel(eligible.map((r) => r.id)), [records]); // eslint-disable-line react-hooks/exhaustive-deps
  const [confirm, setConfirm] = useState(false);
  const augment = useMutation({ mutationFn: () => apiPost<VersionSummary>("/api/validation/augment", { dataset_version_id: vid, validation_ids: sel }), onSuccess: (v) => { setConfirm(false); onDone(`New dataset version v${v.version_number} (augmented, ${v.record_count} rows) created. Retrain on it from the Train Models page.`); }, onError: (e) => { setConfirm(false); onError(errorMessage(e)); } });
  return (
    <Panel title="Add approved results to a new dataset version" subtitle="Step 8: creates a new 'augmented' version (the parent is untouched). Then retrain and compare before promoting.">
      <div className="grid gap-3 md:grid-cols-2">
        <Field label="Dataset"><select className="input" value={dsId ?? ""} onChange={(e) => setDsId(Number(e.target.value))}>{datasets.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}</select></Field>
        <Field label="Parent version"><select className="input" value={vid ?? ""} onChange={(e) => setVid(Number(e.target.value))}>{versions.data?.map((v) => <option key={v.id} value={v.id}>v{v.version_number} · {v.kind} · {v.record_count} rows</option>)}</select></Field>
      </div>
      <div className="mt-3 text-sm text-[var(--muted)]">{eligible.length === 0 ? "No approved records waiting to be added." : `${eligible.length} approved record(s) eligible:`}</div>
      <div className="mt-1 flex flex-wrap gap-2">{eligible.map((r) => <label key={r.id} className="flex items-center gap-1 text-sm"><input type="checkbox" checked={sel.includes(r.id)} onChange={(e) => setSel(e.target.checked ? [...sel, r.id] : sel.filter((x) => x !== r.id))} />{r.experiment_id}</label>)}</div>
      <Button variant="primary" className="mt-3" disabled={!sel.length || !vid} onClick={() => setConfirm(true)}>Create augmented version</Button>
      <ConfirmDialog open={confirm} onClose={() => setConfirm(false)} onConfirm={() => augment.mutate()} title="Create a new dataset version?" confirmLabel="Create version" loading={augment.isPending}>{sel.length} approved validation record(s) will be appended to a copy of the selected version. The original version is not modified.</ConfirmDialog>
    </Panel>
  );
}

function RetrainComparePanel() {
  const models = useQuery({ queryKey: ["models", "cmp"], queryFn: () => apiGet<ModelSummary[]>("/api/models", { target: "cof" }) });
  const prod = models.data?.find((m) => m.status === "production");
  const candidates = (models.data || []).filter((m) => m.status === "candidate" && m.is_best_of_run).slice(0, 3);
  return (
    <Panel title="Steps 9–10: retrain and compare before promoting" subtitle="New models are never promoted automatically. Compare the newest best candidate with the current production model, then promote explicitly on the Model Evaluation page." actions={<Link href="/train" className="btn btn-sm">Retrain</Link>}>
      {!models.data?.length && <div className="text-sm text-[var(--muted)]">No COF models yet.</div>}
      {models.data && models.data.length > 0 && (
        <div className="overflow-auto">
          <table className="table"><thead><tr><th>Model</th><th>Status</th><th>Dataset version</th><th>Val RMSE (CV)</th><th>Val R²</th><th>Test RMSE</th><th>Test R²</th><th>Trained</th></tr></thead>
            <tbody>{[prod, ...candidates].filter(Boolean).map((m) => <tr key={m!.id}><td className="text-white">#{m!.id} {MODEL_LABELS[m!.algorithm] || m!.algorithm}</td><td><Badge tone={m!.status === "production" ? "ok" : "warn"}>{m!.status}</Badge></td><td>{m!.dataset_version_id}</td><td>{fmtNum(m!.metrics.validation?.rmse, 5)}</td><td>{fmtNum(m!.metrics.validation?.r2)}</td><td>{fmtNum(m!.metrics.test?.rmse, 5)}</td><td>{fmtNum(m!.metrics.test?.r2)}</td><td>{fmtDate(m!.trained_at)}</td></tr>)}</tbody>
          </table>
          <div className="mt-2 text-xs text-[var(--muted)]">Metrics of models trained on different dataset versions are computed on different test sets; compare with care. <Link href="/evaluation" className="text-[var(--accent)]">Open Model Evaluation</Link> to promote.</div>
        </div>
      )}
    </Panel>
  );
}
