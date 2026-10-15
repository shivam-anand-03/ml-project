"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { apiGet, apiPost, errorMessage } from "@/lib/api";
import { Alert, Badge, Button, EmptyState, ErrorBox, Field, Loading, PageHeader, Panel, Spinner } from "@/components/ui";
import { fmtDate, fmtNum, MODEL_LABELS } from "@/lib/format";
import { TrainingRunView, useDatasets, useRuns, VersionSummary } from "@/lib/hooks";

interface Candidate { key: string; label: string; family: string; grid: Record<string, string[]>; quick_grid: Record<string, string[]>; notes: string }

export default function TrainPage() {
  const qc = useQueryClient();
  const datasets = useDatasets();
  const candidates = useQuery({ queryKey: ["candidates"], queryFn: () => apiGet<Candidate[]>("/api/models/candidates") });
  const settings = useQuery({ queryKey: ["settings"], queryFn: () => apiGet<{ settings: Record<string, unknown> }>("/api/settings") });
  const runs = useRuns();
  const [dsId, setDsId] = useState<number | null>(null);
  const versions = useQuery({ queryKey: ["versions", dsId], queryFn: () => apiGet<VersionSummary[]>(`/api/datasets/${dsId}/versions`), enabled: dsId !== null });
  const [form, setForm] = useState({ dataset_version_id: 0, target: "cof", models: ["linear", "polynomial", "random_forest", "svr", "xgboost", "mlp"], split_strategy: "auto", test_fraction: 0.2, n_folds: 5, search_mode: "standard", interval_level: 0.9, notes: "", features: [] as string[] });
  const [msg, setMsg] = useState<{ tone: "ok" | "danger"; text: string } | null>(null);
  useEffect(() => { if (dsId === null && datasets.data?.length) setDsId(datasets.data[0].id); }, [datasets.data, dsId]);
  useEffect(() => { if (versions.data?.length) { const v = [...versions.data].reverse().find((x) => x.kind !== "raw") || versions.data[versions.data.length - 1]; setForm((f) => ({ ...f, dataset_version_id: v.id })); } }, [versions.data]);
  useEffect(() => { const s = settings.data?.settings; if (s) setForm((f) => ({ ...f, split_strategy: String(s.default_split_strategy || "auto"), search_mode: String(s.default_search_mode || "standard"), models: (s.default_models as string[]) || f.models, test_fraction: Number(s.default_test_fraction ?? 0.2), n_folds: Number(s.default_n_folds ?? 5), interval_level: Number(s.interval_level ?? 0.9) })); }, [settings.data]);
  const version = versions.data?.find((v) => v.id === form.dataset_version_id);
  const targetAvailable = version?.feature_summary?.targets?.[form.target]?.available;
  const inputs = version?.feature_summary?.inputs || {};
  const candidateFeatures = Object.entries(inputs).filter(([k, v]) => v.present && v.varies && ["input", "metadata"].includes(v.role) && k !== "test_run_id").map(([k]) => k);
  const train = useMutation({
    mutationFn: () => apiPost<TrainingRunView>("/api/models/train", { ...form, features: form.features.length ? form.features : null, notes: form.notes || null }),
    onSuccess: (r) => { setMsg({ tone: "ok", text: `Training run #${r.id} queued. It runs in the background; predictions stay available meanwhile.` }); qc.invalidateQueries({ queryKey: ["runs"] }); },
    onError: (e) => setMsg({ tone: "danger", text: errorMessage(e) }),
  });

  if (datasets.isLoading) return <Loading />;
  if (!datasets.data?.length) return <EmptyState title="No dataset">Upload or import a dataset first.</EmptyState>;
  return (
    <>
      <PageHeader title="Train Models" subtitle="Train and compare candidate regressors on identical, leakage-free splits. Preprocessing is fitted on training folds only; the held-out test set is evaluated once after model selection." />
      {msg && <Alert tone={msg.tone} className="mb-4">{msg.text}</Alert>}
      <div className="grid gap-4 xl:grid-cols-[420px_1fr]">
        <Panel title="Training configuration">
          <div className="space-y-3">
            <Field label="Dataset"><select className="input" value={dsId ?? ""} onChange={(e) => setDsId(Number(e.target.value))}>{datasets.data.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}</select></Field>
            <Field label="Dataset version"><select className="input" value={form.dataset_version_id} onChange={(e) => setForm({ ...form, dataset_version_id: Number(e.target.value) })}>{versions.data?.map((v) => <option key={v.id} value={v.id}>v{v.version_number} · {v.kind} · {v.usable_record_count} usable rows</option>)}</select></Field>
            <Field label="Target"><select className="input" value={form.target} onChange={(e) => setForm({ ...form, target: e.target.value })}><option value="cof">Coefficient of friction (COF)</option><option value="wear_rate">Wear rate</option></select></Field>
            {version && !targetAvailable && <Alert tone="warn" title="Target unavailable">This version has no {form.target === "cof" ? "COF" : "wear-rate"} measurements, so this model cannot be trained.</Alert>}
            <Field label="Candidate models">
              <div className="grid grid-cols-2 gap-1">
                {candidates.data?.map((c) => (
                  <label key={c.key} className="flex items-center gap-2 text-sm text-[var(--muted)]" title={c.notes}><input type="checkbox" checked={form.models.includes(c.key)} onChange={(e) => setForm({ ...form, models: e.target.checked ? [...form.models, c.key] : form.models.filter((m) => m !== c.key) })} />{c.label}</label>
                ))}
              </div>
            </Field>
            <Field label="Input features" hint={`Automatic selection picks varying inputs and drops fixed conditions and collinear duplicates. Candidates in this version: ${candidateFeatures.join(", ") || "–"}`}>
              <div className="grid grid-cols-2 gap-1">
                <label className="flex items-center gap-2 text-sm text-[var(--muted)]"><input type="radio" checked={form.features.length === 0} onChange={() => setForm({ ...form, features: [] })} />Automatic</label>
                {candidateFeatures.map((f) => <label key={f} className="flex items-center gap-2 text-sm text-[var(--muted)]"><input type="checkbox" checked={form.features.includes(f)} onChange={(e) => setForm({ ...form, features: e.target.checked ? [...form.features, f] : form.features.filter((x) => x !== f) })} />{f}</label>)}
              </div>
            </Field>
            <div className="grid grid-cols-2 gap-3">
              <Field label="Split strategy" hint="grouped = whole test runs held out (recommended for replicate time series)">
                <select className="input" value={form.split_strategy} onChange={(e) => setForm({ ...form, split_strategy: e.target.value })}><option value="auto">auto</option><option value="grouped">grouped by test run</option><option value="random">row-wise random</option><option value="chronological">chronological</option></select>
              </Field>
              <Field label="Search mode" hint="standard = full documented grids; quick = one configuration per model">
                <select className="input" value={form.search_mode} onChange={(e) => setForm({ ...form, search_mode: e.target.value })}><option value="standard">standard</option><option value="quick">quick</option></select>
              </Field>
              <Field label="Test fraction"><input className="input" type="number" step="0.05" min={0.05} max={0.5} value={form.test_fraction} onChange={(e) => setForm({ ...form, test_fraction: Number(e.target.value) })} /></Field>
              <Field label="CV folds"><input className="input" type="number" min={2} max={10} value={form.n_folds} onChange={(e) => setForm({ ...form, n_folds: Number(e.target.value) })} /></Field>
              <Field label="Interval level"><input className="input" type="number" step="0.05" min={0.5} max={0.99} value={form.interval_level} onChange={(e) => setForm({ ...form, interval_level: Number(e.target.value) })} /></Field>
              <Field label="Notes"><input className="input" value={form.notes} onChange={(e) => setForm({ ...form, notes: e.target.value })} /></Field>
            </div>
            <Button variant="primary" className="w-full" disabled={!targetAvailable || form.models.length === 0} loading={train.isPending} onClick={() => train.mutate()}>Start training run</Button>
          </div>
        </Panel>
        <div className="space-y-4">
          <Panel title="Hyper-parameter search grids" subtitle="Grid search (scoring: RMSE) with cross-validation on the training runs">
            <div className="grid gap-2 md:grid-cols-2">
              {candidates.data?.map((c) => (
                <div key={c.key} className="rounded-xl border border-[var(--border)] p-3 text-xs">
                  <div className="font-semibold text-white">{c.label} <Badge>{c.family}</Badge></div>
                  <div className="mt-1 text-[var(--muted)]">{Object.keys(c.grid).length ? Object.entries(c.grid).map(([k, v]) => `${k}: [${v.join(", ")}]`).join(" · ") : "no hyper-parameters"}</div>
                </div>
              ))}
            </div>
          </Panel>
          <Panel title="Training runs" subtitle="Runs execute in the background; this list refreshes automatically">
            {runs.isLoading && <Spinner />}
            {runs.isError && <ErrorBox error={runs.error} />}
            {runs.data?.length === 0 && <div className="text-sm text-[var(--muted)]">No training runs yet.</div>}
            <div className="space-y-3">{runs.data?.map((r) => <RunCard key={r.id} run={r} />)}</div>
          </Panel>
        </div>
      </div>
    </>
  );
}

function RunCard({ run }: { run: TrainingRunView }) {
  const [open, setOpen] = useState(false);
  const lb = run.summary?.leaderboard || [];
  const done = run.summary?.models_done?.length || 0, total = run.summary?.n_models || (run.config.models as string[])?.length || 0;
  return (
    <div className="rounded-xl border border-[var(--border)] p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-sm"><span className="font-semibold text-white">Run #{run.id}</span> · target {run.target} · version {run.dataset_version_id} · {String(run.config.split_strategy)} split · {String(run.config.search_mode)} search · <span className="text-[var(--muted)]">{fmtDate(run.created_at)}</span></div>
        <div className="flex items-center gap-2">
          <Badge tone={run.status === "completed" ? "ok" : run.status === "failed" ? "danger" : "warn"}>{run.status}{(run.status === "running" || run.status === "queued") && ` ${done}/${total}`}</Badge>
          <Button size="sm" onClick={() => setOpen((o) => !o)}>{open ? "Hide log" : "Show log"}</Button>
        </div>
      </div>
      {run.status === "running" && <div className="mt-2 h-1.5 w-full overflow-hidden rounded bg-[var(--border)]"><div className="h-full bg-[var(--accent)] transition-all" style={{ width: `${total ? (100 * done) / total : 10}%` }} /></div>}
      {run.error && <Alert tone="danger" className="mt-2">{run.error}</Alert>}
      {run.summary?.split && <div className="mt-2 text-xs text-[var(--muted)]">Features: {run.summary.features?.join(", ")} · {run.summary.split.description}</div>}
      {run.summary?.notes?.map((n, i) => <div key={i} className="mt-1 text-xs text-[var(--warn)]">• {n}</div>)}
      {lb.length > 0 && (
        <div className="mt-2 overflow-auto">
          <table className="table">
            <thead><tr><th>Model</th><th>Validation RMSE (CV)</th><th>Validation R²</th><th>Test RMSE</th><th>Test R²</th><th>Time (s)</th><th></th></tr></thead>
            <tbody>{lb.map((e) => <tr key={e.algorithm}><td>{MODEL_LABELS[e.algorithm] || e.label}</td><td>{fmtNum(e.validation_rmse, 5)}</td><td>{fmtNum(e.validation_r2)}</td><td>{fmtNum(e.test_rmse, 5)}</td><td>{fmtNum(e.test_r2)}</td><td>{fmtNum(e.duration_s, 1)}</td><td>{e.error ? <Badge tone="danger">failed</Badge> : e.algorithm === run.summary.best_algorithm ? <Badge tone="ok">best validated</Badge> : null}</td></tr>)}</tbody>
          </table>
        </div>
      )}
      {open && <pre className="mt-2 max-h-64 overflow-auto rounded-lg bg-black/40 p-2 text-[11px] text-[var(--muted)]">{run.log.map((l) => `${l.t.slice(11, 19)}  ${l.msg}`).join("\n") || "(no log yet)"}</pre>}
    </div>
  );
}
