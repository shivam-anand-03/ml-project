"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { apiGet, apiPost, errorMessage } from "@/lib/api";
import { Alert, Badge, Button, ConfirmDialog, EmptyState, ErrorBox, KeyValue, Loading, PageHeader, Panel, Spinner } from "@/components/ui";
import { Bars, ChartBox, Histogram, ScatterXY } from "@/components/charts";
import { fmtDate, fmtNum, fmtPct, MODEL_LABELS } from "@/lib/format";
import { ModelSummary, useModels } from "@/lib/hooks";

interface Metrics extends ModelSummary { fold_rmse: number[]; test_predictions: { actual: number; predicted: number; residual: number; material?: string; time_s?: number }[]; oof_predictions: { actual: number; predicted: number; residual: number }[]; explanations: Record<string, string> }

export default function EvaluationPage() {
  const qc = useQueryClient();
  const [target, setTarget] = useState("cof");
  const models = useModels({ target });
  const [selected, setSelected] = useState<number | null>(null);
  const [confirm, setConfirm] = useState<{ action: "promote" | "archive"; id: number } | null>(null);
  const [msg, setMsg] = useState<{ tone: "ok" | "danger"; text: string } | null>(null);
  useEffect(() => { if (models.data?.length && (selected === null || !models.data.some((m) => m.id === selected))) setSelected((models.data.find((m) => m.status === "production") || models.data.find((m) => m.is_best_of_run) || models.data[0]).id); }, [models.data, selected]);
  const metrics = useQuery({ queryKey: ["metrics", selected], queryFn: () => apiGet<Metrics>(`/api/models/${selected}/metrics`), enabled: selected !== null });
  const act = useMutation({
    mutationFn: (c: { action: string; id: number }) => apiPost(`/api/models/${c.id}/${c.action}`),
    onSuccess: (_, c) => { setMsg({ tone: "ok", text: c.action === "promote" ? "Model promoted to production. The previous production model was archived (not deleted)." : "Model archived." }); qc.invalidateQueries({ queryKey: ["models"] }); setConfirm(null); },
    onError: (e) => { setMsg({ tone: "danger", text: errorMessage(e) }); setConfirm(null); },
  });
  const explanations = metrics.data?.explanations;

  return (
    <>
      <PageHeader title="Model Evaluation" subtitle="Leaderboard, cross-validation performance, held-out test results, residual analysis and explicit production promotion. All metrics are computed by the evaluation pipeline — never placeholders." actions={<select className="input" style={{ width: 220 }} value={target} onChange={(e) => { setTarget(e.target.value); setSelected(null); }}><option value="cof">COF models</option><option value="wear_rate">Wear-rate models</option></select>} />
      {msg && <Alert tone={msg.tone} className="mb-4">{msg.text}</Alert>}
      {models.isLoading && <Loading />}
      {models.isError && <ErrorBox error={models.error} />}
      {models.data?.length === 0 && <EmptyState title="Model not trained">{target === "wear_rate" ? "No wear-rate model exists. The bundled dataset has no wear measurements, so a wear-rate model cannot be trained from it." : "Train models on the Train Models page to populate the leaderboard."}</EmptyState>}
      {models.data && models.data.length > 0 && (
        <>
          <Panel title="Leaderboard" subtitle="Ranked by out-of-fold cross-validation RMSE (validation); the test columns are the untouched held-out set." className="mb-4">
            <div className="overflow-auto">
              <table className="table">
                <thead><tr><th>ID</th><th>Model</th><th>Status</th><th>Dataset v.</th><th>Split</th><th>CV RMSE ± SD</th><th>Val R²</th><th>Val MAE</th><th>Val MAPE</th><th>Test R²</th><th>Test RMSE</th><th>Test MAE</th><th>Test MAPE</th><th>Train (s)</th><th>Trained</th><th></th></tr></thead>
                <tbody>
                  {[...models.data].sort((a, b) => (a.cv_rmse_mean ?? 9) - (b.cv_rmse_mean ?? 9)).map((m) => {
                    const v = m.metrics.validation || {}, t = m.metrics.test || {};
                    return (
                      <tr key={m.id} className={selected === m.id ? "bg-[var(--accent-soft)]" : ""} onClick={() => setSelected(m.id)} style={{ cursor: "pointer" }}>
                        <td>{m.id}</td><td className="font-medium text-white">{MODEL_LABELS[m.algorithm] || m.algorithm}{m.is_best_of_run && <Badge tone="accent" className="ml-2">best of run</Badge>}</td>
                        <td><Badge tone={m.status === "production" ? "ok" : m.status === "archived" ? "muted" : "warn"}>{m.status}</Badge></td>
                        <td>{m.dataset_version_id}</td><td>{m.split?.strategy}</td><td>{fmtNum(m.cv_rmse_mean, 5)} ± {fmtNum(m.cv_rmse_std, 5)}</td>
                        <td>{fmtNum(v.r2)}</td><td>{fmtNum(v.mae, 5)}</td><td>{fmtPct(v.mape)}</td><td>{fmtNum(t.r2)}</td><td>{fmtNum(t.rmse, 5)}</td><td>{fmtNum(t.mae, 5)}</td><td>{fmtPct(t.mape)}</td>
                        <td>{fmtNum(m.training_duration_s, 1)}</td><td>{fmtDate(m.trained_at)}</td>
                        <td className="flex gap-1">{m.status !== "production" && <Button size="sm" variant="primary" onClick={(e) => { e.stopPropagation(); setConfirm({ action: "promote", id: m.id }); }}>Promote</Button>}{m.status !== "archived" && <Button size="sm" onClick={(e) => { e.stopPropagation(); setConfirm({ action: "archive", id: m.id }); }}>Archive</Button>}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </Panel>
          {metrics.isLoading && <Spinner />}
          {metrics.data && <ModelDetail m={metrics.data} />}
          {explanations && (
            <Panel title="What the metrics mean" className="mt-4">
              <ul className="space-y-1 text-sm text-[var(--muted)]">{Object.entries(explanations).map(([k, v]) => <li key={k}><span className="font-semibold uppercase text-white">{k}</span>: {v}</li>)}</ul>
            </Panel>
          )}
        </>
      )}
      <ConfirmDialog open={!!confirm} onClose={() => setConfirm(null)} onConfirm={() => confirm && act.mutate(confirm)} title={confirm?.action === "promote" ? "Promote this model to production?" : "Archive this model?"} confirmLabel={confirm?.action === "promote" ? "Promote" : "Archive"} danger={confirm?.action === "archive"} loading={act.isPending}>
        {confirm?.action === "promote" ? "The current production model for this target will be archived, not deleted. Predictions will use the promoted version from now on." : "Archived models are kept for traceability but are no longer used for predictions."}
      </ConfirmDialog>
    </>
  );
}

function ModelDetail({ m }: { m: Metrics }) {
  const tr = m.metrics.train || {}, v = m.metrics.validation || {}, t = m.metrics.test || {};
  const resid = m.test_predictions.map((p) => p.residual);
  const bins = histogram(resid, 30);
  const foldData = m.fold_rmse.map((r, i) => ({ fold: `fold ${i + 1}`, rmse: r }));
  return (
    <div className="space-y-4">
      <div className="grid gap-4 lg:grid-cols-3">
        <Panel title={`${MODEL_LABELS[m.algorithm] || m.algorithm} · ${m.version_label}`} subtitle={m.split?.description}>
          <KeyValue items={[["Status", <Badge key="s" tone={m.status === "production" ? "ok" : "warn"}>{m.status}</Badge>], ["Features", m.features.join(", ")], ["Hyper-parameters", JSON.stringify(m.hyperparameters)], ["Training rows / test rows", `${m.n_train} / ${m.n_test}`], ["Held-out runs", m.split?.test_groups?.join(", ") || "–"], ["Training duration", `${fmtNum(m.training_duration_s, 1)} s`], ["Interval", `${m.interval?.method || "–"}; ±${fmtNum(m.interval?.half_width, 4)}; empirical test coverage ${fmtPct((m.interval?.test_coverage ?? 0) * 100, 1)}`]]} />
          {m.split?.warnings?.map((w, i) => <Alert key={i} tone="warn" className="mt-2">{w}</Alert>)}
          {m.split?.small_dataset && <Alert tone="warn" className="mt-2">Small training set: the held-out test estimate is unreliable; rely on cross-validation.</Alert>}
        </Panel>
        <Panel title="Metrics by split" subtitle="train = fit quality; validation = out-of-fold CV; test = untouched held-out set">
          <table className="table"><thead><tr><th>Split</th><th>n</th><th>R²</th><th>MAE</th><th>RMSE</th><th>MAPE</th></tr></thead>
            <tbody>{[["train", tr], ["validation", v], ["test", t]].map(([k, s]) => { const ss = s as Record<string, number | null | undefined>; return <tr key={String(k)}><td className="font-medium text-white">{String(k)}</td><td>{ss.n}</td><td>{fmtNum(ss.r2)}</td><td>{fmtNum(ss.mae, 5)}</td><td>{fmtNum(ss.rmse, 5)}</td><td>{fmtPct(ss.mape)}{ss.mape_coverage != null && ss.mape_coverage < 1 && <span className="text-xs text-[var(--muted)]"> ({fmtPct(ss.mape_coverage * 100, 0)} of rows)</span>}</td></tr>; })}</tbody>
          </table>
        </Panel>
        <ChartBox title="Cross-validation RMSE per fold" subtitle="Spread across folds indicates robustness" height={240}>
          {foldData.length ? <Bars data={foldData} xKey="fold" bars={[{ key: "rmse", name: "RMSE" }]} formatter={(x) => x.toFixed(5)} /> : <div className="text-sm text-[var(--muted)]">No fold data</div>}
        </ChartBox>
      </div>
      <div className="grid gap-4 lg:grid-cols-3">
        <ChartBox title="Actual vs predicted — held-out test set" subtitle="Measured on x, predicted on y; dashed = perfect agreement" height={320}>
          <ScatterXY points={m.test_predictions.map((p) => ({ x: p.actual, y: p.predicted, group: p.material ?? null }))} xLabel="Measured" yLabel="Predicted" identity groups />
        </ChartBox>
        <ChartBox title="Residuals vs predicted (test)" subtitle="residual = measured − predicted" height={320}>
          <ScatterXY points={m.test_predictions.map((p) => ({ x: p.predicted, y: p.residual, group: p.material ?? null }))} xLabel="Predicted" yLabel="Residual" groups />
        </ChartBox>
        <ChartBox title="Error distribution (test residuals)" height={320}>
          <Histogram bins={bins} xLabel="residual" />
        </ChartBox>
      </div>
      <ChartBox title="Actual vs predicted — out-of-fold cross-validation" subtitle="Validation predictions made while each row's fold (or run) was held out" height={300}>
        <ScatterXY points={m.oof_predictions.map((p) => ({ x: p.actual, y: p.predicted }))} xLabel="Measured" yLabel="Predicted (out-of-fold)" identity color="#9b7dff" />
      </ChartBox>
    </div>
  );
}

function histogram(values: number[], n: number) {
  if (!values.length) return [];
  const lo = Math.min(...values), hi = Math.max(...values), w = (hi - lo) / n || 1;
  const bins = Array.from({ length: n }, (_, i) => ({ x0: lo + i * w, x1: lo + (i + 1) * w, count: 0 }));
  for (const v of values) bins[Math.min(n - 1, Math.floor((v - lo) / w))].count++;
  return bins;
}
