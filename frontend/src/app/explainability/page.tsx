"use client";

import { useMutation, useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { apiGet, apiPost, errorMessage } from "@/lib/api";
import { Alert, Badge, Button, EmptyState, ErrorBox, KeyValue, Loading, PageHeader, Panel } from "@/components/ui";
import { Bars, ChartBox, ScatterXY } from "@/components/charts";
import { featureLabel, fmtNum, MODEL_LABELS } from "@/lib/format";
import { ModelSummary, PredictSchema, useModels, usePredictSchema } from "@/lib/hooks";
import { InputForm, toInputs } from "@/components/predict-form";

interface GlobalExplain { model: ModelSummary; method: string | null; error?: string; base_value?: number; n_samples?: number; importance: { feature: string; label: string; unit: string; mean_abs_shap: number; mean_shap: number; share: number }[]; summary_sample?: Record<string, [number | string | null, number][]>; level_effects?: Record<string, { level: string; mean_shap: number; n: number }[]>; disclaimer: string }
interface LocalExplain { base_value: number; prediction: number; contributions: { feature: string; label: string; value: unknown; unit: string; contribution: number }[]; method: string; warnings: string[]; unit: string; disclaimer: string }

export default function ExplainabilityPage() {
  const models = useModels();
  const schema = usePredictSchema();
  const [modelId, setModelId] = useState<number | null>(null);
  useEffect(() => { if (models.data?.length && modelId === null) setModelId((models.data.find((m) => m.status === "production") || models.data.find((m) => m.is_best_of_run) || models.data[0]).id); }, [models.data, modelId]);
  const g = useQuery({ queryKey: ["explain-global", modelId], queryFn: () => apiGet<GlobalExplain>(`/api/models/${modelId}/explain/global`), enabled: modelId !== null });
  const compare = useQuery({ queryKey: ["explain-compare"], queryFn: () => apiGet<Record<string, { available: boolean; model: ModelSummary | null; importance: GlobalExplain["importance"]; method: string | null; reason: string | null }>>("/api/models/compare/explainability") });
  if (models.isLoading) return <Loading />;
  if (models.isError) return <ErrorBox error={models.error} />;
  if (!models.data?.length) return <><PageHeader title="Explainability" /><EmptyState title="Model not trained">Train a model to inspect SHAP feature importance and local explanations.</EmptyState></>;
  const active = models.data.filter((m) => m.status !== "archived");
  return (
    <>
      <PageHeader title="Explainability" subtitle="SHAP-based global importance, summary distributions and single-prediction explanations. These describe how the fitted model uses its inputs — they do not establish physical causation." actions={<select className="input" style={{ width: 320 }} value={modelId ?? ""} onChange={(e) => setModelId(Number(e.target.value))}>{(active.length ? active : models.data).map((m) => <option key={m.id} value={m.id}>#{m.id} {MODEL_LABELS[m.algorithm] || m.algorithm} · {m.target} · {m.status}</option>)}</select>} />
      {g.isLoading && <Loading label="Loading SHAP values…" />}
      {g.isError && <ErrorBox error={g.error} />}
      {g.data && (
        <>
          <Alert tone="info" className="mb-4">{g.data.disclaimer}</Alert>
          {g.data.error || !g.data.importance?.length ? (
            <Alert tone="warn" title="Explanation unavailable for this model">{g.data.error || "No SHAP values were computed."}</Alert>
          ) : (
            <>
              <div className="grid gap-4 lg:grid-cols-2">
                <ChartBox title="Global feature importance (mean |SHAP|)" subtitle={`${g.data.method} · ${g.data.n_samples} training rows`} height={80 + 44 * g.data.importance.length}>
                  <Bars data={g.data.importance.map((i) => ({ name: i.label, value: i.mean_abs_shap }))} xKey="name" bars={[{ key: "value", name: "mean |SHAP|" }]} horizontal formatter={(v) => v.toFixed(5)} />
                </ChartBox>
                <Panel title="Feature contribution analysis">
                  <KeyValue items={g.data.importance.map((i) => [i.label, `${fmtNum(i.mean_abs_shap, 5)} (${(100 * i.share).toFixed(1)} % of total)`])} />
                  {g.data.level_effects && Object.entries(g.data.level_effects).map(([f, lv]) => (
                    <div key={f} className="mt-4">
                      <div className="label mb-1">Mean effect of each {featureLabel(f).toLowerCase()} level</div>
                      <KeyValue items={lv.map((l) => [l.level, `${l.mean_shap >= 0 ? "+" : ""}${fmtNum(l.mean_shap, 4)} (${l.n} rows)`])} />
                    </div>
                  ))}
                  <div className="mt-3 text-xs text-[var(--muted)]">Base value (average model output over the background sample): {fmtNum(g.data.base_value)}</div>
                </Panel>
              </div>
              <div className="mt-4 grid gap-4 lg:grid-cols-2">
                {g.data.summary_sample && Object.entries(g.data.summary_sample).map(([f, pts]) => {
                  const numeric = pts.every((p) => typeof p[0] === "number" || p[0] === null);
                  const data = numeric ? pts.filter((p) => p[0] !== null).map((p) => ({ x: p[0] as number, y: p[1] })) : null;
                  const levels = !numeric ? Object.entries(pts.reduce((acc, p) => { const k = String(p[0]); (acc[k] ||= []).push(p[1]); return acc; }, {} as Record<string, number[]>)) : null;
                  return (
                    <ChartBox key={f} title={`SHAP summary — ${featureLabel(f)}`} subtitle={numeric ? "SHAP value vs feature value (each point = one training row)" : "SHAP value distribution per level (mean ± range)"} height={260}>
                      {numeric ? <ScatterXY points={data!} xLabel={featureLabel(f)} yLabel="SHAP value" /> : <Bars data={levels!.map(([k, v]) => ({ level: k, mean: v.reduce((a, b) => a + b, 0) / v.length, min: Math.min(...v), max: Math.max(...v) }))} xKey="level" bars={[{ key: "mean", name: "mean SHAP" }, { key: "min", name: "min" }, { key: "max", name: "max" }]} />}
                    </ChartBox>
                  );
                })}
              </div>
            </>
          )}
          {schema.data && <LocalPanel modelId={modelId!} model={g.data.model} schema={schema.data} />}
        </>
      )}
      <Panel title="Comparison of feature effects across COF and wear-rate models" className="mt-4">
        {compare.data && (
          <div className="grid gap-4 md:grid-cols-2">
            {Object.entries(compare.data).map(([t, c]) => (
              <div key={t} className="rounded-xl border border-[var(--border)] p-3">
                <div className="mb-2 font-semibold text-white">{t === "cof" ? "COF model" : "Wear-rate model"} {c.model && <Badge tone={c.model.status === "production" ? "ok" : "accent"}>{MODEL_LABELS[c.model.algorithm] || c.model.algorithm} · {c.model.status}</Badge>}</div>
                {c.available ? <KeyValue items={c.importance.map((i) => [i.label, `${(100 * i.share).toFixed(1)} %`])} /> : <Alert tone="warn">{c.reason}</Alert>}
              </div>
            ))}
          </div>
        )}
      </Panel>
    </>
  );
}

function LocalPanel({ modelId, model, schema }: { modelId: number; model: ModelSummary; schema: PredictSchema }) {
  const [values, setValues] = useState<Record<string, string>>({});
  useEffect(() => { const init: Record<string, string> = {}; for (const inp of schema.inputs) init[inp.name] = inp.dtype === "number" ? String(Number((((inp.min ?? 0) + (inp.max ?? 0)) / 2).toFixed(2))) : Object.keys(inp.levels || {})[0] || ""; setValues(init); }, [schema]);
  const local = useMutation({ mutationFn: () => apiPost<LocalExplain>(`/api/models/${modelId}/explain/local`, { inputs: toInputs(schema, values) }) });
  const sameTarget = schema.targets[model.target]?.available;
  return (
    <div className="mt-4 grid gap-4 lg:grid-cols-[360px_1fr]">
      <Panel title="Local explanation — choose conditions" subtitle={`Explains one prediction of model #${modelId} (${model.target})`}>
        {!sameTarget && <Alert tone="warn" className="mb-3">The input form follows the active model schema; this model predicts {model.target}.</Alert>}
        <InputForm schema={schema} values={values} onChange={setValues} />
        <Button variant="primary" className="mt-3 w-full" loading={local.isPending} onClick={() => local.mutate()}>Explain prediction</Button>
      </Panel>
      <div>
        {local.isError && <Alert tone="danger">{errorMessage(local.error)}</Alert>}
        {local.data && (
          <>
            <ChartBox title="Feature contributions for this prediction" subtitle={`base ${fmtNum(local.data.base_value)} + Σ contributions = prediction ${fmtNum(local.data.prediction)} ${local.data.unit || ""} · ${local.data.method}`} height={80 + 44 * local.data.contributions.length}>
              <Bars data={local.data.contributions.map((c) => ({ name: `${c.label} = ${typeof c.value === "number" ? fmtNum(c.value, 2) : c.value}${c.unit ? " " + c.unit : ""}`, contribution: c.contribution }))} xKey="name" bars={[{ key: "contribution", name: "SHAP contribution" }]} horizontal formatter={(v) => v.toFixed(5)} />
            </ChartBox>
            {local.data.warnings.length > 0 && <Alert tone="warn" className="mt-3">{local.data.warnings.join(" ")}</Alert>}
            <div className="mt-2 text-xs text-[var(--muted)]">{local.data.disclaimer}</div>
          </>
        )}
        {!local.data && !local.isError && <EmptyState title="No local explanation yet">Choose conditions and click “Explain prediction”.</EmptyState>}
      </div>
    </div>
  );
}
