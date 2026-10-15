"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { apiGet, apiPut, errorMessage } from "@/lib/api";
import { Alert, Button, DataTable, ErrorBox, Field, KeyValue, Loading, PageHeader, Panel } from "@/components/ui";
import { MODEL_LABELS } from "@/lib/format";

interface SettingsView { settings: Record<string, unknown> & { project: Record<string, string>; default_models: string[] }; defaults: Record<string, unknown>; available_models: string[]; environment: Record<string, unknown> }

export default function SettingsPage() {
  const qc = useQueryClient();
  const q = useQuery({ queryKey: ["settings"], queryFn: () => apiGet<SettingsView>("/api/settings") });
  const dict = useQuery({ queryKey: ["data-dictionary"], queryFn: () => apiGet<{ features: Record<string, unknown>[]; material_catalogue: Record<string, Record<string, string>>; equations: { name: string; formula: string; unit: string; note: string }[] }>("/api/settings/data-dictionary") });
  const [form, setForm] = useState<Record<string, unknown> | null>(null);
  const [msg, setMsg] = useState<{ tone: "ok" | "danger"; text: string } | null>(null);
  useEffect(() => { if (q.data) setForm(q.data.settings); }, [q.data]);
  const save = useMutation({ mutationFn: () => apiPut("/api/settings", form), onSuccess: () => { qc.invalidateQueries({ queryKey: ["settings"] }); qc.invalidateQueries({ queryKey: ["predict-schema"] }); setMsg({ tone: "ok", text: "Settings saved." }); }, onError: (e) => setMsg({ tone: "danger", text: errorMessage(e) }) });
  if (q.isLoading || !form) return <Loading />;
  if (q.isError) return <ErrorBox error={q.error} />;
  const s = form as SettingsView["settings"];
  const set = (k: string, v: unknown) => setForm({ ...form, [k]: v });
  return (
    <>
      <PageHeader title="Settings" subtitle="Prediction-domain policy, training defaults, project information, data dictionary and environment." />
      {msg && <Alert tone={msg.tone} className="mb-4">{msg.text}</Alert>}
      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Prediction domain policy" subtitle="How far outside the experimental data a prediction may go">
          <div className="space-y-3">
            <Field label="Extrapolation margin (fraction of training range)" hint="Inputs within this margin outside the range are predicted with a warning"><input className="input" type="number" step="0.01" min={0} max={1} value={Number(s.extrapolation_margin)} onChange={(e) => set("extrapolation_margin", Number(e.target.value))} /></Field>
            <Field label="Fixed-condition tolerance (relative)" hint="e.g. load entered within ±5 % of the fixed 50 N is accepted"><input className="input" type="number" step="0.01" min={0} max={1} value={Number(s.fixed_condition_tolerance)} onChange={(e) => set("fixed_condition_tolerance", Number(e.target.value))} /></Field>
            <label className="flex items-center gap-2 text-sm text-[var(--muted)]"><input type="checkbox" checked={Boolean(s.allow_far_extrapolation)} onChange={(e) => set("allow_far_extrapolation", e.target.checked)} />Allow far extrapolation (predict with a strong warning instead of refusing)</label>
            <Field label="Prediction interval level"><input className="input" type="number" step="0.05" min={0.5} max={0.99} value={Number(s.interval_level)} onChange={(e) => set("interval_level", Number(e.target.value))} /></Field>
          </div>
        </Panel>
        <Panel title="Training defaults">
          <div className="space-y-3">
            <Field label="Default split strategy"><select className="input" value={String(s.default_split_strategy)} onChange={(e) => set("default_split_strategy", e.target.value)}><option value="auto">auto</option><option value="grouped">grouped</option><option value="random">random</option><option value="chronological">chronological</option></select></Field>
            <Field label="Default search mode"><select className="input" value={String(s.default_search_mode)} onChange={(e) => set("default_search_mode", e.target.value)}><option value="standard">standard</option><option value="quick">quick</option></select></Field>
            <Field label="Default models"><div className="grid grid-cols-2 gap-1">{q.data!.available_models.map((m) => <label key={m} className="flex items-center gap-2 text-sm text-[var(--muted)]"><input type="checkbox" checked={s.default_models.includes(m)} onChange={(e) => set("default_models", e.target.checked ? [...s.default_models, m] : s.default_models.filter((x) => x !== m))} />{MODEL_LABELS[m] || m}</label>)}</div></Field>
            <div className="grid grid-cols-3 gap-3">
              <Field label="Test fraction"><input className="input" type="number" step="0.05" value={Number(s.default_test_fraction)} onChange={(e) => set("default_test_fraction", Number(e.target.value))} /></Field>
              <Field label="CV folds"><input className="input" type="number" value={Number(s.default_n_folds)} onChange={(e) => set("default_n_folds", Number(e.target.value))} /></Field>
              <Field label="Random seed"><input className="input" type="number" value={Number(s.random_seed)} onChange={(e) => set("random_seed", Number(e.target.value))} /></Field>
            </div>
          </div>
        </Panel>
        <Panel title="Project information (used in reports)">
          <div className="space-y-3">
            {["title", "subtitle", "organisation", "author"].map((k) => <Field key={k} label={k}><input className="input" value={s.project?.[k] || ""} onChange={(e) => set("project", { ...s.project, [k]: e.target.value })} /></Field>)}
          </div>
        </Panel>
        <Panel title="Environment">
          <KeyValue items={Object.entries(q.data!.environment).map(([k, v]) => [k, String(v)])} />
          <Button variant="primary" className="mt-4 w-full" loading={save.isPending} onClick={() => save.mutate()}>Save settings</Button>
        </Panel>
      </div>
      {dict.data && (
        <div className="mt-4 space-y-4">
          <Panel title="Data dictionary" subtitle="Definition, unit, role and validity range of every standard feature">
            <DataTable columns={["name", "label", "unit", "dtype", "role", "valid_min", "valid_max", "description"]} rows={dict.data.features} maxHeight={420} />
          </Panel>
          <div className="grid gap-4 md:grid-cols-2">
            <Panel title="Equations"><ul className="space-y-2 text-sm">{dict.data.equations.map((e) => <li key={e.name}><span className="font-semibold text-white">{e.name}:</span> <code className="text-[var(--accent-2)]">{e.formula}</code> <span className="text-[var(--muted)]">[{e.unit}] — {e.note}</span></li>)}</ul></Panel>
            <Panel title="Material catalogue (bundled dataset)"><KeyValue items={Object.entries(dict.data.material_catalogue).map(([k, v]) => [k, v.description])} /></Panel>
          </div>
        </div>
      )}
    </>
  );
}
