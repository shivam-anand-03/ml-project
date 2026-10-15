"use client";

import { useQuery } from "@tanstack/react-query";
import { apiGet } from "@/lib/api";

export interface DatasetSummary {
  id: number; name: string; description: string | null; source_type: string; created_at: string | null; n_files: number; n_versions: number;
  latest_version_id: number | null; latest_version_number: number | null; latest_kind: string | null; record_count: number; usable_record_count: number;
  targets: Record<string, { available: boolean; n: number }>;
}

export interface VersionSummary {
  id: number; dataset_id: number; version_number: number; kind: string; parent_version_id: number | null; notes: string | null; record_count: number;
  usable_record_count: number; created_at: string | null; feature_summary: FeatureSummary; quality_score: number | null; cleaning_steps: number;
}

export interface FeatureSummary {
  targets: Record<string, { available: boolean; n: number; min: number | null; max: number | null; mean: number | null }>;
  inputs: Record<string, { present: boolean; n: number; dtype: string; unit: string; role: string; min?: number | null; max?: number | null; levels?: string[]; n_unique?: number; varies: boolean }>;
  n_rows: number; n_usable: number; n_runs: number; n_materials: number;
}

export interface ModelSummary {
  id: number; training_run_id: number; dataset_version_id: number; target: string; target_unit: string | null; algorithm: string; label: string; version_label: string;
  status: string; is_best_of_run: boolean; features: string[]; hyperparameters: Record<string, unknown>;
  metrics: Record<string, { r2?: number | null; mae?: number | null; rmse?: number | null; mape?: number | null; n?: number; mape_coverage?: number | null }>;
  training_duration_s: number; n_train: number; n_test: number; trained_at: string | null; promoted_at: string | null;
  interval: { method?: string; level?: number; half_width?: number; test_coverage?: number | null };
  split: { strategy?: string; description?: string; warnings?: string[]; test_groups?: string[]; cv_splits?: number; small_dataset?: boolean };
  cv_rmse_mean: number | null; cv_rmse_std: number | null;
}

export interface TrainingRunView {
  id: number; dataset_version_id: number; target: string; status: string; config: Record<string, unknown>; summary: Record<string, unknown> & {
    leaderboard?: { algorithm: string; label: string; error: string | null; duration_s: number; validation_rmse?: number; validation_r2?: number; test_rmse?: number; test_r2?: number }[];
    best_model_version_id?: number; best_algorithm?: string; features?: string[]; notes?: string[]; models_done?: string[]; n_models?: number;
    split?: { strategy: string; description: string; warnings: string[]; test_groups: string[] };
  };
  error: string | null; log: { t: string; msg: string }[]; created_at: string | null; started_at: string | null; finished_at: string | null; is_active: boolean;
  models?: ModelSummary[];
}

export function useDatasets() {
  return useQuery({ queryKey: ["datasets"], queryFn: () => apiGet<DatasetSummary[]>("/api/datasets") });
}

export function useDataset(id: number | null) {
  return useQuery({ queryKey: ["dataset", id], queryFn: () => apiGet<DatasetSummary & { files: FileView[]; versions: VersionSummary[] }>(`/api/datasets/${id}`), enabled: id !== null });
}

export interface FileView {
  id: number; filename: string; file_type: string; size_bytes: number; uploaded_at: string | null;
  inspection: { file_type: string; errors: string[]; sheets: SheetInspection[] };
}

export interface SheetInspection {
  sheet_name: string; n_rows: number; n_cols: number; is_empty: boolean; error?: string; duplicate_rows: number;
  columns: { name: string; dtype: string; non_null: number; null_count: number; n_unique: number; sample_values: unknown[]; min?: number; max?: number; mean?: number }[];
  preview_columns?: string[]; preview?: unknown[][]; parsed_run: Record<string, unknown>; suggested_mapping: Record<string, { feature: string | null; confidence: number }>;
}

export function useModels(params?: Record<string, string | number | undefined>) {
  return useQuery({ queryKey: ["models", params], queryFn: () => apiGet<ModelSummary[]>("/api/models", params) });
}

export function useProductionModels() {
  return useQuery({ queryKey: ["models", "production"], queryFn: () => apiGet<Record<string, { production: ModelSummary | null; latest_best: ModelSummary | null; active: ModelSummary | null }>>("/api/models/production") });
}

export function useRuns() {
  return useQuery({ queryKey: ["runs"], queryFn: () => apiGet<TrainingRunView[]>("/api/models/runs"), refetchInterval: (q) => (q.state.data?.some((r) => r.status === "running" || r.status === "queued") ? 2000 : false) });
}

export interface PredictSchema {
  targets: Record<string, { available: boolean; model_id?: number; algorithm?: string; status?: string; version_label?: string; unit?: string; reason?: string; dataset_version_id?: number }>;
  inputs: { name: string; dtype: string; unit: string; label: string; min?: number; max?: number; p05?: number; p95?: number; mean?: number; levels?: Record<string, number>; targets: string[] }[];
  fixed_conditions: Record<string, { name: string; value: number | string | null; unit: string; label: string; min?: number; max?: number; note?: string; levels?: string[] }>;
  not_recorded: { name: string; label: string; unit: string }[];
  app_settings: Record<string, unknown>;
}

export function usePredictSchema() {
  return useQuery({ queryKey: ["predict-schema"], queryFn: () => apiGet<PredictSchema>("/api/predict/schema") });
}

/** Pick the dataset version used by default across pages: latest non-raw version of the most recent dataset. */
export function useDefaultVersion() {
  const ds = useDatasets();
  const latest = ds.data?.[0];
  return { ...ds, versionId: latest?.latest_version_id ?? null, dataset: latest };
}
