"""PDF / Excel report generation from real stored results."""
from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle  # noqa: E402
from sqlalchemy import select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.config import Settings  # noqa: E402
from app.core.features import STANDARD_FEATURES, TARGET_INFO  # noqa: E402
from app.models import Dataset, DatasetVersion, ModelVersion, Prediction, Report, TrainingRun, ValidationRecord  # noqa: E402
from app.services.analysis import target_vs_time  # noqa: E402
from app.services.app_settings import get_app_settings  # noqa: E402
from app.services.datasets import version_dataframe  # noqa: E402
from app.services.training import model_metrics_dict, model_summary  # noqa: E402

NOTE_MEASURED = "Measured = experimental observation from the dataset. Predicted = model output (not a measurement)."


class ReportError(ValueError):
    pass


# ----------------------------------------------------------------------------------------------
# Context collection
# ----------------------------------------------------------------------------------------------

def collect_context(db: Session, req: dict[str, Any]) -> dict[str, Any]:
    app = get_app_settings(db)
    version = db.get(DatasetVersion, req["dataset_version_id"]) if req.get("dataset_version_id") else (
        db.scalars(select(DatasetVersion).where(DatasetVersion.kind != "raw").order_by(DatasetVersion.id.desc())).first()
        or db.scalars(select(DatasetVersion).order_by(DatasetVersion.id.desc())).first())
    if version is None:
        raise ReportError("No dataset version exists yet — import or upload a dataset first")
    dataset = db.get(Dataset, version.dataset_id)
    if req.get("model_version_ids"):
        models = [m for m in (db.get(ModelVersion, i) for i in req["model_version_ids"]) if m is not None]
    else:
        models = []
        for t in TARGET_INFO:
            prod = db.scalars(select(ModelVersion).where(ModelVersion.target == t, ModelVersion.status == "production").order_by(ModelVersion.promoted_at.desc())).first()
            best = prod or db.scalars(select(ModelVersion).where(ModelVersion.target == t, ModelVersion.status != "archived", ModelVersion.is_best_of_run.is_(True)).order_by(ModelVersion.id.desc())).first()
            if best:
                models.append(best)
    comparison: list[ModelVersion] = []
    runs: list[TrainingRun] = []
    for m in models:
        run = db.get(TrainingRun, m.training_run_id)
        if run and run not in runs:
            runs.append(run)
            comparison.extend(run.models)
    if req.get("prediction_ids"):
        preds = [p for p in (db.get(Prediction, i) for i in req["prediction_ids"]) if p is not None]
    else:
        preds = db.scalars(select(Prediction).order_by(Prediction.id.desc()).limit(int(req.get("include_predictions", 20)))).all()
    validations = db.scalars(select(ValidationRecord).order_by(ValidationRecord.id.desc())).all() if req.get("include_validation", True) else []
    df = version_dataframe(db, version.id)
    return {"app": app, "dataset": dataset, "version": version, "models": models, "comparison": comparison, "runs": runs,
            "predictions": preds, "validations": validations, "df": df, "generated_at": datetime.now(timezone.utc)}


def _ranges(df: pd.DataFrame) -> list[list[str]]:
    rows = [["Parameter", "Unit", "Min", "Max", "Status"]]
    for f in ("normal_load_n", "sliding_speed_m_s", "temperature_c", "time_s", "sliding_distance_m", "material", "lubricant", "material_pair"):
        spec = STANDARD_FEATURES[f]
        if f not in df.columns or df[f].notna().sum() == 0:
            rows.append([spec.label, spec.unit, "–", "–", "not recorded"])
        elif spec.dtype == "number":
            s = df[f].astype(float)
            rows.append([spec.label, spec.unit, f"{s.min():.4g}", f"{s.max():.4g}", "fixed" if s.nunique() <= 1 or (s.max() - s.min()) <= 0.01 * abs(s.max()) else "varied"])
        else:
            lv = sorted(map(str, df[f].dropna().unique()))
            rows.append([spec.label, spec.unit, ", ".join(lv[:8]), "", f"{len(lv)} level(s)"])
    return rows


def _fmt(v: Any, nd: int = 4) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def limitations(ctx: dict[str, Any]) -> list[str]:
    version: DatasetVersion = ctx["version"]
    fs = version.feature_summary or {}
    out = []
    for t, info in TARGET_INFO.items():
        if not fs.get("targets", {}).get(t, {}).get("available"):
            out.append(f"{info['label']} is not available in this dataset version; no {info['short']} model can be trained and {info['short']} predictions are reported as unavailable.")
    for f, e in fs.get("inputs", {}).items():
        spec = STANDARD_FEATURES[f]
        if spec.role != "input":
            continue
        if not e.get("present"):
            out.append(f"{spec.label} was not recorded; its influence cannot be assessed.")
        elif not e.get("varies"):
            val = e.get("levels", [None])[0] if e.get("dtype") == "category" else e.get("min")
            out.append(f"{spec.label} was fixed ({val} {spec.unit}) in every experiment; predictions are only valid at this condition.")
    if fs.get("n_runs", 0) and fs.get("n_materials", 0):
        per = fs["n_runs"] / max(fs["n_materials"], 1)
        if per < 5:
            out.append(f"Only about {per:.0f} replicate runs per material: held-out-run performance estimates carry wide uncertainty.")
    out.append("Model outputs are statistical predictions conditioned on the training data; they are not experimental measurements.")
    out.append("SHAP feature contributions describe model behaviour and do not establish physical causation.")
    return out


# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------

def _fig_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def fig_cof_vs_time(df: pd.DataFrame, target: str) -> bytes | None:
    if target not in df.columns or df[target].notna().sum() == 0 or df["time_s"].isna().all():
        return None
    data = target_vs_time(df, target, "material", bins=60)
    if not data.get("available"):
        return None
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for g in data["groups"]:
        t = [p["t"] for p in g["points"]]
        m = [p["mean"] for p in g["points"]]
        s = [p["std"] or 0 for p in g["points"]]
        ax.plot(t, m, label=f"{g['group']} (n runs = {g['n_runs']})")
        ax.fill_between(t, np.array(m) - np.array(s), np.array(m) + np.array(s), alpha=0.15)
    ax.set_xlabel("Sliding time (s)")
    ax.set_ylabel(f"Measured {TARGET_INFO[target]['short']}")
    ax.set_title(f"Measured {TARGET_INFO[target]['label']} vs time (mean ± SD across replicate runs)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _fig_bytes(fig)


def fig_actual_vs_predicted(mv: ModelVersion) -> bytes | None:
    pts = (mv.cv_results or {}).get("test_predictions", [])
    if not pts:
        return None
    a = np.array([p["actual"] for p in pts])
    p_ = np.array([p["predicted"] for p in pts])
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    axes[0].scatter(a, p_, s=6, alpha=0.5)
    lo, hi = float(min(a.min(), p_.min())), float(max(a.max(), p_.max()))
    axes[0].plot([lo, hi], [lo, hi], "k--", lw=1)
    axes[0].set_xlabel("Measured (test set)")
    axes[0].set_ylabel("Predicted")
    axes[0].set_title(f"{mv.algorithm}: actual vs predicted")
    axes[1].hist(a - p_, bins=30, alpha=0.8)
    axes[1].set_xlabel("Residual (measured − predicted)")
    axes[1].set_title("Residual distribution (test set)")
    for ax in axes:
        ax.grid(alpha=0.3)
    return _fig_bytes(fig)


def fig_importance(mv: ModelVersion) -> bytes | None:
    imp = (mv.global_importance or {}).get("importance", [])
    if not imp:
        return None
    fig, ax = plt.subplots(figsize=(6, 2.6))
    ax.barh([i["label"] for i in imp][::-1], [i["mean_abs_shap"] for i in imp][::-1])
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title(f"Global feature importance — {mv.target} / {mv.algorithm}")
    ax.grid(alpha=0.3, axis="x")
    return _fig_bytes(fig)


# ----------------------------------------------------------------------------------------------
# PDF
# ----------------------------------------------------------------------------------------------

def _table(rows: list[list[Any]], col_widths: list[float] | None = None, font: int = 7) -> Table:
    t = Table([[Paragraph(str(c), ParagraphStyle("c", fontSize=font, leading=font + 2)) for c in r] for r in rows], colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b1d33")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9fb3c8")), ("FONTSIZE", (0, 0), (-1, -1), font),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")]), ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def build_pdf(ctx: dict[str, Any], path: Path, req: dict[str, Any]) -> None:
    styles = getSampleStyleSheet()
    h1, h2, body = styles["Title"], styles["Heading2"], styles["BodyText"]
    small = ParagraphStyle("small", parent=body, fontSize=8, leading=10)
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    el: list[Any] = []
    app, dataset, version, df = ctx["app"], ctx["dataset"], ctx["version"], ctx["df"]
    proj = app.get("project", {})
    title = req.get("title") or "AI TriboPredict — Model and Prediction Report"
    el += [Paragraph(title, h1), Paragraph(proj.get("title", ""), body), Paragraph(proj.get("subtitle", ""), body),
           Paragraph(f"{proj.get('organisation', '')} {('— ' + proj['author']) if proj.get('author') else ''}", body),
           Paragraph(f"Generated {ctx['generated_at'].strftime('%Y-%m-%d %H:%M UTC')} · Dataset version {version.version_number} ({version.kind}) of "
                     f"'{dataset.name}' (dataset id {dataset.id}, version id {version.id})", small),
           Paragraph(NOTE_MEASURED, small), Spacer(1, 8)]

    el += [Paragraph("1. Dataset description", h2),
           Paragraph(dataset.description or "", small),
           _table([["Rows", "Usable rows", "Runs", "Materials", "Quality score", "COF available", "Wear rate available"],
                   [version.record_count, version.usable_record_count, (version.feature_summary or {}).get("n_runs"), (version.feature_summary or {}).get("n_materials"),
                    (version.quality_report or {}).get("score"), (version.feature_summary or {}).get("targets", {}).get("cof", {}).get("available"),
                    (version.feature_summary or {}).get("targets", {}).get("wear_rate", {}).get("available")]]), Spacer(1, 6)]

    el += [Paragraph("2. Data-cleaning summary", h2),
           _table([["Step", "Description", "Rows"]] + [[e["step"], e["description"], e["rows_affected"]] for e in (version.cleaning_log or [])] or [["(raw version — no cleaning applied)", "", ""]],
                  col_widths=[3 * cm, 12.5 * cm, 1.8 * cm]), Spacer(1, 6)]

    el += [Paragraph("3. Experimental parameter ranges", h2), _table(_ranges(df)), Spacer(1, 6)]
    img = fig_cof_vs_time(df, "cof")
    if img:
        el += [Image(io.BytesIO(img), width=16 * cm, height=8.2 * cm), Spacer(1, 6)]

    el += [Paragraph("4. Machine-learning methodology", h2)]
    if ctx["runs"]:
        for run in ctx["runs"]:
            s = run.summary or {}
            sp = s.get("split", {})
            el += [Paragraph(f"Training run {run.id} — target {run.target}: features {', '.join(s.get('features', []))}. Split: {sp.get('description', '')} "
                             f"Search mode: {run.config.get('search_mode')} (grid search, scoring RMSE). Seed {run.config.get('seed')}. "
                             f"Model selection by out-of-fold cross-validation RMSE; the held-out test set is evaluated once.", small)]
            for n in s.get("notes", []):
                el.append(Paragraph("• " + n, small))
    else:
        el.append(Paragraph("No model has been trained for this dataset version yet.", small))
    el.append(Spacer(1, 6))

    el += [Paragraph("5. Model comparison", h2)]
    if ctx["comparison"]:
        rows = [["Target", "Model", "Status", "CV RMSE", "Val R²", "Val MAE", "Test R²", "Test RMSE", "Test MAE", "Test MAPE %", "Train s"]]
        for m in sorted(ctx["comparison"], key=lambda m: (m.target, (m.cv_results or {}).get("cv_rmse_mean") or 9)):
            mm = model_metrics_dict(m)
            rows.append([m.target, m.algorithm + (" ★" if m.is_best_of_run else ""), m.status, _fmt((m.cv_results or {}).get("cv_rmse_mean")),
                         _fmt(mm.get("validation", {}).get("r2")), _fmt(mm.get("validation", {}).get("mae")), _fmt(mm.get("test", {}).get("r2")),
                         _fmt(mm.get("test", {}).get("rmse")), _fmt(mm.get("test", {}).get("mae")), _fmt(mm.get("test", {}).get("mape"), 1), _fmt(m.training_duration_s, 1)])
        el += [_table(rows), Paragraph("★ = best validated model of its training run (lowest cross-validation RMSE).", small)]
    else:
        el.append(Paragraph("Model not trained.", small))
    el.append(Spacer(1, 6))

    el += [Paragraph("6. Evaluation of the selected model(s)", h2)]
    for m in ctx["models"]:
        mm = model_metrics_dict(m)
        el += [Paragraph(f"{TARGET_INFO[m.target]['label']} — {m.version_label} ({m.algorithm}, {m.status}); hyper-parameters: {(m.hyperparameters or {}).get('best_params')}", small),
               _table([["Split", "n", "R²", "MAE", "RMSE", "MAPE %"]] + [[s, mm[s].get("n"), _fmt(mm[s].get("r2")), _fmt(mm[s].get("mae")), _fmt(mm[s].get("rmse")), _fmt(mm[s].get("mape"), 1)] for s in ("train", "validation", "test") if s in mm]),
               Paragraph(f"Prediction interval: {(m.interval or {}).get('method', 'n/a')}; half-width {_fmt((m.interval or {}).get('half_width'))}; empirical test coverage {_fmt((m.interval or {}).get('test_coverage'), 2)}.", small)]
        img = fig_actual_vs_predicted(m)
        if img:
            el.append(Image(io.BytesIO(img), width=16 * cm, height=6.8 * cm))
        if req.get("include_explainability", True):
            img = fig_importance(m)
            if img:
                el.append(Image(io.BytesIO(img), width=13 * cm, height=5.6 * cm))
        el.append(Spacer(1, 6))
    if not ctx["models"]:
        el.append(Paragraph("Model not trained.", small))

    el += [PageBreak(), Paragraph("7. Predictions (model outputs, not measurements)", h2)]
    if ctx["predictions"]:
        rows = [["ID", "Inputs", "Predicted COF", "COF interval", "Predicted wear rate", "In domain", "Model", "Created"]]
        for p in ctx["predictions"]:
            iv = p.cof_interval or {}
            rows.append([p.id, ", ".join(f"{k}={v}" for k, v in (p.inputs or {}).items()), _fmt(p.predicted_cof),
                         f"{_fmt(iv.get('low'))} – {_fmt(iv.get('high'))}" if iv.get("low") is not None else "n/a",
                         _fmt(p.predicted_wear_rate) if p.predicted_wear_rate is not None else "unavailable", p.in_domain, p.cof_model_version_id,
                         p.created_at.strftime("%Y-%m-%d %H:%M") if p.created_at else ""])
        el.append(_table(rows, col_widths=[1 * cm, 5.2 * cm, 1.8 * cm, 2.6 * cm, 2.2 * cm, 1.4 * cm, 1.2 * cm, 2 * cm]))
    else:
        el.append(Paragraph("No predictions recorded.", small))

    el += [Spacer(1, 6), Paragraph("8. Experimental validation results", h2)]
    if ctx["validations"]:
        rows = [["Experiment", "Inputs", "Measured COF", "Predicted COF", "Abs error", "% error", "Status"]]
        for v in ctx["validations"]:
            rows.append([v.experiment_id, ", ".join(f"{k}={val}" for k, val in (v.inputs or {}).items()), _fmt(v.measured_cof), _fmt(v.predicted_cof), _fmt(v.cof_abs_error), _fmt(v.cof_pct_error, 1), v.status])
        errs = [v.cof_abs_error for v in ctx["validations"] if v.cof_abs_error is not None]
        el += [_table(rows), Paragraph(f"COF validation MAE over {len(errs)} record(s): {_fmt(float(np.mean(errs)) if errs else None)}", small)]
    else:
        el.append(Paragraph("No experimental validation records yet.", small))

    el += [Spacer(1, 6), Paragraph("9. Limitations and conclusions", h2)] + [Paragraph("• " + l, small) for l in limitations(ctx)]
    doc.build(el)


# ----------------------------------------------------------------------------------------------
# Excel
# ----------------------------------------------------------------------------------------------

def build_xlsx(ctx: dict[str, Any], path: Path, req: dict[str, Any]) -> None:
    app, dataset, version, df = ctx["app"], ctx["dataset"], ctx["version"], ctx["df"]
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        proj = app.get("project", {})
        pd.DataFrame([
            {"item": "Report title", "value": req.get("title") or "AI TriboPredict report"}, {"item": "Project", "value": proj.get("title")},
            {"item": "Subtitle", "value": proj.get("subtitle")}, {"item": "Organisation", "value": proj.get("organisation")},
            {"item": "Generated (UTC)", "value": ctx["generated_at"].isoformat()}, {"item": "Dataset", "value": f"{dataset.name} (id {dataset.id})"},
            {"item": "Dataset version", "value": f"v{version.version_number} ({version.kind}, id {version.id})"}, {"item": "Note", "value": NOTE_MEASURED},
        ]).to_excel(xw, sheet_name="Project", index=False)
        fs = version.feature_summary or {}
        pd.DataFrame([{"rows": version.record_count, "usable_rows": version.usable_record_count, "runs": fs.get("n_runs"), "materials": fs.get("n_materials"),
                       "quality_score": (version.quality_report or {}).get("score"), "cof_available": fs.get("targets", {}).get("cof", {}).get("available"),
                       "wear_rate_available": fs.get("targets", {}).get("wear_rate", {}).get("available"), "description": dataset.description}]).to_excel(xw, sheet_name="Dataset", index=False)
        pd.DataFrame(version.cleaning_log or []).to_excel(xw, sheet_name="Cleaning", index=False)
        r = _ranges(df)
        pd.DataFrame(r[1:], columns=r[0]).to_excel(xw, sheet_name="Ranges", index=False)
        meth = []
        for run in ctx["runs"]:
            s = run.summary or {}
            meth.append({"training_run": run.id, "target": run.target, "features": ", ".join(s.get("features", [])), "split": s.get("split", {}).get("description"),
                         "search_mode": run.config.get("search_mode"), "seed": run.config.get("seed"), "notes": " | ".join(s.get("notes", []))})
        pd.DataFrame(meth).to_excel(xw, sheet_name="Methodology", index=False)
        comp = []
        for m in ctx["comparison"]:
            mm = model_metrics_dict(m)
            comp.append({"target": m.target, "model_version_id": m.id, "algorithm": m.algorithm, "status": m.status, "best_of_run": m.is_best_of_run,
                         "cv_rmse_mean": (m.cv_results or {}).get("cv_rmse_mean"), "cv_rmse_std": (m.cv_results or {}).get("cv_rmse_std"),
                         **{f"{s}_{k}": mm.get(s, {}).get(k) for s in ("train", "validation", "test") for k in ("r2", "mae", "rmse", "mape", "n")},
                         "training_duration_s": m.training_duration_s, "hyperparameters": str((m.hyperparameters or {}).get("best_params"))})
        pd.DataFrame(comp).to_excel(xw, sheet_name="ModelComparison", index=False)
        preds = [{"prediction_id": p.id, **{f"input_{k}": v for k, v in (p.inputs or {}).items()}, "predicted_cof": p.predicted_cof,
                  "cof_low": (p.cof_interval or {}).get("low"), "cof_high": (p.cof_interval or {}).get("high"),
                  "predicted_wear_rate": p.predicted_wear_rate if p.predicted_wear_rate is not None else "unavailable", "in_domain": p.in_domain,
                  "warnings": " | ".join(p.warnings or []), "cof_model_version_id": p.cof_model_version_id, "created_at": p.created_at.isoformat() if p.created_at else None}
                 for p in ctx["predictions"]]
        pd.DataFrame(preds).to_excel(xw, sheet_name="Predictions (model)", index=False)
        vals = [{"validation_id": v.id, "experiment_id": v.experiment_id, **{f"input_{k}": val for k, val in (v.inputs or {}).items()}, "measured_cof": v.measured_cof,
                 "predicted_cof": v.predicted_cof, "cof_abs_error": v.cof_abs_error, "cof_pct_error": v.cof_pct_error, "measured_wear_rate": v.measured_wear_rate,
                 "predicted_wear_rate": v.predicted_wear_rate, "status": v.status, "model_version_id": v.model_version_id} for v in ctx["validations"]]
        pd.DataFrame(vals).to_excel(xw, sheet_name="Validation (measured)", index=False)
        imp = []
        for m in ctx["models"]:
            for i in (m.global_importance or {}).get("importance", []):
                imp.append({"target": m.target, "model_version_id": m.id, "algorithm": m.algorithm, "feature": i["feature"], "mean_abs_shap": i["mean_abs_shap"], "share": i["share"],
                            "method": (m.global_importance or {}).get("method")})
        pd.DataFrame(imp).to_excel(xw, sheet_name="Explainability", index=False)
        pd.DataFrame({"limitation": limitations(ctx)}).to_excel(xw, sheet_name="Limitations", index=False)


def generate_report(db: Session, settings: Settings, req: dict[str, Any]) -> Report:
    ctx = collect_context(db, req)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = ctx["generated_at"].strftime("%Y%m%d_%H%M%S")
    fmt = req.get("format", "pdf")
    path = settings.reports_dir / f"tribopredict_report_{stamp}.{fmt}"
    if fmt == "pdf":
        build_pdf(ctx, path, req)
    else:
        build_xlsx(ctx, path, req)
    rep = Report(kind=fmt, title=req.get("title") or f"AI TriboPredict report {stamp}", file_path=str(path),
                 params={**req, "model_version_ids": [m.id for m in ctx["models"]], "prediction_ids": [p.id for p in ctx["predictions"]]},
                 dataset_version_id=ctx["version"].id)
    db.add(rep)
    db.commit()
    db.refresh(rep)
    return rep


def report_view(r: Report) -> dict[str, Any]:
    return {"id": r.id, "kind": r.kind, "title": r.title, "params": r.params, "dataset_version_id": r.dataset_version_id,
            "created_at": r.created_at.isoformat() if r.created_at else None, "file_exists": Path(r.file_path).exists(), "filename": Path(r.file_path).name}
