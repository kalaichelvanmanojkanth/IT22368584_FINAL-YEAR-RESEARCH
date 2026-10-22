"""Run actual-data baselines and one fixed historical holdout per treated zone."""
import logging
from pathlib import Path
from time import perf_counter

import pandas as pd

from src.config import ProjectConfig
from src.data.preprocess import preprocess
from src.data.schema import DataError
from src.evaluation.metrics import effect_summary, prediction_metrics
from src.models.base import ModelError
from src.models.cits import ControlledInterruptedTimeSeries
from src.models.synthetic_control import SyntheticControl
from src.utils.reproducibility import run_manifest, seed_everything, write_json

LOGGER = logging.getLogger(__name__)


def analysis_window(panel: pd.DataFrame, unit: str, config: ProjectConfig,
                    analysis: str) -> tuple[pd.DataFrame, pd.DatetimeIndex, pd.Timestamp, pd.DataFrame]:
    """Construct disjoint chronological windows before either estimator sees data."""
    if unit not in config.clean_pre_end:
        raise DataError(f"A documented clean pre-history cutoff is required for {unit}")
    eligible = panel[panel.location_id.isin([unit, *config.control_units])]
    treated = eligible[eligible.location_id.eq(unit)].sort_values("date")
    if treated.empty:
        raise DataError(f"Unknown treated location: {unit}")
    clean_end = pd.Timestamp(config.clean_pre_end[unit])
    start = treated.intervention_start.dropna().unique()
    if len(start) != 1 or clean_end >= start[0]:
        raise DataError("Clean pre-history must end before source intervention start")
    clean = eligible[eligible.date <= clean_end]
    if analysis == "clean_pre_holdout":
        available = pd.DatetimeIndex(clean[clean.location_id.eq(unit)].date)
        if len(available) < config.models.min_pre_months + config.models.holdout_months:
            raise DataError("Insufficient clean history for an untouched historical holdout")
        dates = available[-config.models.holdout_months:]
        cutoff = dates[0]
        pre = clean[clean.date < cutoff]
    elif analysis == "intervention":
        monitor = treated.monitoring_start.dropna().unique()
        if len(monitor) != 1:
            raise DataError("One source-defined monitoring date required")
        cutoff = pd.Timestamp(monitor[0])
        dates = pd.date_range(cutoff, periods=config.models.horizon_months, freq="MS")
        pre = clean
        if len(dates) < config.models.min_post_months:
            raise DataError("Requested horizon is below minimum post-period length")
    else:
        raise DataError(f"Unknown analysis mode: {analysis}")
    post = eligible[eligible.date.isin(dates)]
    for identifier in [unit, *config.control_units]:
        if not pd.DatetimeIndex(post[post.location_id.eq(identifier)].date.sort_values()).equals(dates):
            raise DataError(f"Incomplete requested horizon for {identifier}; no silent truncation")
    if pre.empty or pre.date.max() >= dates.min():
        raise DataError("Training and evaluation periods must not overlap")
    return pre, dates, cutoff, post


def run_baselines(root: Path, config: ProjectConfig, *, figures: bool = True) -> dict:
    seed_everything(config.seed)
    panel, audit = preprocess(root, config)
    manifest = run_manifest(root, config.model_dump(mode="json"), root / config.data.raw_dir / config.data.filename)
    run_id = manifest["run_id"]
    run_dir = root / "artifacts/runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "manifest.json", manifest | {"status": "started"})
    metric_rows, prediction_frames, pre_frames, evidence = [], [], [], []
    for unit in config.treated_units:
        for analysis in ["clean_pre_holdout", "intervention"]:
            for model_class in [ControlledInterruptedTimeSeries, SyntheticControl]:
                started = perf_counter()
                identity = {"run_id": run_id, "study_id": config.data.study_id, "location_id": unit,
                            "analysis": analysis, "model_name": model_class.model_name}
                try:
                    pre, dates, cutoff, post = analysis_window(panel, unit, config, analysis)
                    model = model_class().fit(pre, unit, config.control_units, config.models,
                                             intervention_date=cutoff,
                                             clean_pre_end=pd.Timestamp(config.clean_pre_end[unit]))
                    # Only source-defined control outcomes and treated population pass this boundary.
                    controls = post[post.location_id.isin(config.control_units)]
                    population = post[post.location_id.eq(unit)].set_index("date").loc[dates, "population"]
                    result = model.predict_counterfactual(dates, control_data=controls, treated_population=population)
                    # Access evaluation labels only AFTER prediction is complete.
                    observed = post[post.location_id.eq(unit)].set_index("date").loc[dates, "dengue_cases"].to_numpy(float)
                    frame = result.to_frame().assign(observed=observed, **{k: v for k, v in identity.items() if k != "model_name"})
                    prediction_frames.append(frame)
                    pre_frames.append(pd.DataFrame({"date": model.pre_dates, "observed": model.training_observed,
                                                    "fitted": model.fitted_pre_, **identity}))
                    row = identity | {"status": "completed", "n_pre": len(model.pre_dates), "horizon": len(dates),
                                      "train_start": str(model.pre_dates.min().date()),
                                      "train_end": str(model.pre_dates.max().date()),
                                      "evaluation_start": str(dates.min().date()), "evaluation_end": str(dates.max().date()),
                                      "pre_rmse": result.pre_fit_metrics["rmse"], "pre_mae": result.pre_fit_metrics["mae"],
                                      "holdout_mae": None, "holdout_rmse": None, "holdout_mase": None,
                                      "effect_absolute": None, "effect_percent": None, "estimated_cases_averted": None,
                                      "interval_lower": None, "interval_upper": None, "reliability_score": None,
                                      "decision": "UNCERTAIN_REVIEW_REQUIRED", "error": None,
                                      "warnings": " | ".join(result.warnings), "runtime_seconds": perf_counter() - started}
                    if analysis == "clean_pre_holdout":
                        row.update({f"holdout_{k}": v for k, v in prediction_metrics(observed, result.counterfactual_mean, model.training_observed).items()})
                    else:
                        row.update(effect_summary(observed, result.counterfactual_mean))
                    metric_rows.append(row)
                    detail = identity | model.summary() | {
                        "decision_reason": "Phase B baselines only; reliability model, calibration and abstention validation pending",
                        "calibrated_interval_available": False,
                    }
                    evidence.append(detail)
                    write_json(run_dir / f"{unit}_{analysis}_{model.model_name}.json", detail)
                    LOGGER.info("Completed %s / %s / %s", unit, analysis, model.model_name)
                except (DataError, ModelError) as exc:
                    row = identity | {"status": "failed", "error": str(exc), "runtime_seconds": perf_counter() - started}
                    metric_rows.append(row)
                    LOGGER.error("Failed %s: %s", identity, exc)
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    metrics_path = root / "artifacts/metrics" / f"{run_id}_baselines.csv"
    prediction_path = root / "artifacts/predictions" / f"{run_id}_counterfactuals.csv"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(prediction_path, index=False)
    if pre_frames:
        pd.concat(pre_frames, ignore_index=True).to_csv(root / "artifacts/predictions" / f"{run_id}_pre_fits.csv", index=False)
    figure_paths = []
    if figures and not predictions.empty:
        from src.pipeline.figures import make_figures
        figure_paths = make_figures(panel, predictions, metrics, root / "artifacts/figures" / run_id, config.clean_pre_end)
    manifest.update({"status": "completed" if metrics.status.eq("completed").all() else "completed_with_failures",
                     "completed_analyses": int(metrics.status.eq("completed").sum()),
                     "failed_analyses": int(metrics.status.eq("failed").sum()),
                     "metrics_csv": str(metrics_path.relative_to(root)),
                     "predictions_csv": str(prediction_path.relative_to(root)),
                     "figure_paths": [str(Path(p).relative_to(root)) for p in figure_paths],
                     "audit": audit.to_dict()})
    write_json(run_dir / "manifest.json", manifest)
    write_json(root / "artifacts/latest_run.json", manifest)
    metrics.to_csv(root / "reports/results_table.csv", index=False)
    from src.pipeline.reporting import generate_reports
    generate_reports(root, manifest, metrics)
    LOGGER.info("Saved %s: %d completed, %d failed", run_id, manifest["completed_analyses"], manifest["failed_analyses"])
    return manifest
