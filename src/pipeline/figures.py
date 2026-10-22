"""Phase A/B figures derived only from actual panel and model outputs."""
from pathlib import Path
import os

# Keep font/config cache writable inside the project in sandboxed environments.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parents[2] / ".cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SOURCE = "Source: Niteroi zone data, Figshare 13662203 v3; RACE-Dengue research"


def save_figure(fig, path: Path) -> None:
    fig.text(.01, .01, SOURCE, fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_figures(panel: pd.DataFrame, predictions: pd.DataFrame, metrics: pd.DataFrame,
                 directory: Path, clean_pre_end: dict[str, str]) -> list[str]:
    paths: list[str] = []

    def finish(fig, name):
        path = directory / name
        save_figure(fig, path)
        paths.append(str(path))

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (unit, group) in enumerate(panel.groupby("location_id")):
        ax.plot([group.date.min(), group.date.max()], [i, i], color="grey", linewidth=4,
                label="Observed calendar" if i == 0 else None)
        if unit in clean_pre_end:
            ax.scatter(pd.Timestamp(clean_pre_end[unit]), i, color="green", marker="|", s=220,
                       label="Clean training ends" if unit == next(iter(clean_pre_end)) else None)
        if group.intervention_start.notna().any():
            ax.scatter(group.intervention_start.dropna().iloc[0], i, color="orange", marker="s",
                       label="Zone-wide releases" if unit == next(iter(clean_pre_end)) else None)
            ax.scatter(group.monitoring_start.dropna().iloc[0], i, color="red", marker="x",
                       label="Monitoring starts" if unit == next(iter(clean_pre_end)) else None)
    ax.set_yticks(range(panel.location_id.nunique()), sorted(panel.location_id.unique()))
    ax.set(xlabel="Calendar month", ylabel="Geographic unit", title="Dataset timeline and conservative clean-history exclusions")
    ax.legend(fontsize=8)
    finish(fig, "01_dataset_timeline.png")

    fig, ax = plt.subplots(figsize=(12, 5))
    for unit, group in panel.groupby("location_id"):
        ax.plot(group.date, group.dengue_cases, label=unit, linewidth=1)
    ax.set(xlabel="Calendar month", ylabel="Notified dengue cases / month", title="Real surveillance counts by location", ylim=(0, None))
    ax.legend()
    finish(fig, "02_dengue_cases_by_location.png")

    for unit in sorted(predictions.location_id.unique()):
        rows = predictions[(predictions.location_id == unit) & (predictions.analysis == "intervention")]
        if rows.empty:
            continue
        fig, ax = plt.subplots(figsize=(11, 5))
        for name, group in rows.groupby("model_name"):
            ax.plot(pd.to_datetime(group.date), group.counterfactual_mean, label=name, marker=".")
        unique = rows.drop_duplicates("date").sort_values("date")
        ax.plot(pd.to_datetime(unique.date), unique.observed, color="black", marker="o", label="Observed")
        source_unit = panel[panel.location_id == unit]
        ax.axvline(source_unit.intervention_start.iloc[0], color="orange", linestyle=":", label="Zone-wide releases")
        ax.axvline(source_unit.monitoring_start.iloc[0], color="red", linestyle="--", label="Monitoring starts")
        ax.set(xlabel="Calendar month", ylabel="Notified dengue cases / month",
               title=f"{unit}: observed and counterfactual — uncalibrated baseline estimates", ylim=(0, None))
        ax.legend(fontsize=8)
        finish(fig, f"03_observed_counterfactual_{unit}.png")

    for analysis, column, filename, title in [
        ("intervention", "pre_rmse", "04_pre_fit_comparison.png", "Pre-period fit: in-sample RMSE (not validation accuracy)"),
        ("clean_pre_holdout", "holdout_mae", "05_baseline_holdout_error.png", "Untreated historical holdout: MAE by location"),
    ]:
        selected = metrics[(metrics.analysis == analysis) & (metrics.status == "completed")]
        if selected.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        selected.pivot(index="location_id", columns="model_name", values=column).plot.bar(ax=ax)
        ax.set(xlabel="Geographic unit", ylabel="Error in monthly notified dengue cases", title=title, ylim=(0, None))
        ax.tick_params(axis="x", rotation=0)
        ax.legend(title="Estimator")
        finish(fig, filename)
    return paths
