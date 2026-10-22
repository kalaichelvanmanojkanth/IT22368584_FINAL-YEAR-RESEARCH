"""Observable data-quality checks and computed audit reporting."""
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.schema import CANONICAL_COLUMNS, REQUIRED_COLUMNS, DataError


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    statistics: dict = field(default_factory=dict)
    locations: list[dict] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return asdict(self) | {"valid": self.valid}

    def raise_for_errors(self) -> None:
        if self.errors:
            raise DataError("; ".join(self.errors))


def validate_panel(panel: pd.DataFrame, min_pre: int = 36, min_post: int = 6) -> ValidationReport:
    """Validate without sorting, imputing or changing caller observations."""
    report = ValidationReport(statistics={"rows": len(panel)})
    if missing := set(REQUIRED_COLUMNS) - set(panel):
        report.errors.append(f"Missing required columns: {sorted(missing)}")
        return report
    if panel.empty:
        report.errors.append("No observations")
        return report
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")
    if df["date"].isna().any():
        report.errors.append("Unparseable or missing dates")
        return report
    if not df["date"].dt.is_month_start.all():
        report.errors.append("Dates must be month starts")
    if df[["study_id", "location_id"]].isna().any().any():
        report.errors.append("Missing study/location identifiers")
    duplicated = df.duplicated(["study_id", "location_id", "date"]).sum()
    if duplicated:
        report.errors.append(f"Duplicate study/location/month observations: {duplicated}")
    for column in ["dengue_cases", "population"]:
        values = pd.to_numeric(df[column], errors="coerce")
        if (df[column].notna() & values.isna()).any() or np.isinf(values).any():
            report.errors.append(f"Non-numeric or infinite {column}")
        df[column] = values
    missing_cases = int(df["dengue_cases"].isna().sum())
    if missing_cases:
        report.warnings.append(f"Missing target outcomes: {missing_cases}; model fitting rejects incomplete windows")
    if (df["dengue_cases"].dropna() < 0).any():
        report.errors.append("Negative case counts")
    if (df["dengue_cases"].dropna() % 1 != 0).any():
        report.errors.append("Non-integer case counts")
    if (df["population"].dropna() <= 0).any():
        report.errors.append("Impossible population values (must be positive)")
    if df["population"].isna().any():
        report.warnings.append("Missing population: incidence/count-offset models cannot use these rows")
    if not df["missing_flag"].isin([True, False]).all() or not df["missing_flag"].eq(df["dengue_cases"].isna()).all():
        report.errors.append("missing_flag inconsistent with target missingness")
    for col in ["intervention_status", "control_status"]:
        if not df[col].isin([0, 1, False, True]).all():
            report.errors.append(f"Invalid treatment/control identifier: {col}")
    if "wolbachia_prevalence" in df:
        values = pd.to_numeric(df["wolbachia_prevalence"], errors="coerce")
        if ((values.dropna() < 0) | (values.dropna() > 1)).any():
            report.errors.append("Wolbachia prevalence outside [0,1]")
    for (study, location), group in df.groupby(["study_id", "location_id"], sort=True):
        if not group["date"].is_monotonic_increasing:
            report.errors.append(f"Nonchronological rows: {location}")
        expected = pd.date_range(group["date"].min(), group["date"].max(), freq="MS")
        gaps = len(expected.difference(group["date"]))
        if gaps:
            report.warnings.append(f"Absent calendar months: {location}: {gaps}")
        raw_start = group["intervention_start"]
        start = pd.to_datetime(raw_start, format="mixed", errors="coerce")
        if (raw_start.notna() & start.isna()).any() or not start.dropna().dt.is_month_start.all():
            report.errors.append(f"Invalid intervention date: {location}")
        if group["control_status"].nunique(dropna=False) != 1:
            report.errors.append(f"Inconsistent control status: {location}")
        is_control = bool(group["control_status"].eq(True).all())
        if is_control:
            if start.notna().any() or group["intervention_status"].ne(0).any():
                report.errors.append(f"Control has intervention label/date: {location}")
        elif start.isna().any() or start.nunique() != 1:
            report.errors.append(f"Treated location requires one consistent intervention date: {location}")
        else:
            expected_status = group["date"].ge(start.iloc[0]).astype(int)
            if not group["intervention_status"].eq(expected_status).all():
                report.errors.append(f"Inconsistent intervention labels: {location}")
        if "monitoring_start" in group and not is_control:
            monitor = pd.to_datetime(group["monitoring_start"], errors="coerce")
            if monitor.isna().any() or monitor.nunique() != 1 or (monitor < start).any():
                report.errors.append(f"Invalid monitoring date: {location}")
            elif "monitoring_status" in group and not group["monitoring_status"].eq(group["date"].ge(monitor.iloc[0]).astype(int)).all():
                report.errors.append(f"Inconsistent monitoring labels: {location}")
        n_pre = int(group["intervention_status"].eq(0).sum())
        n_post = int(group["intervention_status"].eq(1).sum())
        if not is_control and (n_pre < min_pre or n_post < min_post):
            report.errors.append(f"Insufficient pre/post history: {location}: {n_pre}/{n_post}")
        report.locations.append({"study_id": str(study), "location_id": str(location),
                                 "rows": len(group), "control": is_control,
                                 "pre_months_source_label": n_pre, "post_months_source_label": n_post,
                                 "missing_outcomes": int(group["dengue_cases"].isna().sum()),
                                 "absent_months": gaps,
                                 "intervention_start": None if start.isna().all() else str(start.dropna().iloc[0].date())})
    if not df["control_status"].eq(True).any() or not df["control_status"].eq(False).any():
        report.errors.append("Both treatment and control locations are required")
    report.statistics.update({
        "studies": int(df["study_id"].nunique()), "locations": int(df["location_id"].nunique()),
        "date_min": str(df["date"].min().date()), "date_max": str(df["date"].max().date()),
        "duplicate_rows": int(duplicated), "missing_outcomes": missing_cases,
        "missing_outcome_percent": 100 * missing_cases / len(df),
        "column_missing_percent": {col: float(100 * df[col].isna().mean()) for col in CANONICAL_COLUMNS if col in df},
    })
    return report


def write_audit(report: ValidationReport, path: Path, provenance: dict) -> None:
    """Write only measured statistics; source-label history is not clean history."""
    lines = ["# Dataset audit", "", f"Validation: {'PASS' if report.valid else 'FAIL'}", "",
             f"Source: {provenance['data_source']}", f"Publication: {provenance['publication']}",
             f"Licence: {provenance['licence']}", "", "## Computed statistics", ""]
    lines.extend(f"- {k}: {v}" for k, v in report.statistics.items() if k != "column_missing_percent")
    lines += ["", "## Per-location audit", "", "| Location | Rows | Control | Source pre | Source post | Missing outcomes | Gaps |",
              "|---|---:|---|---:|---:|---:|---:|"]
    lines += [f"| {g['location_id']} | {g['rows']} | {g['control']} | {g['pre_months_source_label']} | {g['post_months_source_label']} | {g['missing_outcomes']} | {g['absent_months']} |" for g in report.locations]
    lines += ["", "Source pre labels are not guaranteed unexposed: conservative training cutoffs are recorded in configs/niteroi.yaml and each run manifest.", "", "## Column missingness (%)", ""]
    lines += [f"- {k}: {v:.4f}" for k, v in report.statistics.get("column_missing_percent", {}).items()]
    lines += ["", "## Errors", ""] + (report.errors or ["None detected."])
    lines += ["", "## Warnings and study limitations", ""] + (report.warnings + provenance["known_limitations"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
