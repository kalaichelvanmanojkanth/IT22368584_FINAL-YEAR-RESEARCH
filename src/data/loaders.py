"""Version-pinned Figshare download and exact Niteroi XLS adapter."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import ssl
from urllib.error import URLError
from urllib.request import urlopen

import certifi
import numpy as np
import pandas as pd

from src.config import DataConfig
from src.data.schema import CANONICAL_COLUMNS, DataError, DataUnavailableError

ZONE_IDS = {f"Zone {i}": f"zone_{i}" for i in range(1, 5)} | {"Untreated control": "control"}
PUBLICATION = "https://doi.org/10.1371/journal.pntd.0009556"


def checksum(path: Path, algorithm: str = "sha256") -> str:
    return hashlib.new(algorithm, path.read_bytes()).hexdigest()


def download_niteroi(root: Path, config: DataConfig) -> Path:
    """Download immutable version and verify checksum; never overwrite raw files."""
    directory = root / config.raw_dir
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / config.filename
    if target.exists():
        if checksum(target, "md5") != config.expected_md5:
            raise DataError(f"Existing source checksum mismatch: {target}; preserve and inspect it.")
        return target
    url = f"https://api.figshare.com/v2/articles/{config.article_id}/versions/{config.version}"
    context = ssl.create_default_context(cafile=certifi.where())
    try:
        with urlopen(url, context=context, timeout=45) as response:
            metadata = json.load(response)
        entry = next((f for f in metadata["files"] if f["name"] == config.filename), None)
        if entry is None:
            raise DataError("Pinned Figshare version no longer contains the configured filename")
        with urlopen(entry["download_url"], context=context, timeout=90) as response:
            content = response.read()
    except (URLError, TimeoutError, OSError) as exc:
        raise DataUnavailableError(
            f"Public data download failed: {exc}. Follow docs/MANUAL_DOWNLOAD.md. No substitute data created."
        ) from exc
    digest = hashlib.md5(content).hexdigest()
    if digest != config.expected_md5 or digest != entry["computed_md5"]:
        raise DataError("Downloaded source failed the pinned MD5 check")
    with target.open("xb") as stream:
        stream.write(content)
    for filename, value in {
        "source_metadata.json": metadata,
        "retrieval.json": {
            "retrieval_date": datetime.now(timezone.utc).isoformat(),
            "metadata_url": url, "filename": config.filename,
            "sha256": checksum(target), "md5": digest,
        },
    }.items():
        path = directory / filename
        if not path.exists():
            path.write_text(json.dumps(value, indent=2) + "\n")
    return target


def parse_source_month(values: pd.Series, *, nullable: bool = False) -> pd.Series:
    """Parse the documented YYYY_MM representation, not Excel serial guesses."""
    clean = values.replace("", np.nan)
    parsed = pd.to_datetime(clean, format="%Y_%m", errors="coerce")
    invalid = parsed.isna() & clean.notna()
    if invalid.any() or (not nullable and parsed.isna().any()):
        raise DataError("Unparseable or missing source date (expected YYYY_MM)")
    return parsed


def load_niteroi(root: Path, config: DataConfig) -> pd.DataFrame:
    """Map the actual Data sheet without filling absent measurements with zeros."""
    path = root / config.raw_dir / config.filename
    if not path.is_file():
        raise DataUnavailableError(f"Missing real source: {path}. Run download or see docs/MANUAL_DOWNLOAD.md")
    if checksum(path, "md5") != config.expected_md5:
        raise DataError("Source checksum differs from the configured version")
    raw = pd.read_excel(path, sheet_name="Data", engine="xlrd")
    expected = {"yearmonth", "zone", "dengue_cases", "pop", "group", "wmel_pc",
                "yearmonth_start", "yearmonth_monitor", "start_release", "epi_monitor"}
    if missing := expected - set(raw.columns):
        raise DataError(f"Unexpected source format; missing columns: {sorted(missing)}")
    if not raw["zone"].isin(ZONE_IDS).all() or not raw["group"].isin([0, 1]).all():
        raise DataError("Unknown zone/group identifiers in source")
    result = pd.DataFrame(index=raw.index, columns=CANONICAL_COLUMNS)
    result["study_id"], result["country"] = config.study_id, config.country
    result["location_id"] = raw["zone"].map(ZONE_IDS)
    result["location_name"] = raw["zone"]
    result["date"] = parse_source_month(raw["yearmonth"])
    result["year"], result["month"] = result["date"].dt.year, result["date"].dt.month
    for source, canonical in [("dengue_cases", "dengue_cases"), ("pop", "population")]:
        result[canonical] = pd.to_numeric(raw[source], errors="raise")
    result["dengue_incidence"] = result["dengue_cases"] / result["population"] * 100_000
    result["control_status"] = raw["group"].eq(0)
    if not result["control_status"].eq(result["location_id"].eq("control")).all():
        raise DataError("Source group and zone identifiers disagree")
    result["intervention_start"] = parse_source_month(raw["yearmonth_start"], nullable=True)
    result["monitoring_start"] = parse_source_month(raw["yearmonth_monitor"], nullable=True)
    for col in ["start_release", "epi_monitor"]:
        if not raw[col].isin([0, 1]).all():
            raise DataError(f"Invalid binary source field: {col}")
    result["intervention_status"] = raw["start_release"].astype(int)
    result["monitoring_status"] = raw["epi_monitor"].astype(int)
    # Source measures prevalence as percentage; canonical prevalence is a fraction.
    result["wolbachia_prevalence"] = pd.to_numeric(raw["wmel_pc"], errors="raise") / 100
    # No separate continuous intervention exposure or centroids are available.
    result["source_file"], result["source_version"] = config.filename, f"figshare.{config.article_id}.v{config.version}"
    retrieval_path = path.parent / "retrieval.json"
    if retrieval_path.exists():
        retrieval = json.loads(retrieval_path.read_text())
        if retrieval["sha256"] != checksum(path):
            raise DataError("Retrieval manifest SHA256 does not match raw data")
        result["retrieval_date"] = retrieval["retrieval_date"]
    result["missing_flag"] = result["dengue_cases"].isna()
    return result.sort_values(["study_id", "location_id", "date"]).reset_index(drop=True)


def study_metadata(config: DataConfig) -> dict:
    """Study-specific assumptions remain separate from observations."""
    return {
        "study_id": config.study_id, "study_design": "quasi-experimental",
        "intervention_type": "wMel Wolbachia mosquito deployments",
        "description": "Niteroi zone-level monthly surveillance and entomological data",
        "data_source": f"https://doi.org/10.6084/m9.figshare.{config.article_id}.v{config.version}",
        "publication": PUBLICATION, "licence": "CC BY 4.0",
        "geographic_resolution": "release zone / untreated control zone", "time_resolution": "monthly",
        "treatment_definition": "intervention_start: releases started throughout zone; monitoring_start: source post-intervention classification, four weeks later",
        "control_definition": "source group=0: Untreated control",
        "known_limitations": ["One permanent control zone", "Zone 1 includes 2015-16 pilot",
                              "Partial releases can precede zone-wide start", "Suspected and confirmed notifications",
                              "Population uses 2000 census in 2007-09, 2010 census subsequently",
                              "No verified unit centroids or independent exposure dose"],
    }
