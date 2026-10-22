"""Canonical columns; unavailable source measurements remain null."""
CANONICAL_COLUMNS = [
    "study_id", "country", "location_id", "location_name", "date", "year", "month",
    "dengue_cases", "population", "dengue_incidence", "intervention_status",
    "intervention_start", "intervention_exposure", "wolbachia_prevalence",
    "control_status", "latitude", "longitude", "source_file", "source_version",
    "retrieval_date", "missing_flag",
]
REQUIRED_COLUMNS = [
    "study_id", "location_id", "date", "dengue_cases", "population",
    "intervention_status", "intervention_start", "control_status", "missing_flag",
]


class DataError(ValueError):
    """Missing, inconsistent or scientifically unsuitable input data."""


class DataUnavailableError(DataError):
    """The actual source data are unavailable; never fall back to generated rows."""
