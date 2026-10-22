"""All generated observations here are SYNTHETIC UNIT-TEST FIXTURES only."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def panel():
    dates = pd.date_range("2000-01-01", periods=84, freq="MS")
    rows = []
    for unit, control, factor in [("treated", False, 1.0), ("donor_a", True, .8), ("donor_b", True, 1.2)]:
        for i, date in enumerate(dates):
            cases = round(factor * (30 + i * .2 + 12 * np.sin(i * 2 * np.pi / 12)))
            rows.append(dict(study_id="SYNTHETIC_UNIT_TEST", location_id=unit, date=date,
                             dengue_cases=cases, population=100_000., dengue_incidence=float(cases),
                             intervention_start=pd.NaT if control else dates[72],
                             monitoring_start=pd.NaT if control else dates[72],
                             intervention_status=int(not control and i >= 72),
                             control_status=control, missing_flag=False))
    return pd.DataFrame(rows).sort_values(["study_id", "location_id", "date"]).reset_index(drop=True)
