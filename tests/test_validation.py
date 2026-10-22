import numpy as np
import pandas as pd
import pytest
from src.data.validate import validate_panel


def test_duplicate_detection(panel):
    report = validate_panel(pd.concat([panel, panel.iloc[[0]]]))
    assert any("Duplicate" in x for x in report.errors)


def test_negative_cases_rejected(panel):
    panel.loc[0, "dengue_cases"] = -1
    assert any("Negative" in x for x in validate_panel(panel).errors)


@pytest.mark.parametrize("value", [0, -1, np.inf])
def test_impossible_population(panel, value):
    panel.loc[0, "population"] = value
    assert not validate_panel(panel).valid


def test_missing_is_not_zero(panel):
    panel.loc[0, "dengue_cases"] = np.nan
    panel.loc[0, "missing_flag"] = True
    report = validate_panel(panel)
    assert report.statistics["missing_outcomes"] == 1
    assert pd.isna(panel.loc[0, "dengue_cases"])


def test_nonchronological(panel):
    assert not validate_panel(panel.iloc[::-1]).valid


def test_bad_date(panel):
    panel["date"] = panel["date"].astype(object)
    panel.loc[0, "date"] = "bad-date"
    assert not validate_panel(panel).valid


def test_inconsistent_labels(panel):
    panel.loc[0, "intervention_status"] = 1
    assert not validate_panel(panel).valid


def test_insufficient_history(panel):
    assert not validate_panel(panel, min_pre=200).valid


def test_calendar_gap_reported(panel):
    report = validate_panel(panel.drop(index=20))
    assert any("Absent calendar" in w for w in report.warnings)
