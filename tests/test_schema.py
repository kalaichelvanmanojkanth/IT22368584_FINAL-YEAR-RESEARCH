import pandas as pd
import pytest
from pydantic import ValidationError
from src.config import ProjectConfig
from src.data.loaders import parse_source_month
from src.data.schema import DataError
from src.data.validate import validate_panel


def test_required_schema(panel):
    assert validate_panel(panel).valid
    assert not validate_panel(panel.drop(columns="study_id")).valid


def test_date_parsing():
    assert parse_source_month(pd.Series(["2007_01"])).iloc[0] == pd.Timestamp("2007-01-01")
    with pytest.raises(DataError):
        parse_source_month(pd.Series(["2007_99"]))
    assert parse_source_month(pd.Series([None]), nullable=True).isna().all()


def test_typed_config_rejects_unknown_and_overlap():
    with pytest.raises(ValidationError):
        ProjectConfig.model_validate({"unknown": True})
    with pytest.raises(ValidationError):
        ProjectConfig(treated_units=["x"], control_units=["x"])
