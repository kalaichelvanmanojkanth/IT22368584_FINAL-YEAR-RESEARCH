from pathlib import Path

import pytest

from src.config import DataConfig
from src.data.loaders import checksum, load_niteroi
from src.data.schema import CANONICAL_COLUMNS, DataError, DataUnavailableError
from src.data.validate import validate_panel

ROOT = Path(__file__).resolve().parents[1]


def test_missing_real_data_stops(tmp_path):
    with pytest.raises(DataUnavailableError, match="Missing real source"):
        load_niteroi(tmp_path, DataConfig())


def test_corrupt_source_stops(tmp_path):
    config = DataConfig()
    path = tmp_path / config.raw_dir / config.filename
    path.parent.mkdir(parents=True)
    path.write_bytes(b"This is deliberately invalid test input, not data")
    with pytest.raises(DataError, match="checksum"):
        load_niteroi(tmp_path, config)


@pytest.mark.real_data
def test_real_source_mapping_if_available():
    config = DataConfig()
    source = ROOT / config.raw_dir / config.filename
    if not source.exists():
        pytest.skip("Original public XLS not downloaded; no synthetic substitute")
    before = checksum(source)
    df = load_niteroi(ROOT, config)
    assert set(CANONICAL_COLUMNS) <= set(df)
    assert validate_panel(df).valid
    assert df["latitude"].isna().all()
    assert df["intervention_exposure"].isna().all()
    assert df["wolbachia_prevalence"].isna().any()
    assert df["source_version"].eq("figshare.13662203.v3").all()
    assert checksum(source) == before
