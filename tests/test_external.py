from pathlib import Path
import pytest
from src.data.external import SOURCES,load_external
from src.data.schema import CANONICAL_COLUMNS,DataUnavailableError

ROOT=Path(__file__).resolve().parents[1]


def test_external_missing_stops(tmp_path):
    with pytest.raises(DataUnavailableError): load_external(tmp_path,'yogyakarta')


@pytest.mark.real_data
@pytest.mark.parametrize('site',['yogyakarta','colombia'])
def test_external_source_mapping_if_available(site):
    if not (ROOT/'data/raw'/site/SOURCES[site]['file']).exists(): pytest.skip('Original external source not downloaded')
    frame,metadata=load_external(ROOT,site)
    assert set(CANONICAL_COLUMNS)<=set(frame)
    assert frame.control_status.isna().all()
    assert frame.study_id.nunique()==1
    assert not frame.duplicated(['study_id','location_id','date']).any()
    assert 'avg_wmelpc' not in frame
    assert 'wmel_ipol' not in frame
    assert frame.intervention_exposure.isna().all()
    if site=='yogyakarta': assert frame.df_cases_separate.isna().any()
    assert metadata['licence']=='CC BY 4.0'
