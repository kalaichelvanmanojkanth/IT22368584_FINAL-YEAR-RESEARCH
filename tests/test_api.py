"""Exercise the real ASGI app without a network listener or deprecated test client."""
import asyncio
from pathlib import Path
import httpx
import pytest
from pydantic import ValidationError
from api.schemas import AnalysisRequest,AnalysisResponse


def call(app,method,url,**kwargs):
    async def request():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test.local') as client:
            return await client.request(method,url,**kwargs)
    return asyncio.run(request())


def test_api_schema():
    with pytest.raises(ValidationError):
        AnalysisRequest(study_id='niteroi_zone_v3',location_id='zone_2',intervention_date='wrong',horizon_months=0)
    result=AnalysisResponse(study_id='s',location_id='l',intervention_type='wMel',decision='INSUFFICIENT_EVIDENCE')
    assert result.effect_absolute is None
    assert result.reliability_score is None


def test_api_health_and_invalid_input(tmp_path):
    from api.main import create_app
    app=create_app(tmp_path)
    assert call(app,'GET','/health').json()['model_available'] is False
    assert call(app,'POST','/analyze',json={'study_id':'x'}).status_code==422


@pytest.mark.real_data
def test_api_real_analysis_abstains():
    from api.main import create_app
    root=Path(__file__).resolve().parents[1]
    if not (root/'artifacts/latest_reliability.json').exists(): pytest.skip('No trained real-data artifact')
    response=call(create_app(root),'POST','/analyze',json={'study_id':'niteroi_zone_v3','location_id':'zone_2',
                                                       'intervention_date':'2017-09-01','horizon_months':6})
    assert response.status_code==200
    result=response.json()
    assert result['decision']=='INSUFFICIENT_EVIDENCE'
    assert result['effect_absolute'] is None
    assert len(result['trajectories'])==6
    assert result['warnings']
