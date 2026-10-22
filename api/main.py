"""Optional local API. No other component is required for training or inference."""
from pathlib import Path
from fastapi import FastAPI,HTTPException
from api.schemas import AnalysisRequest,AnalysisResponse
from src.data.schema import DataError
from src.pipeline.predict import analyze


def create_app(root: Path | None=None) -> FastAPI:
    repository=root or Path(__file__).resolve().parents[1]
    application=FastAPI(title='RACE-Dengue — Component 03',version='0.2.0',
                        description='Research evidence, not clinical diagnosis. Bind locally unless deployment security is configured.')

    @application.get('/health')
    def health():
        return {'status':'ok','component':'C03','model_available':(repository/'artifacts/latest_reliability.json').exists()}

    @application.post('/analyze',response_model=AnalysisResponse)
    def intervention(request: AnalysisRequest):
        try:
            return analyze(repository,request.study_id,request.location_id,
                           request.intervention_date.isoformat(),request.horizon_months)
        except DataError as exc: raise HTTPException(status_code=422,detail=str(exc)) from exc
    return application

app=create_app()
