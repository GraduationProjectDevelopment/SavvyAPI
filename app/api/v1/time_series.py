from fastapi import APIRouter, HTTPException
from app.services.interfaces.time_series_service import ITimeSeriesService
from app.services.implementation.time_series_service import TimeSeriesService
from app.infrastructure.implementation.time_series_repository import (
    TimeSeriesRepository,
)
from app.models.time_series_dto import TimeSeriesPredictionResponse

router = APIRouter()

# Dependency injection
service: ITimeSeriesService = TimeSeriesService(TimeSeriesRepository())


@router.get("/predict/{user_id}", response_model=TimeSeriesPredictionResponse)
def predict_user_expense(user_id: str):
    try:
        return service.get_next_month_prediction(user_id)
    except HTTPException as e:
        raise e
