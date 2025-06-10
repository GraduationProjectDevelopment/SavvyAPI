from app.services.interfaces.time_series_service import ITimeSeriesService
from app.infrastructure.interfaces.time_series_repository import ITimeSeriesRepository
from app.models.time_series_dto import TimeSeriesPredictionResponse


class TimeSeriesService(ITimeSeriesService):
    def __init__(self, repo: ITimeSeriesRepository):
        self.repo = repo

    def get_next_month_prediction(self, user_id: str) -> TimeSeriesPredictionResponse:
        return self.repo.get_next_month_prediction(user_id)
