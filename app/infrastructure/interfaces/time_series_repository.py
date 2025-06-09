from abc import ABC, abstractmethod
from app.models.time_series_dto import TimeSeriesPredictionResponse


class ITimeSeriesRepository(ABC):

    @abstractmethod
    def get_next_month_prediction(self, user_id: str) -> TimeSeriesPredictionResponse:
        pass
