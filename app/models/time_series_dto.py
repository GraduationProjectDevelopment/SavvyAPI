from typing import Dict, List
from pydantic import BaseModel
from datetime import date


class ExpenseBreakdown(BaseModel):
    Education: float
    Entertainment: float
    Fashion: float
    Food: float
    Lifestyle: float
    Transportation: float
    Health: float


class MonthlyExpense(BaseModel):
    month: str  # e.g., "2025-03"
    total_expense: float
    breakdown: ExpenseBreakdown


class TimeSeriesPredictionResponse(BaseModel):
    user_id: str
    date: str  # next prediction date "2025-07-01"
    month: str  # month label "2025-07"
    predicted_total_expense: float
    history: List[MonthlyExpense]
