from pydantic import BaseModel
from uuid import UUID
from datetime import datetime
from typing import Optional


class CreateGoalRequest(BaseModel):
    user_id: str
    goal_name: str
    target_amount: float


class UpdateGoalRequest(BaseModel):
    goal_name: Optional[str] = None
    target_amount: Optional[float] = None
    amount_saved: Optional[float] = None


class GoalResponse(BaseModel):
    goal_id: UUID
    user_id: UUID
    goal_name: str
    target_amount: float
    amount_saved: float
    created_at: datetime
