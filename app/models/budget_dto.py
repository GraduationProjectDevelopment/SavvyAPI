from pydantic import BaseModel
from uuid import UUID
from typing import Optional
from datetime import datetime


class CreateBudgetRequest(BaseModel):
    user_id: str
    category_id: str
    allocated_amount: float


class UpdateBudgetRequest(BaseModel):
    budget_id: str
    allocated_amount: float


class BudgetResponse(BaseModel):
    budget_id: UUID
    user_id: UUID
    category_id: UUID
    allocated_amount: float
    created_at: Optional[datetime] = None
