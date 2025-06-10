from typing import Optional
from pydantic import BaseModel
from uuid import UUID
from datetime import datetime
from enum import Enum


class TransactionType(str, Enum):
    INCOME = "Income"
    EXPENSE = "Expense"


class CreateTransactionRequest(BaseModel):
    user_id: str
    input_text: str


class TransactionResponse(BaseModel):
    transaction_id: UUID
    user_id: UUID
    category_id: UUID
    description: Optional[str] = None
    created_at: Optional[datetime] = None
    amount: float
    transaction_type: TransactionType
    feedback: Optional[str] = None


class UpdateTransactionRequest(BaseModel):
    transaction_id: str
    user_id: str
    category_id: str
    description: str
    amount: float
    transaction_type: TransactionType
