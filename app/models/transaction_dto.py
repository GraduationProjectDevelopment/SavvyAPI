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
    description: str
    created_at: datetime
    amount: float
    transaction_type: TransactionType
    feedback: str
