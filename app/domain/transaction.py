from datetime import datetime
from typing import Optional
from uuid import UUID
from enum import Enum


class TransactionType(str, Enum):
    INCOME = "Income"
    EXPENSE = "Expense"


class Transaction:
    def __init__(
        self,
        transaction_id: UUID,
        user_id: UUID,
        category_id: UUID,
        description: str,
        amount: float,
        transaction_type: TransactionType,
        created_at: Optional[datetime] = None  
    ):
        self.transaction_id = transaction_id
        self.user_id = user_id
        self.category_id = category_id
        self.description = description
        self.created_at = created_at
        self.amount = amount
        self.transaction_type = transaction_type
