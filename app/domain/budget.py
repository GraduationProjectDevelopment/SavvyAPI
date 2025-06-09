from uuid import UUID
from typing import Optional
from datetime import datetime


class Budget:
    def __init__(
        self,
        budget_id: UUID,
        user_id: UUID,
        category_id: UUID,
        allocated_amount: float,
        created_at: Optional[datetime] = None,
    ):
        self.budget_id = budget_id
        self.user_id = user_id
        self.category_id = category_id
        self.allocated_amount = allocated_amount
        self.created_at = created_at
