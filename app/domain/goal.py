from uuid import UUID
from datetime import datetime
from typing import Optional


class Goal:
    def __init__(
        self,
        goal_id: UUID,
        user_id: UUID,
        goal_name: str,
        target_amount: float,
        amount_saved: float,
        created_at: Optional[datetime] = None,
    ):
        self.goal_id = goal_id
        self.user_id = user_id
        self.goal_name = goal_name
        self.target_amount = target_amount
        self.amount_saved = amount_saved
        self.created_at = created_at or datetime.utcnow()
