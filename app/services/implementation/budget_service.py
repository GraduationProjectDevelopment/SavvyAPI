from uuid import UUID, uuid4
from typing import Optional, List
from datetime import datetime, timezone

from app.services.interfaces.budget_service import IBudgetService
from app.models.budget_dto import (
    CreateBudgetRequest,
    UpdateBudgetRequest,
    BudgetResponse,
)
from app.domain.budget import Budget
from app.infrastructure.interfaces.budget_repository import IBudgetRepository
from datetime import datetime


class BudgetService(IBudgetService):
    def __init__(self, budget_repo: IBudgetRepository):
        self.budget_repo = budget_repo

    def create_budget(self, data: CreateBudgetRequest) -> Optional[BudgetResponse]:
        budget = Budget(
            budget_id=uuid4(),
            user_id=data.user_id,
            category_id=data.category_id,
            allocated_amount=data.allocated_amount,
            created_at=datetime.now(timezone.utc),
        )
        created = self.budget_repo.create_budget(budget)
        if not created:
            return None
        return BudgetResponse(**created.__dict__)

    def get_all_budgets(self) -> List[BudgetResponse]:
        budgets = self.budget_repo.get_all_budgets()
        return [BudgetResponse(**b.__dict__) for b in budgets]

    def get_budget_by_id(self, budget_id) -> Optional[BudgetResponse]:
        budget = self.budget_repo.get_budget_by_id(budget_id)
        return BudgetResponse(**budget.__dict__) if budget else None

    def get_budgets_by_user(self, user_id) -> List[BudgetResponse]:
        budgets = self.budget_repo.get_budgets_by_user(user_id)
        return [BudgetResponse(**b.__dict__) for b in budgets]

    def update_budget(self, data: UpdateBudgetRequest) -> Optional[BudgetResponse]:
        existing = self.budget_repo.get_budget_by_id(data.budget_id)
        if not existing:
            return None
        updated = Budget(
            budget_id=data.budget_id,
            user_id=existing.user_id,
            category_id=existing.category_id,
            allocated_amount=data.allocated_amount,
            created_at=existing.created_at,
        )
        saved = self.budget_repo.update_budget(updated)
        return BudgetResponse(**saved.__dict__) if saved else None

    def delete_budget(self, budget_id) -> bool:
        return self.budget_repo.delete_budget(budget_id)

    def get_budget_by_category_id(self, category_id: UUID) -> List[BudgetResponse]:
        budgets = self.budget_repo.get_budget_by_category_id(category_id)
        return [BudgetResponse(**budget.__dict__) for budget in budgets]
