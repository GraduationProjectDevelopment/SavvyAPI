from abc import ABC, abstractmethod
from typing import Optional, List
from uuid import UUID
from app.models.budget_dto import (
    CreateBudgetRequest,
    UpdateBudgetRequest,
    BudgetResponse,
)


class IBudgetService(ABC):
    @abstractmethod
    def create_budget(self, data: CreateBudgetRequest) -> Optional[BudgetResponse]:
        pass

    @abstractmethod
    def get_all_budgets(self) -> List[BudgetResponse]:
        pass

    @abstractmethod
    def get_budget_by_id(self, budget_id: UUID) -> Optional[BudgetResponse]:
        pass

    @abstractmethod
    def get_budgets_by_user(self, user_id: UUID) -> List[BudgetResponse]:
        pass

    @abstractmethod
    def update_budget(self, data: UpdateBudgetRequest) -> Optional[BudgetResponse]:
        pass

    @abstractmethod
    def delete_budget(self, budget_id: UUID) -> bool:
        pass

    @abstractmethod
    def get_budget_by_category_id(self, category_id: UUID) -> List[BudgetResponse]:
        pass
