from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID
from app.domain.budget import Budget


class IBudgetRepository(ABC):
    @abstractmethod
    def create_budget(self, budget: Budget) -> Optional[Budget]:
        pass

    @abstractmethod
    def get_all_budgets(self) -> List[Budget]:
        pass

    @abstractmethod
    def get_budget_by_id(self, budget_id) -> Optional[Budget]:
        pass

    @abstractmethod
    def get_budgets_by_user(self, user_id) -> List[Budget]:
        pass

    @abstractmethod
    def update_budget(self, budget: Budget) -> Optional[Budget]:
        pass

    @abstractmethod
    def delete_budget(self, budget_id) -> bool:
        pass

    @abstractmethod
    def get_budget_by_category_id(self, category_id: UUID) -> List[Budget]:
        pass
