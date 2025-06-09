from typing import Optional, List
from uuid import UUID
from app.domain.budget import Budget
from app.infrastructure.interfaces.budget_repository import IBudgetRepository
from app.core.supabase import get_supabase
from datetime import datetime


class BudgetRepository(IBudgetRepository):
    def create_budget(self, budget: Budget) -> Optional[Budget]:
        data = {
            "budget_id": str(budget.budget_id),
            "user_id": str(budget.user_id),
            "category_id": str(budget.category_id),
            "allocated_amount": budget.allocated_amount,
        }
        response = get_supabase().table("budgets").insert(data).execute()
        if response.data:
            return budget
        return None

    def get_all_budgets(self) -> List[Budget]:
        response = get_supabase().table("budgets").select("*").execute()
        return [Budget(**item) for item in response.data] if response.data else []

    def get_budget_by_id(self, budget_id: UUID) -> Optional[Budget]:
        response = (
            get_supabase()
            .table("budgets")
            .select("*")
            .eq("budget_id", str(budget_id))
            .single()
            .execute()
        )
        return Budget(**response.data) if response.data else None

    def get_budgets_by_user(self, user_id: UUID) -> List[Budget]:
        response = (
            get_supabase()
            .table("budgets")
            .select("*")
            .eq("user_id", str(user_id))
            .execute()
        )
        return [Budget(**item) for item in response.data] if response.data else []

    def update_budget(self, budget: Budget) -> Optional[Budget]:
        data = {
            "allocated_amount": budget.allocated_amount,
        }
        response = (
            get_supabase()
            .table("budgets")
            .update(data)
            .eq("budget_id", str(budget.budget_id))
            .execute()
        )
        return budget if response.data else None

    def delete_budget(self, budget_id: UUID) -> bool:
        response = (
            get_supabase()
            .table("budgets")
            .delete()
            .eq("budget_id", str(budget_id))
            .execute()
        )
        return bool(response.data)

    def get_budget_by_category_id(self, category_id: UUID) -> List[Budget]:
        response = (
            get_supabase()
            .table("budgets")
            .select("*")
            .eq("category_id", str(category_id))
            .execute()
        )
        return [Budget(**record) for record in response.data] if response.data else []
