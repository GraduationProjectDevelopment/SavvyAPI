from fastapi import APIRouter, HTTPException
from uuid import UUID
from typing import List
from app.models.budget_dto import (
    CreateBudgetRequest,
    UpdateBudgetRequest,
    BudgetResponse,
)
from app.services.implementation.budget_service import BudgetService
from app.infrastructure.implementation.budget_repository import BudgetRepository
from app.services.interfaces.budget_service import IBudgetService


router = APIRouter()
budget_service: IBudgetService = BudgetService(BudgetRepository())


@router.post("/", response_model=BudgetResponse)
def create_budget(data: CreateBudgetRequest):
    budget = budget_service.create_budget(data)
    if not budget:
        raise HTTPException(status_code=400, detail="Budget creation failed")
    return budget


@router.get("/", response_model=List[BudgetResponse])
def get_all():
    return budget_service.get_all_budgets()


@router.get("/{budget_id}", response_model=BudgetResponse)
def get_by_id(budget_id: UUID):
    budget = budget_service.get_budget_by_id(budget_id)
    if not budget:
        raise HTTPException(status_code=404, detail="Budget not found")
    return budget


@router.get("/user/{user_id}", response_model=List[BudgetResponse])
def get_by_user(user_id: UUID):
    return budget_service.get_budgets_by_user(user_id)


@router.put("/", response_model=BudgetResponse)
def update_budget(data: UpdateBudgetRequest):
    updated = budget_service.update_budget(data)
    if not updated:
        raise HTTPException(status_code=404, detail="Budget not found or update failed")
    return updated


@router.delete("/{budget_id}")
def delete_budget(budget_id: UUID):
    success = budget_service.delete_budget(budget_id)
    if not success:
        raise HTTPException(status_code=404, detail="Budget not found")
    return {"message": "Budget deleted successfully"}


@router.get("/category/{category_id}", response_model=List[BudgetResponse])
def get_by_category(category_id: UUID):
    return budget_service.get_budget_by_category_id(category_id)
