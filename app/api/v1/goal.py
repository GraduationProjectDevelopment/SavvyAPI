from fastapi import APIRouter, HTTPException
from uuid import UUID
from typing import List
from app.models.goal_dto import CreateGoalRequest, GoalResponse, UpdateGoalRequest
from app.services.implementation.goal_service import GoalService
from app.infrastructure.implementation.goal_repository import GoalRepository
from app.services.interfaces.goal_service import IGoalService


router = APIRouter()
goal_service: IGoalService = GoalService(GoalRepository())


@router.post("/", response_model=GoalResponse)
def create_goal(data: CreateGoalRequest):
    goal = goal_service.create_goal(data)
    if not goal:
        raise HTTPException(status_code=400, detail="Goal creation failed")
    return goal


@router.get("/{goal_id}", response_model=GoalResponse)
def get_goal(goal_id: UUID):
    goal = goal_service.get_goal(goal_id)
    if not goal:
        raise HTTPException(status_code=404, detail="Goal not found")
    return goal


@router.get("/user/{user_id}", response_model=List[GoalResponse])
def get_user_goals(user_id: UUID):
    return goal_service.get_user_goals(user_id)


@router.put("/{goal_id}", response_model=GoalResponse)
def update_goal(goal_id: UUID, data: UpdateGoalRequest):
    updated = goal_service.update_goal(goal_id, data)
    if not updated:
        raise HTTPException(status_code=404, detail="Goal not found or update failed")
    return updated


@router.delete("/{goal_id}", status_code=204)
def delete_goal(goal_id: UUID):
    success = goal_service.delete_goal(goal_id)
    if not success:
        raise HTTPException(status_code=404, detail="Goal not found or delete failed")
