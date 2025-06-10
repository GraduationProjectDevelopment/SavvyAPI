from typing import List, Optional
from uuid import uuid4, UUID
from app.domain.goal import Goal
from app.services.interfaces.goal_service import IGoalService
from app.models.goal_dto import CreateGoalRequest, UpdateGoalRequest, GoalResponse
from app.infrastructure.interfaces.goal_repository import IGoalRepository
from datetime import datetime


class GoalService(IGoalService):
    def __init__(self, goal_repository: IGoalRepository):
        self.goal_repository = goal_repository

    def create_goal(self, data: CreateGoalRequest) -> Optional[GoalResponse]:
        goal = Goal(
            goal_id=uuid4(),
            user_id=data.user_id,
            goal_name=data.goal_name,
            target_amount=data.target_amount,
            amount_saved=0.0,
            created_at=datetime.utcnow(),
        )
        result = self.goal_repository.create_goal(goal)
        return GoalResponse(**vars(result)) if result else None

    def get_goal(self, goal_id: UUID) -> Optional[GoalResponse]:
        goal = self.goal_repository.get_goal(goal_id)
        return GoalResponse(**vars(goal)) if goal else None

    def get_user_goals(self, user_id: UUID) -> List[GoalResponse]:
        return [
            GoalResponse(**vars(g))
            for g in self.goal_repository.get_goals_by_user(user_id)
        ]

    def update_goal(
        self, goal_id: UUID, data: UpdateGoalRequest
    ) -> Optional[GoalResponse]:
        update_data = {k: v for k, v in data.dict().items() if v is not None}
        goal = self.goal_repository.update_goal(goal_id, update_data)
        return GoalResponse(**vars(goal)) if goal else None

    def delete_goal(self, goal_id: UUID) -> bool:
        return self.goal_repository.delete_goal(goal_id)
