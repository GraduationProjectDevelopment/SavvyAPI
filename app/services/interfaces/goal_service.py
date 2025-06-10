from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID
from app.models.goal_dto import CreateGoalRequest, UpdateGoalRequest, GoalResponse


class IGoalService(ABC):
    @abstractmethod
    def create_goal(self, data: CreateGoalRequest) -> Optional[GoalResponse]:
        pass

    @abstractmethod
    def get_goal(self, goal_id: UUID) -> Optional[GoalResponse]:
        pass

    @abstractmethod
    def get_user_goals(self, user_id: UUID) -> List[GoalResponse]:
        pass

    @abstractmethod
    def update_goal(
        self, goal_id: UUID, data: UpdateGoalRequest
    ) -> Optional[GoalResponse]:
        pass

    @abstractmethod
    def delete_goal(self, goal_id: UUID) -> bool:
        pass
