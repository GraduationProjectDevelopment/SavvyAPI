from abc import ABC, abstractmethod
from typing import List, Optional
from app.domain.goal import Goal
from uuid import UUID


class IGoalRepository(ABC):
    @abstractmethod
    def create_goal(self, goal: Goal) -> Optional[Goal]:
        pass

    @abstractmethod
    def get_goal(self, goal_id: UUID) -> Optional[Goal]:
        pass

    @abstractmethod
    def get_goals_by_user(self, user_id: UUID) -> List[Goal]:
        pass

    @abstractmethod
    def update_goal(self, goal_id: UUID, goal_data: dict) -> Optional[Goal]:
        pass

    @abstractmethod
    def delete_goal(self, goal_id: UUID) -> bool:
        pass
