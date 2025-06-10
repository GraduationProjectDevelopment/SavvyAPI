from typing import List, Optional
from app.infrastructure.interfaces.goal_repository import IGoalRepository
from app.domain.goal import Goal
from uuid import UUID, uuid4
from app.core.supabase import get_supabase
from datetime import datetime


class GoalRepository(IGoalRepository):
    def create_goal(self, goal: Goal) -> Optional[Goal]:
        data = {
            "goal_id": str(goal.goal_id),
            "user_id": str(goal.user_id),
            "goal_name": goal.goal_name,
            "target_amount": goal.target_amount,
            "amount_saved": goal.amount_saved,
            "created_at": goal.created_at.isoformat(),
        }
        response = get_supabase().table("goals").insert(data).execute()
        return goal if response.data else None

    def get_goal(self, goal_id: UUID) -> Optional[Goal]:
        response = (
            get_supabase()
            .table("goals")
            .select("*")
            .eq("goal_id", str(goal_id))
            .limit(1)
            .execute()
        )

        if response.data:
            data = response.data[0]
            return Goal(
                goal_id=UUID(data["goal_id"]),
                user_id=UUID(data["user_id"]),
                goal_name=data["goal_name"],
                target_amount=data["target_amount"],
                amount_saved=data["amount_saved"],
                created_at=datetime.fromisoformat(data["created_at"]),
            )

        return None

    def get_goals_by_user(self, user_id: UUID) -> List[Goal]:
        response = (
            get_supabase()
            .table("goals")
            .select("*")
            .eq("user_id", str(user_id))
            .execute()
        )
        return [
            Goal(
                goal_id=UUID(row["goal_id"]),
                user_id=UUID(row["user_id"]),
                goal_name=row["goal_name"],
                target_amount=row["target_amount"],
                amount_saved=row["amount_saved"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in response.data
        ]

    def update_goal(self, goal_id: UUID, goal_data: dict) -> Optional[Goal]:
        response = (
            get_supabase()
            .table("goals")
            .update(goal_data)
            .eq("goal_id", str(goal_id))
            .execute()
        )
        if response.data:
            updated = response.data[0]
            return Goal(
                goal_id=UUID(updated["goal_id"]),
                user_id=UUID(updated["user_id"]),
                goal_name=updated["goal_name"],
                target_amount=updated["target_amount"],
                amount_saved=updated["amount_saved"],
                created_at=datetime.fromisoformat(updated["created_at"]),
            )
        return None

    def delete_goal(self, goal_id: UUID) -> bool:
        response = (
            get_supabase().table("goals").delete().eq("goal_id", str(goal_id)).execute()
        )
        return bool(response.data)
