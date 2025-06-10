from app.infrastructure.interfaces.user_repository import IUserRepository
from app.services.interfaces.user_service import IUserService
from app.domain.user import User
from typing import List, Optional
from uuid import UUID, uuid4
from fastapi import UploadFile


class UserService(IUserService):
    def __init__(self, repo: IUserRepository):
        self.repo = repo

    def create_user(self, user: User) -> Optional[User]:
        return self.repo.create_user(user)

    def get_user_by_id(self, user_id: UUID) -> Optional[User]:
        return self.repo.get_user_by_id(user_id)

    def update_user(self, user: User) -> Optional[User]:
        return self.repo.update_user(user)

    def delete_user(self, user_id: UUID) -> bool:
        return self.repo.delete_user(user_id)

    def upload_avatar(self, user_id: UUID, file: UploadFile) -> Optional[str]:
        return self.repo.upload_avatar(user_id, file)

    def get_all_users(self) -> List[User]:
        return self.repo.get_all_users()

    def login(self, email: str, password: str) -> Optional[dict]:
        return self.repo.login(email, password)
