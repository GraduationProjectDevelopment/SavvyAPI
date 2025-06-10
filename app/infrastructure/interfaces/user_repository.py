from abc import ABC, abstractmethod
from app.domain.user import User
from typing import List, Optional
from uuid import UUID
from fastapi import UploadFile


class IUserRepository(ABC):
    @abstractmethod
    def create_user(self, user: User) -> Optional[User]:
        pass

    @abstractmethod
    def get_user_by_id(self, user_id: UUID) -> Optional[User]:
        pass

    @abstractmethod
    def update_user(self, user: User) -> Optional[User]:
        pass

    @abstractmethod
    def delete_user(self, user_id: UUID) -> bool:
        pass

    @abstractmethod
    def upload_avatar(self, user_id: UUID, file: UploadFile) -> Optional[str]:
        pass

    @abstractmethod
    def get_all_users(self) -> List[User]:
        pass

    @abstractmethod
    def login(self, email: str, password: str) -> Optional[dict]:
        pass
