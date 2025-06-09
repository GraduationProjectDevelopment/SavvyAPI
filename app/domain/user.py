from typing import Optional
from datetime import date
from uuid import UUID


class User:
    def __init__(
        self,
        user_id: UUID,
        email: str,
        full_name: str,
        phonenumber: Optional[str] = None,
        data_of_birth: Optional[date] = None,
        avatar_url: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.user_id = user_id
        self.email = email
        self.full_name = full_name
        self.phonenumber = phonenumber
        self.data_of_birth = data_of_birth
        self.avatar_url = avatar_url
        self.password = password
