from app.infrastructure.interfaces.user_repository import IUserRepository
from app.domain.user import User
from app.core.supabase import get_supabase
from uuid import UUID
from typing import List, Optional
from datetime import datetime
from uuid import uuid4
from fastapi import UploadFile


class UserRepository(IUserRepository):
    def create_user(self, user: User) -> Optional[User]:
        data = {
            "email": user.email,
            "full_name": user.full_name,
            "phonenumber": user.phonenumber,
            "data_of_birth": (
                user.data_of_birth.isoformat() if user.data_of_birth else None
            ),
            "avatar_url": user.avatar_url,
            "password": user.password,
        }
        response = get_supabase().table("users").insert(data).execute()
        if response.data:
            return user
        return None

    # Implement get, update, delete similarly

    def get_user_by_id(self, user_id: UUID) -> Optional[User]:
        result = (
            get_supabase()
            .table("users")
            .select("*")
            .eq("user_id", str(user_id))
            .single()
            .execute()
        )
        if not result.data:
            return None
        user = result.data
        return User(
            user_id=UUID(user["user_id"]),
            email=user["email"],
            full_name=user["full_name"],
            phonenumber=user.get("phonenumber"),
            data_of_birth=user.get("data_of_birth"),
            avatar_url=user.get("avatar_url"),
        )

    def update_user(self, user: User) -> Optional[User]:
        data_of_birth = user.data_of_birth
        if isinstance(data_of_birth, str):
            data_of_birth = datetime.fromisoformat(data_of_birth).date()

        user_data = {
            "email": user.email,
            "full_name": user.full_name,
            "phonenumber": user.phonenumber,
            "data_of_birth": data_of_birth.isoformat() if data_of_birth else None,
            "avatar_url": user.avatar_url,
            "password": user.password,
        }

        result = (
            get_supabase()
            .table("users")
            .update(user_data)
            .eq("user_id", str(user.user_id))
            .execute()
        )
        if not result.data:
            return None
        return user

    def delete_user(self, user_id: UUID) -> bool:
        try:
            result = (
                get_supabase()
                .table("users")
                .delete()
                .eq("user_id", str(user_id))
                .execute()
            )
            return bool(result.data)
        except Exception as e:
            print(f"Delete error: {e}")
            return False

    def upload_avatar(self, user_id: UUID, file: UploadFile) -> Optional[str]:
        supabase = get_supabase()

        file_ext = file.filename.split(".")[-1]
        filename = f"{uuid4()}.{file_ext}"
        file_bytes = file.file.read()

        try:
            supabase.storage.from_("avatars").upload(
                path=filename,
                file=file_bytes,
                file_options={"content-type": file.content_type},
            )
            public_url = supabase.storage.from_("avatars").get_public_url(filename)

            user = self.get_user_by_id(user_id)
            if not user:
                return None

            user.avatar_url = public_url
            self.update_user(user)

            return public_url
        except Exception as e:
            print(f"Upload failed: {e}")
            return None

    def get_all_users(self) -> List[User]:
        result = get_supabase().table("users").select("*").execute()
        users = result.data or []
        return [
            User(
                user_id=UUID(user["user_id"]),
                email=user["email"],
                full_name=user["full_name"],
                phonenumber=user.get("phonenumber"),
                data_of_birth=user.get("data_of_birth"),
                avatar_url=user.get("avatar_url"),
                password=user.get("password"),  # Optional: Include if needed
            )
            for user in users
        ]

    def login(self, email: str, password: str) -> Optional[dict]:
        try:
            result = get_supabase().auth.sign_in_with_password(
                {"email": email, "password": password}
            )
            if result.session:
                return {
                    "access_token": result.session.access_token,
                    "refresh_token": result.session.refresh_token,
                }
            return None
        except Exception as e:
            print(f"Login error: {e}")
            return None
