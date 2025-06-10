from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import date
from uuid import UUID



class UserCreate(BaseModel):
    email: EmailStr
    full_name: str
    phonenumber: Optional[str]
    data_of_birth: Optional[date]
    password: str
    avatar_url: Optional[str] = None  # Optional field for avatar during creation


class UserOut(BaseModel):
    user_id: UUID
    email: str
    full_name: str
    phonenumber: Optional[str] = None
    data_of_birth: Optional[date] = None
    avatar_url: Optional[str] = None  # Public URL of the profile picture


class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    phonenumber: Optional[str] = None
    email: Optional[EmailStr] = None
    data_of_birth: Optional[date] = None
    avatar_url: Optional[str] = None


class DeleteAccountRequest(BaseModel):
    user_id: str
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
