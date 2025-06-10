from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from app.models.user_dto import (
    UserCreate,
    UserOut,
    UserUpdate,
    LoginRequest,
    LoginResponse,
)
from app.services.implementation.user_service import UserService
from app.services.interfaces.user_service import IUserService
from app.infrastructure.implementation.user_repository import UserRepository
from app.domain.user import User
from uuid import UUID, uuid4
from passlib.context import CryptContext
from fastapi import UploadFile

router = APIRouter()
# Use interface for type hinting
user_service: IUserService = UserService(UserRepository())
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


@router.post("/", response_model=UserOut, summary="Create new user")
def create_user(user_dto: UserCreate):
    hashed_pw = pwd_context.hash(user_dto.password)

    user = User(
        user_id=uuid4(),
        email=user_dto.email,
        full_name=user_dto.full_name,
        phonenumber=user_dto.phonenumber,
        data_of_birth=user_dto.data_of_birth,
        avatar_url=user_dto.avatar_url,
        password=hashed_pw,
    )
    created = user_service.create_user(user)
    if not created:
        raise HTTPException(status_code=400, detail="User creation failed")
    return UserOut(**created.__dict__)


@router.post("/upload-profile-image")
async def upload_profile_image(
    user_id: UUID = Query(...), file: UploadFile = File(...)
):
    avatar_url = user_service.upload_avatar(user_id, file)
    if not avatar_url:
        raise HTTPException(status_code=404, detail="User not found or upload failed")
    return {"avatar_url": avatar_url}


@router.get("/{user_id}", response_model=UserOut)
def get_user_by_id(user_id: UUID):
    user = user_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return UserOut(**user.__dict__)


@router.put("/{user_id}", response_model=UserOut)
def update_user(user_id: UUID, update_data: UserUpdate):
    user = User(
        user_id=user_id,
        email=update_data.email,
        full_name=update_data.full_name,
        phonenumber=update_data.phonenumber,
        data_of_birth=update_data.data_of_birth,
        avatar_url=update_data.avatar_url,
    )
    updated = user_service.update_user(user)
    if not updated:
        raise HTTPException(status_code=400, detail="Update failed")
    return UserOut(**updated.__dict__)


@router.patch("/{user_id}", response_model=UserOut)
def update_user_partial(user_id: UUID, update_data: UserUpdate):
    existing_user = user_service.get_user_by_id(user_id)
    if not existing_user:
        raise HTTPException(status_code=404, detail="User not found")

    # Merge updates
    if update_data.full_name is not None:
        existing_user.full_name = update_data.full_name
    if update_data.phonenumber is not None:
        existing_user.phonenumber = update_data.phonenumber
    if update_data.email is not None:
        existing_user.email = update_data.email
    if update_data.avatar_url is not None:
        existing_user.avatar_url = update_data.avatar_url
    if update_data.data_of_birth is not None:
        existing_user.data_of_birth = update_data.data_of_birth
    if update_data.password:
        hashed_pw = pwd_context.hash(update_data.password)
        existing_user.password = hashed_pw  # You must add this to entity + repo

    updated_user = user_service.update_user(existing_user)
    if not updated_user:
        raise HTTPException(status_code=400, detail="Failed to update user")

    return UserOut(**updated_user.__dict__)


@router.delete("/{user_id}")
def delete_user(user_id: UUID):
    success = user_service.delete_user(user_id)
    if not success:
        raise HTTPException(status_code=404, detail="Delete failed")
    return {"message": "User deleted successfully"}


@router.get("/", response_model=List[UserOut], summary="Get all users")
def get_all_users():
    users = user_service.get_all_users()
    return [UserOut(**user.__dict__) for user in users]


@router.post("/login", response_model=LoginResponse)
def login(request: LoginRequest):
    result = user_service.login(request.email, request.password)
    if not result:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    return LoginResponse(
        access_token=result["access_token"], refresh_token=result["refresh_token"]
    )
