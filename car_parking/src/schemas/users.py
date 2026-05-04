from datetime import datetime
from typing import Literal

from pydantic import BaseModel, EmailStr, Field

from car_parking.src.schemas.parking import CurrentParking


class UserModel(BaseModel):
    username: str = Field(..., min_length=5, max_length=15)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=15)
    license_plate: str = Field(..., min_length=2, max_length=30)


class UserResponse(BaseModel):
    username: str
    email: EmailStr
    license_plate: str = Field(..., min_length=2, max_length=30)

    class Config:
        orm_mode = True


class UserParkingResponse(BaseModel):
    user: UserResponse
    parking: CurrentParking | str

    class Config:
        orm_mode = True


class ChangePassword(BaseModel):
    new_password: str = Field(..., min_length=8, max_length=15)


class UserRoleUpdate(BaseModel):
    role: Literal["admin", "user"]


class UserByCarResponse(BaseModel):
    id: int
    username: str
    email: EmailStr
    created_at: datetime
    confirmed: bool
    license_plate: str
    banned: bool
    tariff: int