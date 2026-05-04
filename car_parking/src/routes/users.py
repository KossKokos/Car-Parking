from fastapi import APIRouter, Depends, status, HTTPException
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session

from car_parking.src.database.db import get_db
from car_parking.src.database.models import User
from car_parking.src.repository import users as repository_users
from car_parking.src.services.auth import service_auth
from car_parking.src.schemas.users import UserParkingResponse
from car_parking.src.schemas.parking import ParkingInfo
from car_parking.src.services import (
    roles as service_roles,
    logout as service_logout,
    banned as service_banned
)
from car_parking.src.services.exceptions import user_exceptions as user_exception

router = APIRouter(prefix="/users", tags=["users"])
security = HTTPBearer()

allowd_operation = service_roles.RoleRights(["user", "admin"])
allowd_operation_by_admin = service_roles.RoleRights(["admin"])


def _raise_for_user_domain_error(error: user_exception.UserDomainError) -> None:
    if isinstance(error, (user_exception.CarNotRegisteredError, user_exception.UserNotFoundError)):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        )

    if isinstance(error, user_exception.UserTariffNotFoundError):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        )

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=str(error),
    )


@router.get(
    "/me",
    response_model=UserParkingResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[
        Depends(service_logout.logout_dependency),
        Depends(allowd_operation),
        Depends(service_banned.banned_dependency),
    ],
)
async def read_users_me(
    current_user: User = Depends(service_auth.get_current_user),
    db: Session = Depends(get_db),
):
    try:
        return await repository_users.get_user_me(current_user, db)
    except user_exception.UserDomainError as error:
        _raise_for_user_domain_error(error)


@router.get(
    "/profile",
    response_model=ParkingInfo,
    status_code=status.HTTP_200_OK,
    dependencies=[
        Depends(service_logout.logout_dependency),
        Depends(allowd_operation),
        Depends(service_banned.banned_dependency),
    ],
    description="Any User",
)
async def get_user_profile(
    current_user: User = Depends(service_auth.get_current_user),
    db: Session = Depends(get_db),
):
    try:
        return await repository_users.get_parking_info(
            current_user.license_plate,
            db,
        )
    except user_exception.UserDomainError as error:
        _raise_for_user_domain_error(error)
