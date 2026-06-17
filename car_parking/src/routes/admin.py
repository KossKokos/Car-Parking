from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from car_parking.src.database.db import get_db
from car_parking.src.database.models import User
from car_parking.src.repository import (
    admin as repository_admin,
    car as repository_cars,
    users as repository_users,
)
from car_parking.src.schemas.parking import ParkingInfo
from car_parking.src.schemas.users import UserResponse, UserRoleUpdate
from car_parking.src.services import (
    logout as service_logout,
    roles as service_roles,
)
from car_parking.src.services.exceptions.user_exceptions import (
    CarNotRegisteredError,
    UserDomainError,
    UserTariffNotFoundError,
)

from car_parking.src.services.auth import service_auth
from car_parking.src.conf.constants import SUPERADMIN_USER_ID


router = APIRouter(prefix="/admin", tags=["admin"])

allowed_admin = service_roles.RoleRights(["admin"])

ADMIN_DEPENDENCIES = [
    Depends(service_logout.logout_dependency),
    Depends(allowed_admin),
]


async def _get_user_or_404(user_id: int, db: Session) -> User:
    """Load an admin target user or raise the route's 404 response."""
    user = await repository_users.get_user_by_id(user_id, db)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    return user


def _ensure_can_change_user_ban_status(
    *,
    target_user: User,
    current_user: User,
) -> None:
    """Prevent admins from banning themselves or changing the superadmin."""
    if current_user.id == target_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permission denied. User cannot change own banned status.",
        )

    if target_user.id == SUPERADMIN_USER_ID:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permission denied. Superadmin status cannot be changed.",
        )


def _ensure_can_delete_user(
    *,
    target_user: User,
    current_user: User,
) -> None:
    """Enforce delete rules for self, superadmin, and admin targets."""
    if current_user.id == target_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permission denied. Cannot delete own account.",
        )

    if target_user.id == SUPERADMIN_USER_ID:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permission denied. Superadmin user cannot be deleted.",
        )

    if target_user.role == "admin" and current_user.id != SUPERADMIN_USER_ID:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permission denied. Only superadmin can delete another admin.",
        )


def _ensure_can_change_user_role(
    *,
    target_user: User,
    current_user: User,
) -> None:
    """Enforce role-change rules for protected and admin accounts."""
    if current_user.id == target_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permission denied. Own role cannot be changed.",
        )

    if target_user.id == SUPERADMIN_USER_ID:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permission denied. Superadmin role cannot be changed.",
        )

    if target_user.role == "admin" and current_user.id != SUPERADMIN_USER_ID:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Permission denied. Admin role can be changed only by "
                "superadmin."
            ),
        )


def _build_car_ban_response(
    *,
    car,
    user: User | None,
    action: str,
) -> dict:
    """Build a consistent car-ban response for registered and unregistered cars."""
    user_id_key = f"{action} user id"
    user_email_key = f"{action} user email"

    if user is None:
        return {
            "car id": car.id,
            "license plate": car.license_plate,
            "car ban status": car.banned,
            user_id_key: "not registered user",
            user_email_key: "N/A",
            "user ban status": "N/A",
        }

    return {
        "car id": car.id,
        "license plate": car.license_plate,
        "car ban status": car.banned,
        user_id_key: user.id,
        user_email_key: user.email,
        "user ban status": user.banned,
    }


def _raise_for_user_domain_error(error: UserDomainError) -> None:
    """Translate user domain errors into HTTP exceptions for admin routes."""
    if isinstance(error, CarNotRegisteredError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        )

    if isinstance(error, UserTariffNotFoundError):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        )

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=str(error),
    )


@router.get(
    "/",
    status_code=status.HTTP_200_OK,
    dependencies=ADMIN_DEPENDENCIES,
)
async def get_all_usernames(
    current_user: User = Depends(service_auth.get_current_user),
    db: Session = Depends(get_db),
):
    usernames = await repository_admin.return_all_users(db)
    return usernames


@router.patch(
    "/ban/{user_id}",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
    dependencies=ADMIN_DEPENDENCIES,
)
async def ban_user(
    user_id: int,
    current_user: User = Depends(service_auth.get_current_user),
    db: Session = Depends(get_db),
):
    user = await _get_user_or_404(user_id, db)

    _ensure_can_change_user_ban_status(
        target_user=user,
        current_user=current_user,
    )

    return await repository_admin.set_user_banned_status(
        user,
        True,
        db,
    )


@router.patch(
    "/unban/{user_id}",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
    dependencies=ADMIN_DEPENDENCIES,
)
async def unban_user(
    user_id: int,
    current_user: User = Depends(service_auth.get_current_user),
    db: Session = Depends(get_db),
):
    user = await _get_user_or_404(user_id, db)

    _ensure_can_change_user_ban_status(
        target_user=user,
        current_user=current_user,
    )

    return await repository_admin.set_user_banned_status(
        user,
        False,
        db,
    )


@router.delete(
    "/{user_id}",
    status_code=status.HTTP_200_OK,
    dependencies=ADMIN_DEPENDENCIES,
)
async def delete_user(
    user_id: int,
    current_user: User = Depends(service_auth.get_current_user),
    db: Session = Depends(get_db),
):
    user = await _get_user_or_404(user_id, db)

    _ensure_can_delete_user(
        target_user=user,
        current_user=current_user,
    )

    await repository_users.delete_user(user_id, db)

    return {"message": "User successfully deleted"}


@router.patch(
    "/change_role/{user_id}",
    response_model=UserResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=ADMIN_DEPENDENCIES,
)
async def change_user_role(
    user_id: int,
    body: UserRoleUpdate,
    current_user: User = Depends(service_auth.get_current_user),
    db: Session = Depends(get_db),
):
    user = await _get_user_or_404(user_id, db)

    _ensure_can_change_user_role(
        target_user=user,
        current_user=current_user,
    )

    if body.role not in {"admin", "user"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid role provided",
        )

    return await repository_admin.change_user_role(user, body, db)


@router.patch(
    "/ban_car/{license_plate}",
    status_code=status.HTTP_200_OK,
    dependencies=ADMIN_DEPENDENCIES,
)
async def ban_car(
    license_plate: str,
    db: Session = Depends(get_db),
):
    normalized_license_plate = license_plate.upper()

    car = await repository_cars.get_car_by_license_plate(
        normalized_license_plate,
        db,
    )

    if car is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Car not found",
        )

    user = await repository_users.get_user_by_car_license_plate(
        normalized_license_plate,
        db,
    )

    if user is not None and user.id != SUPERADMIN_USER_ID:
        await repository_admin.set_user_banned_status(user, True, db)

    await repository_cars.set_car_banned_status(car, True, db)

    return _build_car_ban_response(
        car=car,
        user=user,
        action="banned",
    )


@router.patch(
    "/unban_car/{license_plate}",
    status_code=status.HTTP_200_OK,
    dependencies=ADMIN_DEPENDENCIES,
)
async def unban_car(
    license_plate: str,
    db: Session = Depends(get_db),
):
    normalized_license_plate = license_plate.upper()

    car = await repository_cars.get_car_by_license_plate(
        normalized_license_plate,
        db,
    )

    if car is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Car not found",
        )

    user = await repository_users.get_user_by_car_license_plate(
        normalized_license_plate,
        db,
    )

    if user is not None and user.id != SUPERADMIN_USER_ID:
        await repository_admin.set_user_banned_status(user, False, db)

    await repository_cars.set_car_banned_status(car, False, db)

    return _build_car_ban_response(
        car=car,
        user=user,
        action="unbanned",
    )


@router.get(
    "/search_user/{license_plate}",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
    dependencies=ADMIN_DEPENDENCIES,
)
async def search_user_by_license_plate(
    license_plate: str,
    db: Session = Depends(get_db),
):
    normalized_license_plate = license_plate.upper()

    user = await repository_users.get_user_by_car_license_plate(
        normalized_license_plate,
        db,
    )

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"The user with license plate {normalized_license_plate} is not registered.",
        )

    return user


@router.get(
    "/create_csv/{license_plate}/{filename}",
    status_code=status.HTTP_200_OK,
    dependencies=ADMIN_DEPENDENCIES,
    responses={
        status.HTTP_200_OK: {
            "description": "Parking history CSV file download.",
            "content": {
                "text/csv": {
                    "schema": {
                        "type": "string",
                        "format": "binary",
                    },
                },
            },
        },
    },
)
async def create_csv_file(
    license_plate: str,
    filename: str,
    db: Session = Depends(get_db),
):
    try:
        csv_content, download_filename = await repository_admin.create_parking_csv(
            license_plate,
            filename,
            db,
        )
        return Response(
            content=csv_content,
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="{download_filename}"'
                ),
            },
        )

    except UserDomainError as error:
        _raise_for_user_domain_error(error)


@router.get(
    "/get_profile/{license_plate}",
    response_model=ParkingInfo,
    status_code=status.HTTP_200_OK,
    dependencies=ADMIN_DEPENDENCIES,
)
async def get_profile_by_car(
    license_plate: str,
    db: Session = Depends(get_db),
):
    try:
        return await repository_users.get_parking_info(
            license_plate,
            db,
        )
    except UserDomainError as error:
        _raise_for_user_domain_error(error)


@router.get(
    "/users",
    dependencies=ADMIN_DEPENDENCIES,
)
async def get_all_users(db: Session = Depends(get_db)):
    users = await repository_admin.get_all_users(db)
    return users


@router.patch(
    "/change_tariff/{user_id}",
    dependencies=ADMIN_DEPENDENCIES,
)
async def change_user_tariff(
    user_id: int,
    new_tariff: str,
    db: Session = Depends(get_db),
):
    user = await repository_admin.change_tariff(
        user_id,
        new_tariff,
        db,
    )

    return {
        "message": "Tariff changed successfully",
        "user_id": user.id,
        "tariff_id": user.tariff_id,
    }


@router.post(
    "/create_tariff/{tariff_name}/{tariff_value}",
    status_code=status.HTTP_201_CREATED,
    dependencies=ADMIN_DEPENDENCIES,
)
async def create_tariff(
    tariff_name: str,
    tariff_value: int,
    db: Session = Depends(get_db),
):
    tariff = await repository_admin.add_tariff(
        tariff_name,
        tariff_value,
        db,
    )

    return {
        "message": f"Tariff {tariff.tariff_name} has been created",
        "tariff": {
            "id": tariff.id,
            "tariff_name": tariff.tariff_name,
            "tariff_value": tariff.tariff_value,
        },
    }
