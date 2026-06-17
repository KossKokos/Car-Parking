from datetime import datetime

import pytz
from sqlalchemy.orm import Session

from car_parking.src.conf.constants import ADMIN_USER_ID, DEFAULT_USER_TARIFF_ID, TIMEZONE
from car_parking.src.database.models import Tariff, User
from car_parking.src.schemas.parking import CurrentParking, ParkingInfo, ParkingResponse
from car_parking.src.schemas.users import (
    UserModel,
    UserParkingResponse,
    UserResponse,
)
from car_parking.src.utils import users_helpers as user_helpers 

from ..services.parking_calculations import (
    calculate_parking_cost,
    calculate_parking_duration_hours,
)
from car_parking.src.services.exceptions import user_exceptions as user_exceptions


async def create_user(body: UserModel, db: Session) -> User:
    """Create a user with normalized plate text and the default tariff."""
    user = User(**body.dict())
    user.license_plate = user_helpers._normalize_license_plate(body.license_plate)
    user.tariff_id = DEFAULT_USER_TARIFF_ID

    db.add(user)
    db.flush()

    if user.id == ADMIN_USER_ID:
        user.role = "admin"
                
    db.commit()
    db.refresh(user)
    return user


async def get_user_by_email(email: str, db: Session) -> User | None:
    return db.query(User).filter(User.email == email).first()


async def get_user_by_username(username: str, db: Session) -> User | None:
    return db.query(User).filter(User.username == username).first()


async def update_token(user: User, refresh_token: str, db: Session) -> None:
    user.refresh_token = refresh_token
    db.commit()
    db.refresh(user)


async def confirmed_email(email: str, db: Session) -> None:
    """Mark a user's email as confirmed, raising if the user is missing."""
    user = await get_user_by_email(email, db)
    
    if user is None:
        raise user_exceptions.UserNotFoundError("User not found.")

    user.confirmed = True
    db.commit()


async def change_password(user: User, new_password: str, db: Session) -> None:
    user.password = new_password
    db.commit() 
    db.refresh(user)


async def get_user_by_id(user_id: int, db: Session) -> User | None:
    return db.query(User).filter(User.id == user_id).first()


async def delete_user(user_id: int, db: Session) -> None:
    user = await get_user_by_id(user_id, db)

    if user is None:
        raise user_exceptions.UserNotFoundError("User not found.")

    db.delete(user)
    db.commit()


async def get_user_by_car_license_plate(
    license_plate: str,
    db: Session,
) -> User | None:
    return db.query(User).filter_by(
        license_plate=user_helpers._normalize_license_plate(license_plate)
    ).first()


async def get_parking_info(license_plate: str, db: Session) -> ParkingInfo:
    """Build completed parking history and aggregate totals for one car."""
    normalized_license_plate = user_helpers._normalize_license_plate(license_plate)

    user = await get_user_by_car_license_plate(normalized_license_plate, db)
    car = user_helpers._get_car_by_license_plate(normalized_license_plate, db)

    if not car:
        raise user_exceptions.CarNotRegisteredError("This car is not registered.")

    parking_info = user_helpers._get_closed_parking_sessions_by_license_plate(
        normalized_license_plate,
        db,
    )

    total_payment_amount = await user_helpers.calculate_amount_cost(parking_info)
    total_parking_time = await user_helpers.calculate_amount_duration(parking_info)
    parking_history = ParkingInfo(
        user=user.username if user else "Unregister user",
        total_payment_amount=total_payment_amount,
        total_parking_time=total_parking_time,
        parking_info=[],
    )
    for parking in parking_info:
        parking_history.parking_info.append(
            ParkingResponse(
                enter_time=user_helpers._format_datetime_for_response(parking.enter_time),
                departure_time=user_helpers._format_datetime_for_response(parking.departure_time),
                license_plate=parking.license_plate,
                amount_paid=parking.amount_paid,
                duration=parking.duration,
                status=parking.status,
            )
        )
    return parking_history


async def get_user_me(user: User, db: Session) -> UserParkingResponse:
    """Return the user's profile with live parking cost when a car is parked."""
    
    user_parking = user_helpers._get_current_parking_session_by_license_plate(
        user.license_plate,
        db,
    )
    
    tariff = db.query(Tariff).filter_by(id=user.tariff_id).first()
    if tariff is None:
        raise user_exceptions.UserTariffNotFoundError("User tariff not found.")
    
    if user_parking:
        time_on_parking = calculate_parking_duration_hours(
            user_parking.enter_time, datetime.now(pytz.timezone(TIMEZONE))
        )
        current_cost = calculate_parking_cost(time_on_parking, tariff.tariff_value)

        user_park = UserParkingResponse(
            user=UserResponse(
                username=user.username,
                email=user.email,
                license_plate=user.license_plate,
            ),
            parking=CurrentParking(
                enter_time=user_helpers._format_datetime_for_response(user_parking.enter_time),
                time_on_parking=time_on_parking,
                parking_cost=current_cost,
            ),
        )
        return user_park
    user_park = UserParkingResponse(
        user=UserResponse(
            username=user.username, email=user.email, license_plate=user.license_plate
        ),
        parking="You don't have a car parked right now.",
    )
    return user_park
