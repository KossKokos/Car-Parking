from datetime import datetime
import pytz

from fastapi import File, HTTPException, status
from sqlalchemy.orm import Session

from car_parking.src.conf.constants import (
    EXTENSIONS,
    PARKING_AVAILABILITY_DATETIME_FORMAT,
    PARKING_COUNT_DATA,
    TIMEZONE,
)
from car_parking.src.schemas.parking import (
    CurrentParkingAvailabilityData,
    ParkingAvailabilityData,
    ParkingOperationResult,
)
from car_parking.src.utils import parking_helpers as helpers

from ..database.models import Car, Parking, ParkingCount

from ..repository import users as repository_users
from ..repository.car import create_car

from ..services.exceptions.parking_exceptions import (
    CarNotInParkingError,
    ParkingAlreadyClosedError,
    ParkingFullError,
    ParkingPlaceNotFoundError,
)


async def get_parking_place_by_car_license_plate(
    license_plate: str,
    db: Session,
) -> Parking | None:
    return helpers._get_active_parking_place_by_license_plate(
        license_plate=license_plate,
        db=db,
    )


async def create_parking_place(license_plate: str, db: Session) -> Parking:
    parking_place = Parking(license_plate=license_plate)

    db.add(parking_place)
    db.flush()
    db.refresh(parking_place)

    return parking_place


async def change_parking_status_not_authorised(
    parking_place_id: int,
    db: Session,
) -> ParkingOperationResult:
    """Close an unpaid parking session and calculate its invoice immediately."""
    parking_place = helpers._get_parking_place_by_id(parking_place_id, db)

    if parking_place is None:
        return "Parking place not found"

    user = helpers._get_user_by_license_plate(parking_place.license_plate, db)

    parking_place = helpers._apply_invoice_to_parking_place(
        parking_place=parking_place,
        user=user,
        db=db,
    )

    parking_place.status = True
    parking_count = helpers._get_parking_count(db)

    parking = helpers._build_parking_schema(
        parking_place,
        message="The barrier is open, See you next time!",
    )

    helpers._decrease_occupied_count(parking_count)
    db.commit()
    return parking


async def confirm_authorised_payment(
    parking_place_id: int,
    db: Session,
) -> ParkingOperationResult:
    """Mark an invoiced parking session as paid and free one parking space."""
    parking_place = helpers._get_parking_place_by_id(parking_place_id, db)

    if parking_place is None:
        raise ParkingPlaceNotFoundError("Parking place not found.")

    if parking_place.status is True:
        raise ParkingAlreadyClosedError("Parking place is already closed.")

    parking_place.status = True

    parking_count = helpers._get_parking_count(db)
    helpers._decrease_occupied_count(parking_count)

    parking_status = helpers._build_parking_schema(
        parking_place,
        message="The barrier is open, See you next time!",
    )

    db.commit()
    return parking_status


async def calculate_invoice(
    parking_place_id: int,
    db: Session,
) -> Parking:
    """Calculate and persist departure time, duration, and amount for a session."""
    parking_place = helpers._get_parking_place_by_id(parking_place_id, db)

    if parking_place is None:
        raise ParkingPlaceNotFoundError("Parking place not found.")
    
    user = helpers._get_user_by_license_plate(parking_place.license_plate, db)

    parking_place = helpers._apply_invoice_to_parking_place(
        parking_place=parking_place,
        user=user,
        db=db,
    )

    db.commit()
    return parking_place


async def entry_to_the_parking(
    license_plate: str,
    db: Session,
) -> ParkingOperationResult:
    """Create an active parking session unless the car is already parked."""
    car = db.query(Car).filter(Car.license_plate == license_plate).first()

    parking_count = helpers._get_parking_count(db)

    if helpers._is_parking_full(parking_count):
        raise ParkingFullError("Sorry, there are no available parking places.")

    if not car:
        await create_car(license_plate, db)

    parking_place = helpers._get_active_parking_place_by_license_plate(license_plate, db)

    user = await repository_users.get_user_by_car_license_plate(license_plate, db)

    if not parking_place:
        parking_place = await create_parking_place(license_plate, db)

        message = (
            f"Parking successful, please check your email<< {user.email} >> for details"
            if user
            else "Parking successful, to get details please sign up for our Car Parking service"
        )

        helpers._increase_occupied_count(parking_count)

        db.commit()
        db.refresh(parking_place)
        db.refresh(parking_count)

        parking = helpers._build_parking_schema(
            parking_place,
            message=message,
        )

        return parking

    return helpers._build_parking_schema(
        parking_place,
        message="This car already in parking.",
    )


async def exit_from_the_parking(
    license_plate: str,
    db: Session,
) -> ParkingOperationResult:
    """Prepare an invoice for the active parking session of a leaving car."""
    user = await repository_users.get_user_by_car_license_plate(license_plate, db)
    parking_place = helpers._get_active_parking_place_by_license_plate(license_plate, db)

    if parking_place:
        parking_place = await calculate_invoice(parking_place.id, db)

        message = (
            f"Parking invoice sent to your email << {user.email}>>. Please confirm payment"
            if user
            else f"Your parking ID = << {parking_place.id} >>Confirm payment, please."
        )

        parking = helpers._build_parking_schema(
            parking_place,
            message=message,
        )

        return parking
    raise CarNotInParkingError(f"Car {license_plate} is not currently in parking.")


async def seed_parking_count(db: Session) -> None:
    """Insert the initial parking capacity row when the table is empty."""
    if db.query(ParkingCount).count() == 0:
        for data in PARKING_COUNT_DATA:
            parking_count = ParkingCount(**data)
            db.add(parking_count)

        db.commit()


async def is_valid_file_ext(file: File) -> bool:
    file_ext = file.filename.split(".")[-1]
    if file_ext not in EXTENSIONS:
        return False
    return True


APP_TIMEZONE = pytz.timezone(TIMEZONE)


def _parse_parking_availability_datetime(value: str) -> datetime:
    """Parse an availability query timestamp in the configured app timezone."""
    try:
        parsed_datetime = datetime.strptime(
            value,
            PARKING_AVAILABILITY_DATETIME_FORMAT,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Wrong date format. Use YYYY.MM.DD HH:MM, "
                "for example 2026.04.30 14:30"
            ),
        ) from exc

    return APP_TIMEZONE.localize(parsed_datetime)


async def get_parking_availability_at(
    requested_at: datetime,
    db: Session,
) -> ParkingAvailabilityData:
    """Calculate free and occupied places at a requested historical timestamp."""
    parking_count = helpers._get_parking_count(db)

    occupied_places: int = helpers._count_occupied_places_at(requested_at, db)
    free_places: int = parking_count.total_quantity - occupied_places

    if free_places < 0:
        free_places = 0

    return {
        "requested_at": requested_at.strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": TIMEZONE,
        "total_places": parking_count.total_quantity,
        "occupied_places": occupied_places,
        "free_places": free_places,
    }


async def get_current_parking_availability(
    db: Session,
) -> CurrentParkingAvailabilityData:
    """Calculate current availability from active parking rows."""
    parking_count = helpers._get_parking_count(db)

    occupied_places = (
        db.query(Parking)
        .filter(Parking.status.is_(False))
        .count()
    )

    free_places: int = parking_count.total_quantity - occupied_places

    if free_places < 0:
        free_places = 0

    return {
        "total_places": parking_count.total_quantity,
        "occupied_places": occupied_places,
        "free_places": free_places,
        "stored_occupied_quantity": parking_count.occupied_quantity,
    }
