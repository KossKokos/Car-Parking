from fastapi import HTTPException, status
import pytz

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from car_parking.src.conf.constants import DEFAULT_TARIFF_ID, PARKING_ALREADY_CLOSED_DETAIL, PARKING_FULL_DETAIL, RESPONSE_DATETIME_FORMAT

from car_parking.src.database.models import Parking, ParkingCount, Tariff, User
from car_parking.src.schemas.parking import ParkingResponse, ParkingSchema

from car_parking.src.services.parking_calculations import calculate_parking_duration_hours, calculate_parking_cost
from car_parking.src.services.exceptions.parking_exceptions import (
    CarNotInParkingError, 
    ParkingAlreadyClosedError, 
    ParkingError, 
    ParkingFullError, 
    ParkingPlaceNotFoundError
)


def _format_datetime_for_response(value: datetime | None) -> str | None:
    if value is None:
        return None

    return value.strftime(RESPONSE_DATETIME_FORMAT)


def _build_parking_schema(
    parking_place: Parking,
    message: str,
    *,
    response_status: bool = False,
    format_departure_time: bool = False,
) -> ParkingSchema:
    departure_time = parking_place.departure_time

    if format_departure_time:
        departure_time = _format_datetime_for_response(departure_time)

    return ParkingSchema(
        info=ParkingResponse(
            id=parking_place.id,
            enter_time=_format_datetime_for_response(parking_place.enter_time),
            departure_time=departure_time,
            license_plate=parking_place.license_plate,
            amount_paid=parking_place.amount_paid,
            duration=parking_place.duration,
            status=response_status,
        ),
        status=message,
    )


def _get_tariff_for_user(user: User | None, db: Session) -> Tariff:
    tariff_id = user.tariff_id if user else DEFAULT_TARIFF_ID

    tariff = db.query(Tariff).filter_by(id=tariff_id).first()

    if tariff is None:
        tariff = db.query(Tariff).filter_by(id=DEFAULT_TARIFF_ID).first()

    if tariff is None:
        raise ValueError("Default parking tariff is missing")

    return tariff


def _apply_invoice_to_parking_place(
    parking_place: Parking,
    user: User | None,
    db: Session,
) -> Parking:
    departure_time = datetime.now(pytz.timezone("Europe/Kiev"))
    duration = calculate_parking_duration_hours(
        parking_place.enter_time,
        departure_time,
    )

    tariff = _get_tariff_for_user(user, db)

    parking_place.departure_time = departure_time
    parking_place.duration = duration
    parking_place.amount_paid = calculate_parking_cost(
        duration,
        tariff.tariff_value,
    )

    return parking_place

def _get_parking_count(db: Session) -> ParkingCount:
    parking_count = db.query(ParkingCount).first()

    if parking_count is None:
        raise ValueError("Parking count row is missing")

    return parking_count


def _is_parking_full(parking_count: ParkingCount) -> bool:
    return parking_count.occupied_quantity >= parking_count.total_quantity


def _increase_occupied_count(parking_count: ParkingCount) -> None:
    if _is_parking_full(parking_count):
        raise ValueError("Parking is full")

    parking_count.occupied_quantity += 1


def _decrease_occupied_count(parking_count: ParkingCount) -> None:
    if parking_count.occupied_quantity <= 0:
        parking_count.occupied_quantity = 0
        return

    parking_count.occupied_quantity -= 1


def _get_parking_place_by_id(
    parking_place_id: int,
    db: Session,
) -> Parking | None:
    return db.query(Parking).filter(Parking.id == parking_place_id).first()


def _get_active_parking_place_by_license_plate(
    license_plate: str,
    db: Session,
) -> Parking | None:
    return (
        db.query(Parking)
        .filter(
            Parking.license_plate == license_plate,
            Parking.status.is_(False),
        )
        .first()
    )


def _get_user_by_license_plate(
    license_plate: str,
    db: Session,
) -> User | None:
    return db.query(User).filter(User.license_plate == license_plate).first()


def _format_route_datetime(value) -> str:
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d %H:%M:%S")

    return str(value)


def _normalize_datetime_to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = pytz.timezone("Europe/London").localize(value)

    return value.astimezone(timezone.utc)
                            

def _count_occupied_places_at(
    requested_at: datetime,
    db: Session,
) -> int:
    requested_at_utc = _normalize_datetime_to_utc(requested_at)

    return (
        db.query(Parking)
        .filter(
            Parking.enter_time <= requested_at_utc,
            (
                (Parking.departure_time.is_(None))
                | (Parking.departure_time > requested_at_utc)
            ),
        )
        .count()
    )


def _raise_for_parking_error(error: ParkingError) -> None:
    if isinstance(error, ParkingFullError):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        )

    if isinstance(error, ParkingAlreadyClosedError):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        )

    if isinstance(error, (ParkingPlaceNotFoundError, CarNotInParkingError)):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        )

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=str(error),
    )