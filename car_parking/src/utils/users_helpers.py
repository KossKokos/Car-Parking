from datetime import datetime

from sqlalchemy.orm import Session

from car_parking.src.conf.constants import RESPONSE_DATETIME_FORMAT
from car_parking.src.database.models import Car, Parking


def _normalize_license_plate(license_plate: str) -> str:
    return license_plate.upper()


def _get_car_by_license_plate(license_plate: str, db: Session) -> Car | None:
    return db.query(Car).filter(
        Car.license_plate == _normalize_license_plate(license_plate)
    ).first()


def _get_closed_parking_sessions_by_license_plate(
    license_plate: str,
    db: Session,
) -> list[Parking]:
    return (
        db.query(Parking)
        .filter(
            Parking.license_plate == _normalize_license_plate(license_plate),
            Parking.status.is_(True),
        )
        .all()
    )


def _get_current_parking_session_by_license_plate(
    license_plate: str,
    db: Session,
) -> Parking | None:
    return (
        db.query(Parking)
        .filter(
            Parking.license_plate == _normalize_license_plate(license_plate),
            Parking.status.is_(False),
        )
        .first()
    )


def _format_datetime_for_response(value: datetime) -> str:
    return value.strftime(RESPONSE_DATETIME_FORMAT)


async def calculate_amount_cost(list_of_parking: list[Parking]):
    return sum((park.amount_paid or 0) for park in list_of_parking)


async def calculate_amount_duration(list_of_parking: list[Parking]):
    return sum((park.duration or 0) for park in list_of_parking)
