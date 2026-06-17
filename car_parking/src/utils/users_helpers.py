from datetime import datetime, timezone

import pytz
from sqlalchemy.orm import Session

from car_parking.src.conf.constants import RESPONSE_DATETIME_FORMAT, TIMEZONE
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
    """Return completed parking sessions for one normalized license plate."""
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
    """Return the active parking session for a license plate, if one exists."""
    return (
        db.query(Parking)
        .filter(
            Parking.license_plate == _normalize_license_plate(license_plate),
            Parking.status.is_(False),
        )
        .first()
    )


APP_TIMEZONE = pytz.timezone(TIMEZONE)


def _format_datetime_for_response(value: datetime | None) -> str | None:
    if value is None:
        return None

    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        value = value.replace(tzinfo=timezone.utc)

    value = value.astimezone(APP_TIMEZONE)
    return value.strftime(RESPONSE_DATETIME_FORMAT)


async def calculate_amount_cost(list_of_parking: list[Parking]):
    return sum((park.amount_paid or 0) for park in list_of_parking)


async def calculate_amount_duration(list_of_parking: list[Parking]):
    return sum((park.duration or 0) for park in list_of_parking)
