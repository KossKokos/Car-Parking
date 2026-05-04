from sqlalchemy.orm import Session

from car_parking.src.database.models import Car


def normalize_license_plate(license_plate: str) -> str:
    return license_plate.upper()


def build_car(license_plate: str) -> Car:
    return Car(license_plate=normalize_license_plate(license_plate))


async def create_car(license_plate: str, db: Session) -> Car:
    car = build_car(license_plate)

    db.add(car)
    db.commit()
    db.refresh(car)

    return car


async def get_car_by_license_plate(
    license_plate: str,
    db: Session,
) -> Car | None:
    return (
        db.query(Car)
        .filter(Car.license_plate == normalize_license_plate(license_plate))
        .first()
    )


async def set_car_banned_status(
    car: Car,
    is_banned: bool,
    db: Session,
) -> Car:
    car.banned = is_banned

    db.commit()
    db.refresh(car)

    return car