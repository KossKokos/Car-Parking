import csv

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from car_parking.src.database.models import Tariff, User
from car_parking.src.repository import users as repository_users
from car_parking.src.schemas.users import UserRoleUpdate
from car_parking.src.services.csv_generator import build_csv_file_path


async def get_tariff_by_name(
    tariff_name: str,
    db: Session,
) -> Tariff | None:
    return (
        db.query(Tariff)
        .filter(Tariff.tariff_name == tariff_name.upper())
        .first()
    )


async def change_user_role(
    user: User,
    body: UserRoleUpdate,
    db: Session,
) -> User:
    user.role = body.role

    db.commit()
    db.refresh(user)

    return user


async def return_all_users(db: Session) -> dict[str, str]:
    users: list = db.query(User).all()

    return {
        f"username(id: {user.id})": user.username
        for user in users
    }


async def get_all_users(db: Session) -> list[User]:
    return db.query(User).all()


async def set_user_banned_status(
    user: User,
    is_banned: bool,
    db: Session,
) -> User:
    user.banned = is_banned

    db.commit()
    db.refresh(user)

    return user


async def create_parking_csv(
    license_plate: str,
    filename: str,
    db: Session,
) -> str:
    normalized_license_plate = license_plate.upper()
    file_path = build_csv_file_path(filename)

    parking_history = await repository_users.get_parking_info(
        normalized_license_plate,
        db,
    )

    with file_path.open("w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "Name",
            "Total Payment Amount",
            "Total Parking Time",
            "enter_time",
            "departure_time",
            "license_plate",
            "amount_paid",
            "duration",
            "status",
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for parking_info in parking_history.parking_info:
            writer.writerow(
                {
                    "enter_time": parking_info.enter_time,
                    "departure_time": parking_info.departure_time,
                    "license_plate": parking_info.license_plate,
                    "amount_paid": parking_info.amount_paid,
                    "duration": parking_info.duration,
                    "status": parking_info.status,
                }
            )

        writer.writerow({"Name": parking_history.user})
        writer.writerow({"Total Payment Amount": parking_history.total_payment_amount})
        writer.writerow({"Total Parking Time": parking_history.total_parking_time})

    return f"CSV file created: {file_path.name}"


async def change_tariff(
    user_id: int,
    new_tariff: str,
    db: Session,
) -> User:
    user = await repository_users.get_user_by_id(
        user_id=user_id,
        db=db,
    )

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    tariff = await get_tariff_by_name(new_tariff, db)

    if tariff is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tariff not found",
        )

    user.tariff_id = tariff.id

    db.commit()
    db.refresh(user)

    return user


async def add_tariff(
    tariff_name: str,
    tariff_cost: int,
    db: Session,
) -> Tariff:
    normalized_tariff_name = tariff_name.upper()

    existing_tariff = await get_tariff_by_name(
        normalized_tariff_name,
        db,
    )

    if existing_tariff is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Tariff {normalized_tariff_name} already exists",
        )

    new_tariff = Tariff(
        tariff_name=normalized_tariff_name,
        tariff_value=tariff_cost,
    )

    db.add(new_tariff)
    db.commit()
    db.refresh(new_tariff)

    return new_tariff