import csv
from io import StringIO

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from car_parking.src.database.models import Tariff, User
from car_parking.src.repository import users as repository_users
from car_parking.src.schemas.users import UserRoleUpdate
from car_parking.src.services.csv_generator import build_csv_download_filename


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
) -> tuple[str, str]:
    """Build a downloadable CSV table for one car's completed parking history."""
    normalized_license_plate = license_plate.upper()
    download_filename = build_csv_download_filename(filename)

    parking_history = await repository_users.get_parking_info(
        normalized_license_plate,
        db,
    )

    fieldnames = [
        "Name",
        "Total Payment Amount (GBP)",
        "Total Parking Time (hours)",
        "Entry Time",
        "Departure Time",
        "License Plate",
        "Amount Paid (GBP)",
        "Duration (hours)",
        "Status",
    ]

    csvfile = StringIO(newline="")
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    summary_values = {
        "Name": parking_history.user,
        "Total Payment Amount (GBP)": (
            f"{parking_history.total_payment_amount:.2f}"
        ),
        "Total Parking Time (hours)": (
            f"{parking_history.total_parking_time:.2f}"
        ),
    }

    if not parking_history.parking_info:
        writer.writerow(
            {
                **summary_values,
                "License Plate": normalized_license_plate,
                "Status": "No completed parking sessions found",
            }
        )

    for parking_info in parking_history.parking_info:
        writer.writerow(
            {
                **summary_values,
                "Entry Time": parking_info.enter_time,
                "Departure Time": parking_info.departure_time,
                "License Plate": parking_info.license_plate,
                "Amount Paid (GBP)": (
                    f"{parking_info.amount_paid:.2f}"
                    if parking_info.amount_paid is not None
                    else ""
                ),
                "Duration (hours)": (
                    f"{parking_info.duration:.2f}"
                    if parking_info.duration is not None
                    else ""
                ),
                "Status": "Closed" if parking_info.status else "Open",
            }
        )

    return csvfile.getvalue(), download_filename


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
