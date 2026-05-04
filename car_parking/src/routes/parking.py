import io

import numpy as np
from PIL import Image
from sqlalchemy.orm import Session

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)

from car_parking.src.services.exceptions.parking_exceptions import ParkingError

from ..conf.constants import LICENSE_PLATE_NOT_FOUND_DETAIL, LICENSE_PLATE_NOT_FOUND_MESSAGE

from ..database.db import get_db
from ..schemas.parking import ParkingAvailabilityResponse, ParkingSchema

from ..repository import car as repository_car
from ..repository import parking as repository_parking
from ..repository import tariff as repository_tariff
from ..repository import users as repository_users
from ..utils.parking_helpers import _format_route_datetime, _raise_for_parking_error

from ..services import email as service_email
from ..services.plate_reader import pr as PlateReader

router = APIRouter(prefix="/parking", tags=["parking"])


async def _read_uploaded_image_as_numpy(file: UploadFile) -> np.ndarray:
    valid_ext = await repository_parking.is_valid_file_ext(file)

    if not valid_ext:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file extension",
        )

    file_content = await file.read()

    try:
        image = Image.open(io.BytesIO(file_content))
        return np.array(image, dtype="uint8")

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image file",
        ) from exc


async def _detect_license_plate_from_upload(file: UploadFile) -> str | None:
    image = await _read_uploaded_image_as_numpy(file)
    return await PlateReader.get_prediction(image)


async def _get_banned_car_message_or_none(license_plate: str, db: Session) -> str | None:
    car = await repository_car.get_car_by_license_plate(license_plate, db)

    if car and car.banned is True:
        return f"Your car << {car.license_plate} >> banned. Contact parking administrator"

    return None


async def _schedule_parking_enter_email(
    *,
    background_tasks: BackgroundTasks,
    request: Request,
    parking_place: ParkingSchema,
    license_plate: str,
    db: Session,
) -> None:
    user = await repository_users.get_user_by_car_license_plate(license_plate, db)

    if not user:
        return

    tariff = await repository_tariff.get_tariff_by_tariff_id(user.tariff_id, db)
    enter_time = _format_route_datetime(parking_place.info.enter_time)

    background_tasks.add_task(
        service_email.parking_enter_message,
        user.email,
        user.username,
        user.license_plate,
        enter_time,
        tariff.tariff_name,
        tariff.tariff_value,
        request.base_url,
    )


async def _schedule_parking_exit_email(
    *,
    background_tasks: BackgroundTasks,
    request: Request,
    parking_info: ParkingSchema,
    parking_place_id: int,
    license_plate: str,
    db: Session,
) -> None:
    user = await repository_users.get_user_by_car_license_plate(license_plate, db)

    if not user:
        return

    tariff = await repository_tariff.get_tariff_by_tariff_id(user.tariff_id, db)

    enter_time = _format_route_datetime(parking_info.info.enter_time)
    departure_time = _format_route_datetime(parking_info.info.departure_time)

    background_tasks.add_task(
        service_email.parking_exit_message,
        user.email,
        user.username,
        user.license_plate,
        parking_place_id,
        enter_time,
        departure_time,
        tariff.tariff_name,
        tariff.tariff_value,
        parking_info.info.duration,
        parking_info.info.amount_paid,
        request.base_url,
    )


async def _handle_enter_parking(
    background_tasks: BackgroundTasks,
    request: Request,
    file: UploadFile,
    db: Session,
):
    license_plate = await _detect_license_plate_from_upload(file)

    if license_plate is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=LICENSE_PLATE_NOT_FOUND_DETAIL,
        )

    banned_message = await _get_banned_car_message_or_none(license_plate, db)

    if banned_message:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=banned_message,
        )

    try:
        parking_place = await repository_parking.entry_to_the_parking(license_plate, db)
    except ParkingError as error:
        _raise_for_parking_error(error)
    
    await _schedule_parking_enter_email(
        background_tasks=background_tasks,
        request=request,
        parking_place=parking_place,
        license_plate=license_plate,
        db=db,
    )

    return parking_place


async def _handle_exit_parking(
    background_tasks: BackgroundTasks,
    request: Request,
    file: UploadFile,
    db: Session,
):
    license_plate = await _detect_license_plate_from_upload(file)

    if license_plate is None:
        return LICENSE_PLATE_NOT_FOUND_MESSAGE

    banned_message = await _get_banned_car_message_or_none(license_plate, db)

    if banned_message:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=banned_message,
        )

    parking_place = await repository_parking.get_parking_place_by_car_license_plate(
        license_plate,
        db,
    )

    if not parking_place:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Parking place for car {license_plate} not found",
        )

    try:
        parking_info = await repository_parking.exit_from_the_parking(license_plate, db)
    except ParkingError as error:
        _raise_for_parking_error(error)

    await _schedule_parking_exit_email(
        background_tasks=background_tasks,
        request=request,
        parking_info=parking_info,
        parking_place_id=parking_place.id,
        license_plate=license_plate,
        db=db,
    )

    return parking_info
############################################################################################################################
############################################################################################################################
############################################################################################################################
@router.post(
    "/parking/{license_plate}",
    response_model=ParkingSchema,
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
)
async def enter_parking_legacy(
    license_plate: str,
    background_tasks: BackgroundTasks,
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    return await _handle_enter_parking(
        background_tasks=background_tasks,
        request=request,
        file=file,
        db=db,
    )

@router.post(
    "/enter",
    response_model=ParkingSchema,
    status_code=status.HTTP_200_OK,
)
async def enter_parking(
    background_tasks: BackgroundTasks,
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    return await _handle_enter_parking(
        background_tasks=background_tasks,
        request=request,
        file=file,
        db=db,
    )


@router.post(
    "/exit_parking/{license_plate}",
    response_model=ParkingSchema,
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
)
async def exit_parking_legacy(
    license_plate: str,
    background_tasks: BackgroundTasks,
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    return await _handle_exit_parking(
        background_tasks=background_tasks,
        request=request,
        file=file,
        db=db,
    )


@router.post(
    "/exit",
    response_model=ParkingSchema,
    status_code=status.HTTP_200_OK,
)
async def exit_parking(
    background_tasks: BackgroundTasks,
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    return await _handle_exit_parking(
        background_tasks=background_tasks,
        request=request,
        file=file,
        db=db,
    )


@router.get(
    "/confirm_payment/{parking_place_id}",
    response_model=ParkingSchema,
    status_code=status.HTTP_202_ACCEPTED,
)
async def confirm_payment(
    parking_place_id: int,
    db: Session = Depends(get_db),
):
    try:
        return await repository_parking.confirm_authorised_payment(
            parking_place_id,
            db,
        )
    except ParkingError as error:
        _raise_for_parking_error(error)


@router.get(
    "/availability",
    response_model=ParkingAvailabilityResponse,
    status_code=status.HTTP_200_OK,
)
async def get_parking_availability(
    at: str = Query(
        ...,
        description="Date/time in format YYYY.MM.DD HH:MM, for example 2026.04.30 14:30",
    ),
    db: Session = Depends(get_db),
):
    requested_at = repository_parking._parse_parking_availability_datetime(at)

    return await repository_parking.get_parking_availability_at(
        requested_at,
        db,
    )


@router.get(
    "/availability/current",
    status_code=status.HTTP_200_OK,
)
async def get_current_parking_availability(db: Session = Depends(get_db)):
    return await repository_parking.get_current_parking_availability(db)