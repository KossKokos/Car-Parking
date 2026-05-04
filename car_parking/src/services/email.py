import logging
from pathlib import Path
from typing import Any

from fastapi_mail import ConnectionConfig, FastMail, MessageSchema, MessageType
from fastapi_mail.errors import ConnectionErrors
from pydantic import EmailStr

from car_parking.src.conf.config import settings
from car_parking.src.conf import constants
from car_parking.src.services.auth import service_auth


logger = logging.getLogger(__name__)

TEMPLATE_FOLDER = Path(__file__).parent.parent / "templates"

conf = ConnectionConfig(
    MAIL_USERNAME=settings.MAIL_USERNAME,
    MAIL_PASSWORD=settings.MAIL_PASSWORD,
    MAIL_FROM=EmailStr(settings.MAIL_FROM),
    MAIL_PORT=settings.MAIL_PORT,
    MAIL_SERVER=settings.MAIL_SERVER,
    MAIL_FROM_NAME=constants.MAIL_FROM_NAME,
    MAIL_STARTTLS=False,
    MAIL_SSL_TLS=True,
    USE_CREDENTIALS=True,
    VALIDATE_CERTS=True,
    TEMPLATE_FOLDER=TEMPLATE_FOLDER,
)


async def _send_template_email(
    *,
    subject: str,
    recipient: EmailStr,
    template_name: str,
    template_body: dict[str, Any],
) -> None:
    message = MessageSchema(
        subject=subject,
        recipients=[recipient],
        template_body=template_body,
        subtype=MessageType.html,
    )

    try:
        mail_client = FastMail(conf)
        await mail_client.send_message(
            message,
            template_name=template_name,
        )

    except ConnectionErrors as exc:
        logger.exception("Failed to send email to %s", recipient)
        raise exc


async def _create_email_token(email: EmailStr) -> str:
    return await service_auth.create_email_token({"sub": email})


async def send_email(email: EmailStr, username: str, host: str) -> None:
    token_verification = await _create_email_token(email)

    await _send_template_email(
        subject="Confirm your email",
        recipient=email,
        template_name=constants.EMAIL_CONFIRMATION_TEMPLATE,
        template_body={
            "host": host,
            "username": username,
            "token": token_verification,
        },
    )


async def send_reset_password_email(
    email: EmailStr,
    username: str,
    host: str,
) -> None:
    token_verification = await _create_email_token(email)

    await _send_template_email(
        subject="Reset password",
        recipient=email,
        template_name=constants.RESET_PASSWORD_TEMPLATE,
        template_body={
            "host": host,
            "username": username,
            "token": token_verification,
        },
    )


async def parking_enter_message(
    email: EmailStr,
    username: str,
    license_plate: str,
    enter_time: str,
    tariff_name: str,
    tariff_value: int | float | str,
    host: str,
) -> None:
    token_verification = await _create_email_token(email)

    await _send_template_email(
        subject="Parking place info",
        recipient=email,
        template_name=constants.PARKING_ENTER_TEMPLATE,
        template_body={
            "host": host,
            "username": username,
            "license_plate": license_plate,
            "enter_time": enter_time,
            "tariff_name": tariff_name,
            "tariff_value": tariff_value,
            "token": token_verification,
        },
    )


async def parking_exit_message(
    email: EmailStr,
    username: str,
    license_plate: str,
    parking_place_id: int,
    enter_time: str,
    departure_time: str,
    tariff_name: str,
    tariff_value: int | float | str,
    duration: int | float | str,
    amount_paid: int | float | str,
    host: str,
) -> None:
    token_verification = await _create_email_token(email)

    await _send_template_email(
        subject="Invoice for payment",
        recipient=email,
        template_name=constants.PARKING_EXIT_TEMPLATE,
        template_body={
            "host": host,
            "username": username,
            "license_plate": license_plate,
            "parking_place_id": parking_place_id,
            "enter_time": enter_time,
            "departure_time": departure_time,
            "tariff_name": tariff_name,
            "tariff_value": tariff_value,
            "duration": duration,
            "amount_paid": amount_paid,
            "token": token_verification,
        },
    )