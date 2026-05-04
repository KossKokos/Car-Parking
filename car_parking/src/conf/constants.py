EXTENSIONS = [
    "jpg", 
    "jpeg", 
    "webp", 
    "pdf", 
    "avif",
    "png"
]
# repository/parking
PARKING_COUNT_DATA = [
            {"total_quantity": 30, "occupied_quantity": 0},
        ]

TARIFFS_DATA = [
            {"tariff_name": "STANDART", "tariff_value": 30},
            {"tariff_name": "AUTORIZED", "tariff_value": 20},
            {"tariff_name": "MAX_LIMIT", "tariff_value": 1000},
        ]
RESPONSE_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_TARIFF_ID = 1

# repository/users
DEFAULT_USER_TARIFF_ID = 2
ADMIN_USER_ID = 1

# routes/parking
PARKING_AVAILABILITY_DATETIME_FORMAT = "%Y.%m.%d %H:%M"
TIMEZONE = "Europe/London"

# error handling
LICENSE_PLATE_NOT_FOUND_DETAIL = (
    "License plate not found. Please upload a clearer image where the car plate is visible."
)
LICENSE_PLATE_NOT_FOUND_MESSAGE = (
    "License plate not found, please send better picture where car is visible"
)
PARKING_FULL_DETAIL = "Sorry, there are no available parking places."
PARKING_PLACE_NOT_FOUND_DETAIL = "Parking place not found."
PARKING_ALREADY_CLOSED_DETAIL = "Parking place is already closed."

# services/auth.py
ACCESS_TOKEN_SCOPE = "access_token"
REFRESH_TOKEN_SCOPE = "refresh_token"
EMAIL_TOKEN_SCOPE = "email_token"

DEFAULT_ACCESS_TOKEN_EXPIRE_MINUTES = 60
DEFAULT_REFRESH_TOKEN_EXPIRE_DAYS = 7
DEFAULT_EMAIL_TOKEN_EXPIRE_DAYS = 7

# services/email.py
EMAIL_CONFIRMATION_TEMPLATE = "email_template.html"
RESET_PASSWORD_TEMPLATE = "reset_password.html"

# Keep these typo-based template names for now because the files are named this way.
PARKING_ENTER_TEMPLATE = "parking_enter_message.html"
PARKING_EXIT_TEMPLATE = "parking_exit_message.html"

MAIL_FROM_NAME = "Car parking"

# routes/auth.py
SUPERADMIN_USER_ID = 1