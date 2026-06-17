from datetime import datetime, timezone
from decimal import Decimal
from typing import Union


Number = Union[int, float, Decimal]


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        return value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc)


def calculate_parking_duration_hours(start_time: datetime, end_time: datetime) -> float:
    """
    Calculate parking duration in hours.
    - returns hours as float
    - rounds to 2 decimal places
    """
    time_difference = _as_utc(end_time) - _as_utc(start_time)
    hours = time_difference.total_seconds() / 3600
    return round(float(hours), 2)


def calculate_parking_cost(hours: float, hourly_rate: float) -> float:
    """
    Calculate parking cost.
    - multiplies duration by tariff value
    - returns rounded float
    """
    result = float(hours) * float(hourly_rate)
    return round(result, 2)
