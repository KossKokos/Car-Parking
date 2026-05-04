from datetime import datetime
from decimal import Decimal
from typing import Union


Number = Union[int, float, Decimal]


def calculate_parking_duration_hours(start_time: datetime, end_time: datetime) -> float:
    """
    Calculate parking duration in hours.
    - returns hours as float
    - rounds to 2 decimal places
    """
    time_difference = end_time - start_time
    hours = time_difference.days * 24 + time_difference.seconds / 3600
    return round(float(hours), 2)


def calculate_parking_cost(hours: float, hourly_rate: float) -> float:
    """
    Calculate parking cost.
    - multiplies duration by tariff value
    - returns rounded float
    """
    result = float(hours) * float(hourly_rate)
    return round(result, 2)