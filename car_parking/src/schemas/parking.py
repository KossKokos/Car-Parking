from datetime import datetime
from typing import TypeAlias, TypedDict

from pydantic import BaseModel, Field


class ParkingResponse(BaseModel):
    enter_time: datetime
    departure_time: datetime | None
    license_plate: str
    amount_paid: float | None
    duration: float | None
    status: bool


class ParkingInfo(BaseModel):
    user: str
    total_payment_amount: float
    total_parking_time: float
    parking_info: list[ParkingResponse] = Field(default_factory=list)


class CurrentParking(BaseModel):
    enter_time: datetime
    time_on_parking: float
    parking_cost: float


class ParkingSchema(BaseModel):
    info: ParkingResponse
    status: str


class ParkingAvailabilityResponse(BaseModel):
    requested_at: str
    timezone: str
    total_places: int
    occupied_places: int
    free_places: int


ParkingOperationResult: TypeAlias = ParkingSchema
LegacyFreePlacesResult: TypeAlias = int | str


class ParkingAvailabilityData(TypedDict):
    requested_at: str
    timezone: str
    total_places: int
    occupied_places: int
    free_places: int


class CurrentParkingAvailabilityData(TypedDict):
    total_places: int
    occupied_places: int
    free_places: int
    stored_occupied_quantity: int