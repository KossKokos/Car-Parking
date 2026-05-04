from pydantic import BaseModel

from car_parking.src.database.models import Car


class CarResponse(BaseModel):
    id: int
    license_plate: str
    banned: bool

    @staticmethod
    def from_db_model(db_model: Car):
        return CarResponse(
            id=db_model.id,
            license_plate=db_model.license_plate,
            banned=db_model.banned,
        )

    class Config:
        orm_mode = True
        schema_extra = {
            "example": {"id": 1, "license_plate": "xx0000yy", "banned": "False"}
        }
