from sqlalchemy.orm import Session

from car_parking.src.database.models import Tariff
from ..conf.constants import TARIFFS_DATA

async def get_tariff_by_tariff_id(tariff_id: int, db: Session) -> Tariff | None:
    tariff = db.query(Tariff).filter(Tariff.id == tariff_id).first()
    return tariff


async def seed_tariff_table(db: Session):
    if db.query(Tariff).count() == 0:
        
        for data in TARIFFS_DATA:
            tariff = Tariff(**data)
            db.add(tariff)

        db.commit()
