import os

# To hide TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import uvicorn
from fastapi import FastAPI

from car_parking.src.database.db import get_db_session
from car_parking.src.repository import parking as repository_parking
from car_parking.src.repository import tariff as repository_tariff
from car_parking.src.routes import admin, auth, health, parking, users


def register_routes(app: FastAPI) -> None:
    app.include_router(health.router)
    app.include_router(auth.router, prefix="/api")
    app.include_router(users.router, prefix="/api")
    app.include_router(parking.router, prefix="/api")
    app.include_router(admin.router, prefix="/api")


def create_app() -> FastAPI:
    app = FastAPI(debug=True)
    register_routes(app)
    return app


app = create_app()


async def seed_initial_data() -> None:
    with get_db_session() as db:
        await repository_tariff.seed_tariff_table(db)
        await repository_parking.seed_parking_count(db)


async def main() -> None:
    await seed_initial_data()
    uvicorn.run("main:app", host="0.0.0.0", port=80, reload=True)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())