import uvicorn
from fastapi import FastAPI

from car_parking.src.conf.config import settings
from car_parking.src.database.db import get_db_session
from car_parking.src.repository import parking as repository_parking
from car_parking.src.repository import tariff as repository_tariff
from car_parking.src.routes import admin, auth, health, parking, users


def register_routes(app: FastAPI) -> None:
    """Attach all application routers with their public API prefixes."""
    app.include_router(health.router)
    app.include_router(auth.router, prefix="/api")
    app.include_router(users.router, prefix="/api")
    app.include_router(parking.router, prefix="/api")
    app.include_router(admin.router, prefix="/api")


def create_app() -> FastAPI:
    """Build the FastAPI application and register startup tasks."""
    app = FastAPI(debug=settings.APP_DEBUG)
    register_routes(app)

    @app.on_event("startup")
    async def seed_app_data() -> None:
        """Seed required lookup rows before the app accepts requests."""
        await seed_initial_data()

    return app


app = create_app()


async def seed_initial_data() -> None:
    """Populate required tariff and parking-count rows if they are missing."""
    with get_db_session() as db:
        await repository_tariff.seed_tariff_table(db)
        await repository_parking.seed_parking_count(db)


def main() -> None:
    """Run the ASGI server from the project entrypoint."""
    uvicorn.run(
        "main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.APP_RELOAD,
    )


if __name__ == "__main__":
    main()
