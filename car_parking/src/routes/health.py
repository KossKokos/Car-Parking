import logging
import re
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import bindparam, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from redis import Redis
from redis.exceptions import RedisError

from car_parking.src.conf.config import settings
from car_parking.src.database.db import get_db


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])

TABLE_NAME_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str


class DatabaseHealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    checked_tables: list[str]


class RedisHealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    ping: Literal["pong"]
    read_write_check: bool


def get_redis_client() -> Redis:
    """Create a short-timeout Redis client for health checks."""
    return Redis.from_url(
        settings.REDIS_URL,
        decode_responses=True,
        socket_connect_timeout=2,
        socket_timeout=2,
    )


@router.get(
    "",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
)
async def app_health():
    """
    Basic application health check.

    This only confirms that the FastAPI application is running.
    It does not check PostgreSQL or Redis.
    """
    return {
        "status": "ok",
        "service": "car-parking-api",
    }


@router.get(
    "/db",
    response_model=DatabaseHealthResponse,
    status_code=status.HTTP_200_OK,
)
def db_health(db: Session = Depends(get_db)):
    """
    Database health check.

    Checks:
    - database connection is alive
    - required tables are configured
    - required tables exist
    - required tables are queryable
    """
    try:
        db.execute(text("SELECT 1")).scalar_one()

        required_tables = sorted(set(settings.REQUIRED_TABLES))

        if not required_tables:
            logger.error("Database health check failed: REQUIRED_TABLES is empty")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service unavailable",
            )

        invalid_tables = [
            table
            for table in required_tables
            if not TABLE_NAME_PATTERN.fullmatch(table)
        ]

        if invalid_tables:
            logger.error(
                "Database health check failed: invalid table names in REQUIRED_TABLES: %s",
                invalid_tables,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service unavailable",
            )

        stmt = text("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name IN :required_tables
        """).bindparams(bindparam("required_tables", expanding=True))

        rows = db.execute(
            stmt,
            {"required_tables": required_tables},
        ).fetchall()

        found_tables = sorted(row[0] for row in rows)
        missing_tables = sorted(set(required_tables) - set(found_tables))

        if missing_tables:
            logger.error(
                "Database health check failed: missing required tables: %s",
                missing_tables,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service unavailable",
            )

        for table in required_tables:
            db.execute(text(f'SELECT COUNT(*) FROM "{table}"')).scalar_one()

        return {
            "status": "ok",
            "service": "postgresql",
            "checked_tables": required_tables,
        }

    except HTTPException:
        raise

    except SQLAlchemyError as exc:
        logger.exception("Database health check failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unavailable",
        ) from exc
    

@router.get(
    "/redis",
    response_model=RedisHealthResponse,
    status_code=status.HTTP_200_OK,
)
def redis_health():
    """
    Redis health check.

    Checks:
    - Redis connection is alive
    - Redis responds to PING
    - Redis can write, read, and delete a temporary health-check key
    """
    client = get_redis_client()
    healthcheck_key = "healthcheck:redis"

    try:
        ping_result = client.ping()

        if ping_result is not True:
            logger.error("Redis health check failed: PING did not return True")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service unavailable",
            )

        client.set(healthcheck_key, "ok", ex=10)
        value = client.get(healthcheck_key)
        client.delete(healthcheck_key)

        if value != "ok":
            logger.error("Redis health check failed: read/write check failed")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service unavailable",
            )

        return {
            "status": "ok",
            "service": "redis",
            "ping": "pong",
            "read_write_check": True,
        }

    except HTTPException:
        raise

    except RedisError as exc:
        logger.exception("Redis health check failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unavailable",
        ) from exc

    finally:
        client.close()
