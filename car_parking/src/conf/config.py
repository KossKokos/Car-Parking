from pathlib import Path

from pydantic import BaseSettings, EmailStr

ENV_FILE = Path(__file__).parent.parent.parent.parent / ".env"


class Settings(BaseSettings):
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 80
    APP_DEBUG: bool = False
    APP_RELOAD: bool = False

    # API SECRETS
    SECRET_KEY: str = "SECRET_KEY"
    ALGORITHM: str = "ALGORITHM"

    # POSTGRESQL URL
    SQLALCHEMY_DATABASE_URL: str = "SQLALCHEMY_DATABASE_URL"

    # Reserved for future Redis-backed app logic and /health/redis.
    REDIS_URL: str = "redis://localhost:6380/0"

    # EMAIL CONNECTION
    MAIL_USERNAME: str = "MAIL_USERNAME"
    MAIL_PASSWORD: str = "MAIL_PASSWORD"
    MAIL_FROM: EmailStr = "example@example.com"
    MAIL_PORT: int = 587
    MAIL_SERVER: str = "MAIL_SERVER"

    # DATABASE CHECKS
    REQUIRED_TABLES: list[str] = []

    class Config:
        env_file = ENV_FILE
        env_file_encoding = "utf-8"


settings = Settings()
