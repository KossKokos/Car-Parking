import json

from pathlib import Path
from typing import List

from pydantic import BaseSettings, EmailStr

ENV_FILE = Path(__file__).parent.parent.parent.parent / ".env"

class Settings(BaseSettings):
    
    # API SECRETS
    SECRET_KEY: str = "SECRET_KEY"
    ALGORITHM: str = "ALGORITHM"

    # POSTGRESQL URL
    SQLALCHEMY_DATABASE_URL: str = "SQLALCHEMY_DATABASE_URL"

    # REDIS CONNECTION
    REDIS_DB: int = 0
    REDIS_USER: str = "REDIS_USER"
    REDIS_PASSWORD: str = "REDIS_PASSWORD"
    REDIS_HOST: str = "REDIS_HOST"
    REDIS_PORT: int = 6380

    # REDIS URL
    REDIS_URL: str = "redis://localhost:6380/0"

    # EMAIL CONNECTION
    MAIL_USERNAME: str = "MAIL_USERNAME"
    MAIL_PASSWORD: str = "MAIL_PASSWORD"
    MAIL_FROM: EmailStr = "example@example.com"
    MAIL_PORT: int = 587
    MAIL_SERVER: str = "MAIL_SERVER"

    # CLOUDINARY CONNECTION
    CLOUDINARY_NAME: str = "CLOUDINARY_NAME"
    CLOUDINARY_API_KEY: str = "CLOUDINARY_API_KEY"
    CLOUDINARY_API_SECRET: str = "CLOUDINARY_API_SECRET"

    # DATABASE CHECKS
    REQUIRED_TABLES: List[str] = []

    # ROBOFLOW  
    ROBOFLOW_API_KEY: str = ""
    ROBOFLOW_PLATE_DETECT_URL: str = "" 
    ROBOFLOW_CONFIDENCE: int = 0
    ROBOFLOW_OVERLAP: int = 0

    class Config:
        env_file = ENV_FILE
        env_file_encoding = "utf-8"

    # classmethod to parse varibles like REQUIRED_TABLES from json to python format
    @classmethod
    def parse_json_list(cls, value):
        if isinstance(value, str):
            return json.loads(value)
        return value

settings = Settings()
