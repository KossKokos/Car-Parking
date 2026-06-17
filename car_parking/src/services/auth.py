from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from car_parking.src.conf.config import settings
from car_parking.src.database.db import get_db
from car_parking.src.repository import users as repository_users

from car_parking.src.conf import constants


class Auth:
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    SECRET_KEY = settings.SECRET_KEY
    ALGORITHM = settings.ALGORITHM

    # Swagger uses this URL for the OAuth2 password login form.
    oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return self.pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        return self.pwd_context.hash(password)

    def _create_token(
        self,
        *,
        data: dict,
        scope: str,
        expires_delta: timedelta,
    ) -> str:
        """Create a signed JWT with a scope claim used to separate token types."""
        to_encode = data.copy()
        now = datetime.utcnow()
        expire = now + expires_delta

        to_encode.update(
            {
                "iat": now,
                "exp": expire,
                "scope": scope,
            }
        )

        return jwt.encode(
            to_encode,
            self.SECRET_KEY,
            algorithm=self.ALGORITHM,
        )

    async def create_access_token(
        self,
        data: dict,
        expires_delta: Optional[float] = None,
    ) -> str:
        """Create a bearer access token for authenticated API requests."""
        expire_delta = (
            timedelta(seconds=expires_delta)
            if expires_delta
            else timedelta(minutes=constants.DEFAULT_ACCESS_TOKEN_EXPIRE_MINUTES)
        )

        return self._create_token(
            data=data,
            scope=constants.ACCESS_TOKEN_SCOPE,
            expires_delta=expire_delta,
        )

    def sync_create_access_token(
        self,
        data: dict,
        expires_delta: Optional[float] = None,
    ) -> str:
        """Create an access token from synchronous code paths."""
        expire_delta = (
            timedelta(seconds=expires_delta)
            if expires_delta
            else timedelta(minutes=constants.DEFAULT_ACCESS_TOKEN_EXPIRE_MINUTES)
        )

        return self._create_token(
            data=data,
            scope=constants.ACCESS_TOKEN_SCOPE,
            expires_delta=expire_delta,
        )

    async def create_refresh_token(
        self,
        data: dict,
        expires_delta: Optional[float] = None,
    ) -> str:
        """Create a long-lived token that can be exchanged for new credentials."""
        expire_delta = (
            timedelta(seconds=expires_delta)
            if expires_delta
            else timedelta(days=constants.DEFAULT_REFRESH_TOKEN_EXPIRE_DAYS)
        )

        return self._create_token(
            data=data,
            scope=constants.REFRESH_TOKEN_SCOPE,
            expires_delta=expire_delta,
        )

    async def get_current_user(
        self,
        token: str = Depends(oauth2_scheme),
        db: Session = Depends(get_db),
    ):
        """Resolve the authenticated user from a valid access token."""
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

        try:
            payload = jwt.decode(
                token,
                self.SECRET_KEY,
                algorithms=[self.ALGORITHM],
            )

            if payload.get("scope") != constants.ACCESS_TOKEN_SCOPE:
                raise credentials_exception

            email = payload.get("sub")

            if email is None:
                raise credentials_exception

        except JWTError as exc:
            raise credentials_exception from exc

        user = await repository_users.get_user_by_email(email, db)

        if user is None:
            raise credentials_exception

        return user

    async def decode_refresh_token(self, refresh_token: str) -> str:
        """Validate a refresh token and return the email stored in its subject."""
        try:
            payload = jwt.decode(
                refresh_token,
                self.SECRET_KEY,
                algorithms=[self.ALGORITHM],
            )

            if payload.get("scope") != constants.REFRESH_TOKEN_SCOPE:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid scope for token",
                )

            email = payload.get("sub")

            if email is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Could not validate credentials",
                )

            return email

        except JWTError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
            ) from exc

    async def create_email_token(self, data: dict) -> str:
        """Create a scoped token for email confirmation and password reset links."""
        return self._create_token(
            data=data,
            scope=constants.EMAIL_TOKEN_SCOPE,
            expires_delta=timedelta(days=constants.DEFAULT_EMAIL_TOKEN_EXPIRE_DAYS),
        )

    def sync_create_email_token(self, data: dict) -> str:
        """Create an email token from synchronous code paths."""
        return self._create_token(
            data=data,
            scope=constants.EMAIL_TOKEN_SCOPE,
            expires_delta=timedelta(days=constants.DEFAULT_EMAIL_TOKEN_EXPIRE_DAYS),
        )

    async def decode_email_token(self, email_token: str) -> str:
        """Validate an email-scoped token and return the email subject."""
        try:
            payload = jwt.decode(
                email_token,
                self.SECRET_KEY,
                algorithms=[self.ALGORITHM],
            )

            if payload.get("scope") != constants.EMAIL_TOKEN_SCOPE:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid scope for email token",
                )

            email = payload.get("sub")

            if email is None:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Invalid token for email",
                )

            return email

        except JWTError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Invalid token for email",
            ) from exc


service_auth = Auth()
