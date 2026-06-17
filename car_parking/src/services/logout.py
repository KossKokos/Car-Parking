from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from car_parking.src.database.db import get_db
from car_parking.src.database.models import User
from car_parking.src.repository.logout import get_blacklisted_token_by_user_id
from car_parking.src.services.auth import service_auth


security = HTTPBearer()


class LogoutDependency:
    async def __call__(
        self,
        current_user: User = Depends(service_auth.get_current_user),
        credentials: HTTPAuthorizationCredentials = Security(security),
        db: Session = Depends(get_db),
    ) -> None:
        """Reject requests that reuse the current user's blacklisted access token."""
        blacklisted_token = await get_blacklisted_token_by_user_id(
            current_user.id,
            db,
        )

        access_token = credentials.credentials

        if (
            blacklisted_token is not None
            and blacklisted_token.blacklisted_token == access_token
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Operation forbidden for {current_user.email}. "
                    "Please login again!"
                ),
            )


logout_dependency = LogoutDependency()
