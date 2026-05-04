from fastapi import Depends, HTTPException, status

from car_parking.src.database.models import User
from car_parking.src.services.auth import service_auth


class BannedDependency:
    async def __call__(
        self,
        current_user: User = Depends(service_auth.get_current_user),
    ) -> None:
        if current_user.banned is True:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"User {current_user.email} banned. "
                    "Please contact your administrator!"
                ),
            )


banned_dependency = BannedDependency()