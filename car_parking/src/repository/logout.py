from sqlalchemy.orm import Session

from car_parking.src.database.models import BlacklistedToken, User


async def get_blacklisted_token_by_user_id(
    user_id: int,
    db: Session,
) -> BlacklistedToken | None:
    return (
        db.query(BlacklistedToken)
        .filter(BlacklistedToken.user_id == user_id)
        .first()
    )


async def blacklist_access_token_for_user(
    *,
    access_token: str,
    user_id: int,
    db: Session,
) -> BlacklistedToken:
    user = db.query(User).filter(User.id == user_id).first()

    if user is None:
        raise ValueError("User not found.")

    blacklisted_token = await get_blacklisted_token_by_user_id(
        user_id,
        db,
    )

    if blacklisted_token is None:
        blacklisted_token = BlacklistedToken(
            user_id=user_id,
            blacklisted_token=access_token,
        )
        db.add(blacklisted_token)
    else:
        blacklisted_token.blacklisted_token = access_token

    user.refresh_token = None

    db.commit()
    db.refresh(blacklisted_token)

    return blacklisted_token