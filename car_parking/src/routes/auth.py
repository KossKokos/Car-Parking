from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
    Security,
    BackgroundTasks,
    Request,
)
from fastapi.security import (
    OAuth2PasswordRequestForm,
    HTTPAuthorizationCredentials,
    HTTPBearer,
)
from sqlalchemy.orm import Session

from car_parking.src.database.db import get_db
from car_parking.src.database.models import User
from car_parking.src.repository import users as repository_users
from car_parking.src.repository import car as repository_car
from car_parking.src.repository.logout import blacklist_access_token_for_user
from car_parking.src.services.auth import service_auth
from car_parking.src.services import (
    email as service_email,
    roles as service_roles,
    banned as service_banned,
    logout as service_logout,
)
from car_parking.src.schemas import (
    users as schema_users,
    token as schema_token,
    email as schema_email,
)


router = APIRouter(prefix="/auth", tags=["auth"])
security = HTTPBearer()

allowed_admin = service_roles.RoleRights(["admin"])
allowed_user_or_admin = service_roles.RoleRights(["user", "admin"])


async def _ensure_signup_data_is_unique(
    body: schema_users.UserModel,
    db: Session,
) -> None:
    existing_user_with_email = await repository_users.get_user_by_email(
        body.email,
        db,
    )

    if existing_user_with_email:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"User with email: {body.email} already exists",
        )

    existing_user_with_username = await repository_users.get_user_by_username(
        body.username,
        db,
    )

    if existing_user_with_username:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"User with name: {body.username} already exists",
        )

    existing_user_with_license_plate = (
        await repository_users.get_user_by_car_license_plate(
            body.license_plate,
            db,
        )
    )

    if existing_user_with_license_plate:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"User with license plate: {body.license_plate} already exists",
        )


async def _ensure_car_exists(license_plate: str, db: Session) -> None:
    normalized_license_plate = license_plate.upper()

    car = await repository_car.get_car_by_license_plate(
        normalized_license_plate,
        db,
    )

    if car is None:
        await repository_car.create_car(normalized_license_plate, db)


def _schedule_verification_email(
    *,
    background_tasks: BackgroundTasks,
    request: Request,
    user: User,
) -> None:
    background_tasks.add_task(
        service_email.send_email,
        user.email,
        user.username,
        request.base_url,
    )


async def _authenticate_login_user(
    *,
    email: str,
    password: str,
    db: Session,
) -> User:
    user = await repository_users.get_user_by_email(email, db)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email",
        )

    if user.banned is True:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"User {user.email} banned. Please contact your administrator!",
        )

    if not user.confirmed:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email is not confirmed",
        )

    if not service_auth.verify_password(password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid password",
        )

    return user


async def _create_token_response(
    *,
    user: User,
    db: Session,
) -> dict[str, str]:
    access_token = await service_auth.create_access_token(
        data={"sub": user.email},
    )
    refresh_token = await service_auth.create_refresh_token(
        data={"sub": user.email},
    )

    await repository_users.update_token(user, refresh_token, db)

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
    }


async def _validate_refresh_token_user(
    *,
    refresh_token: str,
    db: Session,
) -> User:
    email = await service_auth.decode_refresh_token(refresh_token)
    user = await repository_users.get_user_by_email(email, db)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User with this token doesn't exist",
        )

    if user.refresh_token != refresh_token:
        user.refresh_token = None
        db.commit()

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )

    if user.banned is True:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"User {user.email} banned. Please contact your administrator!",
        )

    return user


async def _get_user_from_email_token(
    *,
    token: str,
    db: Session,
) -> User:
    email = await service_auth.decode_email_token(token)
    user = await repository_users.get_user_by_email(email, db)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Verification error",
        )

    return user


def _schedule_confirmation_email(
    *,
    background_tasks: BackgroundTasks,
    request: Request,
    user: User,
) -> None:
    background_tasks.add_task(
        service_email.send_email,
        user.email,
        user.username,
        request.base_url,
    )


def _schedule_reset_password_email(
    *,
    background_tasks: BackgroundTasks,
    request: Request,
    user: User,
) -> None:
    background_tasks.add_task(
        service_email.send_reset_password_email,
        user.email,
        user.username,
        request.base_url,
    )
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
@router.post("/signup", status_code=status.HTTP_201_CREATED)
async def signup(
    body: schema_users.UserModel,
    background_tasks: BackgroundTasks,
    request: Request,
    db: Session = Depends(get_db),
):
    await _ensure_signup_data_is_unique(body, db)

    body.password = service_auth.get_password_hash(body.password)

    await _ensure_car_exists(body.license_plate, db)

    user = await repository_users.create_user(body, db)

    _schedule_verification_email(
        background_tasks=background_tasks,
        request=request,
        user=user,
    )

    return {
        "user": user,
        "detail": (
            f"User successfully created, please check your email "
            f"<< {user.email} >> for verification"
        ),
    }


@router.post(
    "/login",
    response_model=schema_token.TokenResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def login(
    body: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    user = await _authenticate_login_user(
        email=body.username,
        password=body.password,
        db=db,
    )

    return await _create_token_response(
        user=user,
        db=db,
    )


@router.get(
    "/refresh_token",
    response_model=schema_token.TokenResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def refresh_token(
    credentials: HTTPAuthorizationCredentials = Security(security),
    db: Session = Depends(get_db),
):
    provided_refresh_token = credentials.credentials

    user = await _validate_refresh_token_user(
        refresh_token=provided_refresh_token,
        db=db,
    )

    return await _create_token_response(
        user=user,
        db=db,
    )


@router.get("/confirmed_email/{token}", status_code=status.HTTP_202_ACCEPTED)
async def confirm_email(
    token: str,
    db: Session = Depends(get_db),
):
    user = await _get_user_from_email_token(
        token=token,
        db=db,
    )

    if user.confirmed:
        return {"detail": "Email is already confirmed"}

    await repository_users.confirmed_email(user.email, db)

    return {"detail": "Email is confirmed"}


@router.post("/request_email", status_code=status.HTTP_202_ACCEPTED)
async def request_email(
    body: schema_email.RequestEmail,
    background_tasks: BackgroundTasks,
    request: Request,
    db: Session = Depends(get_db),
):
    user = await repository_users.get_user_by_email(body.email, db)

    if user is not None and user.confirmed:
        return {"detail": "Email is already confirmed"}

    if user is not None:
        _schedule_confirmation_email(
            background_tasks=background_tasks,
            request=request,
            user=user,
        )

    return {"detail": "Check your email for further information"}


@router.post("/reset_password", status_code=status.HTTP_202_ACCEPTED)
async def reset_password_request(
    body: schema_email.RequestEmail,
    background_tasks: BackgroundTasks,
    request: Request,
    db: Session = Depends(get_db),
):
    user = await repository_users.get_user_by_email(body.email, db)

    if user is not None:
        _schedule_reset_password_email(
            background_tasks=background_tasks,
            request=request,
            user=user,
        )

    return {"detail": "Check your email for further information"}


@router.patch("/change_password/{token}", status_code=status.HTTP_202_ACCEPTED)
async def reset_password(
    body: schema_users.ChangePassword,
    token: str,
    db: Session = Depends(get_db),
):
    user = await _get_user_from_email_token(
        token=token,
        db=db,
    )

    hashed_password = service_auth.get_password_hash(body.new_password)

    await repository_users.change_password(
        user,
        hashed_password,
        db,
    )

    return {"detail": "User's password was changed successfully"}



@router.get(
    "/logout",
    status_code=status.HTTP_200_OK,
    dependencies=[
        Depends(service_logout.logout_dependency),
        Depends(allowed_user_or_admin),
    ],
)
async def logout(
    credentials: HTTPAuthorizationCredentials = Security(security),
    db: Session = Depends(get_db),
    current_user: User = Depends(service_auth.get_current_user),
):
    access_token = credentials.credentials

    await blacklist_access_token_for_user(
        access_token=access_token,
        user_id=current_user.id,
        db=db,
    )

    return {"message": f"User {current_user.email} successfully logged out"}
