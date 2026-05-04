from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from ..conf.config import settings


SQLALCHEMY_DATABASE_URL = settings.SQLALCHEMY_DATABASE_URL

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI database dependency.

    Use this only in route dependencies:

        db: Session = Depends(get_db)
    """
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Manual database session context manager.

    Use this outside request/response flow, for example:
    - startup seed tasks
    - CLI scripts
    - maintenance scripts
    """
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


@contextmanager
def db_transaction(session: Session) -> Generator[Session, None, None]:
    """
    Transaction helper for an existing session.

    This function does not close the session because it does not own it.
    The caller that created the session is responsible for closing it.
    """
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise