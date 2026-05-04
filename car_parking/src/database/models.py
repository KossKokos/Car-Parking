from sqlalchemy import (
    Column,
    Integer,
    String,
    ForeignKey,
    Boolean,
    func,
    CheckConstraint,
    Numeric,
)
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql.sqltypes import DateTime


Base = declarative_base()


class User(Base):
    __tablename__ = "users_table"

    id = Column(Integer, primary_key=True)
    username = Column(String(50), nullable=False, unique=True)
    email = Column(String(100), nullable=False, unique=True)
    password = Column(String(100), nullable=False)
    created_at = Column("created_at", DateTime, default=func.now())
    refresh_token = Column(String(255), nullable=True)
    confirmed = Column(Boolean, default=False)  # email confirmed
    banned = Column(Boolean, default=False)
    role = Column(String(20), nullable=False, default="user")
    license_plate = Column(
        "license_plate", ForeignKey("cars_table.license_plate"), unique=True
    )
    tariff_id = Column("tariff_id", ForeignKey("tariffs_table.id"))
    blacklisted_token = relationship(
        "BlacklistedToken", uselist=False, back_populates="user"
    )
    cars = relationship("Car", uselist=True, back_populates="user")
    tariff = relationship("Tariff", back_populates="user")

    __table_args__ = (
        CheckConstraint(role.in_(["admin", "user"]), name="check_valid_role"),
    )

    def __str__(self):
        return str(self.id)


class Parking(Base):
    __tablename__ = "parking_places_table"

    id = Column(Integer, primary_key=True)
    enter_time = Column(DateTime(timezone=True), server_default=func.now())
    departure_time = Column(DateTime(timezone=True))
    status = Column(
        Boolean, default=False
    )  # false - parking place created and car on the parking / true - car left and parking place closed
    license_plate = Column(
        "license_plate", ForeignKey("cars_table.license_plate", ondelete="CASCADE")
    )
    amount_paid = Column(Numeric)
    duration = Column(Numeric)
    car = relationship("Car", uselist=False, back_populates="parking_place")


class Car(Base):
    __tablename__ = "cars_table"

    id = Column(Integer, primary_key=True)
    license_plate = Column(String(30), nullable=False, unique=True)
    banned = Column(Boolean, default=False)
    user = relationship("User", uselist=False, back_populates="cars")
    parking_place = relationship("Parking", uselist=False, back_populates="car")


class Tariff(Base):
    __tablename__ = "tariffs_table"

    id = Column(Integer, primary_key=True)
    tariff_name = Column(String(30), nullable=False, unique=True)
    tariff_value = Column(Numeric, default=20)
    user = relationship("User", back_populates="tariff")

    def __repr__(self) -> str:
        return (
            f"Tariff("
            f"id={self.id!r}, "
            f"tariff_name={self.tariff_name!r}, "
            f"tariff_value={self.tariff_value!r}"
            f")"
        )


class ParkingCount(Base):
    __tablename__ = "parking_count_table"

    id = Column(Integer, primary_key=True)
    total_quantity = Column(Integer, nullable=False, default=30)
    # Changed typo in name
    occupied_quantity = Column(Integer, default=0)

    def __repr__(self) -> str:
        return (
            f"ParkingCount("
            f"id={self.id!r}, "
            f"total_quantity={self.total_quantity!r}, "
            f"occupied_quantity={self.occupied_quantity!r}"
            f")"
        )


# Backwards-compatible alias for old imports.
# TODO: Replace all Parking_count imports/usages with ParkingCount,
# then remove this alias.


class BlacklistedToken(Base):
    __tablename__ = "blacklisted_tokens"

    id = Column(Integer, primary_key=True)
    blacklisted_token = Column(String(255), nullable=True)
    user_id = Column(
        "user_id", ForeignKey("users_table.id", ondelete="CASCADE"), unique=True
    )
    user = relationship("User", back_populates="blacklisted_token")
