class ParkingError(Exception):
    """Base class for parking domain errors."""


class ParkingFullError(ParkingError):
    """Raised when the parking is full."""


class ParkingPlaceNotFoundError(ParkingError):
    """Raised when a parking place/session cannot be found."""


class ParkingAlreadyClosedError(ParkingError):
    """Raised when trying to close an already closed parking session."""


class CarNotInParkingError(ParkingError):
    """Raised when a car does not have an active parking session."""