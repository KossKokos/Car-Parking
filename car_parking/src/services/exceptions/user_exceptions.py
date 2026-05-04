class UserDomainError(Exception):
    """Base class for user-related domain errors."""


class CarNotRegisteredError(UserDomainError):
    """Raised when parking history is requested for an unknown car."""


class UserTariffNotFoundError(UserDomainError):
    """Raised when a user's assigned tariff cannot be found."""

class UserNotFoundError(UserDomainError):
    """Raised when a user cannot be found."""