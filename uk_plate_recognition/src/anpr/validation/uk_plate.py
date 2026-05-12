from __future__ import annotations

import re


UK_PLATE_PATTERN = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{3}$")
UK_PLATE_SEARCH_PATTERN = re.compile(r"[A-Z]{2}[0-9]{2}[A-Z]{3}")


def clean_plate_text(value: str) -> str:
    """
    Normalise plate text by removing spaces/symbols and uppercasing.

    Example:
        "aa04 qzh" -> "AA04QZH"
    """
    if value is None:
        raise ValueError("Plate value cannot be None.")

    return re.sub(r"[^A-Za-z0-9]", "", str(value)).upper()


def is_valid_uk_plate(value: str) -> bool:
    """
    Check whether a value follows the fixed UK LLDDLLL format.
    """
    cleaned = clean_plate_text(value)
    return bool(UK_PLATE_PATTERN.fullmatch(cleaned))


def validate_uk_plate(value: str) -> str:
    """
    Return cleaned plate text if valid, otherwise raise ValueError.
    """
    cleaned = clean_plate_text(value)

    if not UK_PLATE_PATTERN.fullmatch(cleaned):
        raise ValueError(
            f"Invalid UK plate label: {value!r}. "
            "Expected format: LLDDLLL, for example AA04QZH."
        )

    return cleaned