from __future__ import annotations

from anpr.validation.uk_plate import validate_uk_plate


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"

POSITION_CHARSETS = (
    LETTERS,  # position 0
    LETTERS,  # position 1
    DIGITS,   # position 2
    DIGITS,   # position 3
    LETTERS,  # position 4
    LETTERS,  # position 5
    LETTERS,  # position 6
)

NUM_POSITIONS = 7
CLASS_COUNTS = tuple(len(charset) for charset in POSITION_CHARSETS)

CHAR_TO_INDEX = [
    {char: index for index, char in enumerate(charset)}
    for charset in POSITION_CHARSETS
]

INDEX_TO_CHAR = [
    {index: char for index, char in enumerate(charset)}
    for charset in POSITION_CHARSETS
]


def encode_label(label: str) -> list[int]:
    """
    Convert a UK plate label into 7 position-specific class indices.

    Example:
        AA04QZH -> [0, 0, 0, 4, 16, 25, 7]
    """
    cleaned_label = validate_uk_plate(label)

    encoded: list[int] = []

    for position, char in enumerate(cleaned_label):
        try:
            encoded.append(CHAR_TO_INDEX[position][char])
        except KeyError as exc:
            raise ValueError(
                f"Invalid character {char!r} at position {position} "
                f"for label {label!r}."
            ) from exc

    return encoded


def decode_indices(indices: list[int] | tuple[int, ...]) -> str:
    """
    Convert 7 position-specific class indices back into a plate string.
    """
    if len(indices) != NUM_POSITIONS:
        raise ValueError(
            f"Expected {NUM_POSITIONS} indices, got {len(indices)}."
        )

    chars: list[str] = []

    for position, index in enumerate(indices):
        if index not in INDEX_TO_CHAR[position]:
            raise ValueError(
                f"Invalid class index {index} at position {position}."
            )

        chars.append(INDEX_TO_CHAR[position][index])

    return "".join(chars)