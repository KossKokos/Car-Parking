from __future__ import annotations

from pathlib import Path

from anpr.validation.uk_plate import (
    UK_PLATE_SEARCH_PATTERN,
    clean_plate_text,
    validate_uk_plate,
)


def parse_label_from_filename(
    file_path: str | Path,
    allow_embedded_plate: bool = False,
) -> str:
    """
    Extract a UK plate label from an image filename.

    Recommended strict filename:
        AA04QZH.png

    Also supports:
        AA04 QZH.png
        aa04qzh.jpg

    If allow_embedded_plate=True, this can also handle names like:
        AA04QZH_001.png

    But strict mode is safer for the baseline.
    """
    path = Path(file_path)
    stem = path.stem

    cleaned_stem = clean_plate_text(stem)

    try:
        return validate_uk_plate(cleaned_stem)
    except ValueError:
        if allow_embedded_plate:
            match = UK_PLATE_SEARCH_PATTERN.search(cleaned_stem)
            if match:
                return validate_uk_plate(match.group(0))

        raise ValueError(
            f"Could not parse a valid UK plate label from filename: {path.name!r}"
        )