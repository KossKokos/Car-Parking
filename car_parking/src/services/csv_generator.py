from pathlib import Path
import re

from fastapi import HTTPException, status


CSV_DIRECTORY = Path(__file__).resolve().parents[2] / "csv_files"
SAFE_FILENAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


def build_csv_file_path(filename: str) -> Path:
    if not SAFE_FILENAME_PATTERN.fullmatch(filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename. Use only letters, numbers, underscores, and hyphens.",
        )

    CSV_DIRECTORY.mkdir(parents=True, exist_ok=True)

    return CSV_DIRECTORY / f"{filename}.csv"