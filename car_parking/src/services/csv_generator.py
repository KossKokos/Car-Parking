import re

from fastapi import HTTPException, status


SAFE_FILENAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


def build_csv_download_filename(filename: str) -> str:
    """Validate a user-provided download name and append the CSV extension."""
    if not SAFE_FILENAME_PATTERN.fullmatch(filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Invalid filename. Use only letters, numbers, underscores, "
                "and hyphens."
            ),
        )

    return f"{filename}.csv"
