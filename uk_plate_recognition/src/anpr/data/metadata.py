from __future__ import annotations

from pathlib import Path

import pandas as pd

from anpr.data.label_parser import parse_label_from_filename
from anpr.data.split import assign_splits


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_image_files(image_dir: str | Path) -> list[Path]:
    image_dir = Path(image_dir)

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    return sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def make_relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def build_metadata_csv(
    image_dir: str | Path,
    output_csv: str | Path,
    project_root: str | Path,
    invalid_csv: str | Path | None = None,
    source_name: str = "plate_crop_dataset",
    allow_embedded_plate: bool = False,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Build metadata CSV from cropped plate images.

    Expected filename:
        AA04QZH.png

    Output columns:
        image_id
        image_path
        label
        label_length
        source
        split
    """
    image_dir = Path(image_dir)
    output_csv = Path(output_csv)
    project_root = Path(project_root)

    image_files = find_image_files(image_dir)

    valid_rows: list[dict] = []
    invalid_rows: list[dict] = []

    for image_path in image_files:
        try:
            label = parse_label_from_filename(
                image_path,
                allow_embedded_plate=allow_embedded_plate,
            )

            valid_rows.append(
                {
                    "image_id": image_path.stem,
                    "image_path": make_relative_path(image_path, project_root),
                    "label": label,
                    "label_length": len(label),
                    "source": source_name,
                }
            )

        except ValueError as exc:
            invalid_rows.append(
                {
                    "image_path": make_relative_path(image_path, project_root),
                    "reason": str(exc),
                }
            )

    if not valid_rows:
        raise ValueError(
            "No valid plate images found. Check filenames and expected LLDDLLL format."
        )

    df = pd.DataFrame(valid_rows)

    df = assign_splits(
        df,
        train_size=0.8,
        val_size=0.1,
        test_size=0.1,
        group_col="label",
        random_state=random_state,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    if invalid_csv is not None and invalid_rows:
        invalid_csv = Path(invalid_csv)
        invalid_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(invalid_rows).to_csv(invalid_csv, index=False)

    return df