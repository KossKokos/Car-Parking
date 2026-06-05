from __future__ import annotations

import argparse
import csv
import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class LocalImageMetadataRecord:
    source_name: str
    source_page_url: str
    image_url: str
    local_path: str
    filename: str
    alt_text: str
    batch_number: int
    downloaded_at: str
    status: str
    image_width: int | None = None
    image_height: int | None = None
    element_timing: str | None = None
    error: str | None = None


def find_image_files(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    if not image_dir.is_dir():
        raise NotADirectoryError(f"Expected directory, got: {image_dir}")

    image_paths = [
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    return sorted(image_paths)


def get_existing_local_paths(metadata_csv: Path) -> set[str]:
    if not metadata_csv.exists() or metadata_csv.stat().st_size == 0:
        return set()

    with metadata_csv.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames or "local_path" not in reader.fieldnames:
            return set()

        return {
            row["local_path"]
            for row in reader
            if row.get("local_path")
        }


def get_image_dimensions(image_path: Path) -> tuple[int | None, int | None]:
    image = cv2.imread(str(image_path))

    if image is None:
        return None, None

    height, width = image.shape[:2]
    return int(width), int(height)


def make_local_image_url(relative_path: Path) -> str:
    """
    Mock URL for local images.

    This is intentionally not a real web URL.
    It only gives the metadata a stable unique source reference.
    """
    digest = hashlib.sha256(str(relative_path).encode("utf-8")).hexdigest()[:12]
    return f"local://phone_pictures/{digest}/{relative_path.name}"


def append_records_to_csv(
    records: list[LocalImageMetadataRecord],
    csv_path: Path,
) -> None:
    if not records:
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    new_rows = [asdict(record) for record in records]
    new_fieldnames = list(new_rows[0].keys())

    file_missing_or_empty = (
        not csv_path.exists()
        or csv_path.stat().st_size == 0
    )

    if file_missing_or_empty:
        with csv_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=new_fieldnames)
            writer.writeheader()
            writer.writerows(new_rows)
        return

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        existing_fieldnames = reader.fieldnames or []
        existing_rows = list(reader)

    merged_fieldnames = list(dict.fromkeys(existing_fieldnames + new_fieldnames))

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=merged_fieldnames)
        writer.writeheader()

        for row in existing_rows:
            writer.writerow(row)

        for row in new_rows:
            writer.writerow(row)


def register_phone_images(
    project_root: Path,
    image_dir: Path,
    metadata_csv: Path,
    source_name: str,
) -> None:
    image_paths = find_image_files(image_dir)
    existing_local_paths = get_existing_local_paths(metadata_csv)

    records: list[LocalImageMetadataRecord] = []
    skipped = 0

    for image_path in image_paths:
        relative_path = image_path.relative_to(project_root)
        relative_path_str = relative_path.as_posix()

        if relative_path_str in existing_local_paths:
            skipped += 1
            continue

        image_width, image_height = get_image_dimensions(image_path)

        record = LocalImageMetadataRecord(
            source_name=source_name,
            source_page_url="local://phone_pictures",
            image_url=make_local_image_url(relative_path),
            local_path=relative_path_str,
            filename=image_path.name,
            alt_text="Phone picture of a real car in London",
            batch_number=0,
            downloaded_at=datetime.now(timezone.utc).isoformat(),
            status="downloaded",
            image_width=image_width,
            image_height=image_height,
            element_timing=None,
            error=None,
        )

        records.append(record)

    append_records_to_csv(records, metadata_csv)

    print("Phone image registration complete.")
    print(f"Image directory: {image_dir}")
    print(f"Metadata CSV: {metadata_csv}")
    print(f"Found images: {len(image_paths)}")
    print(f"New records added: {len(records)}")
    print(f"Skipped existing records: {skipped}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--image-dir",
        default="data/raw/phone_pictures",
    )
    parser.add_argument(
        "--metadata-csv",
        default="data/metadata/scraped_images.csv",
    )
    parser.add_argument(
        "--source-name",
        default="phone_pictures",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(args.project_root).resolve()
    image_dir = (project_root / args.image_dir).resolve()
    metadata_csv = project_root / args.metadata_csv

    register_phone_images(
        project_root=project_root,
        image_dir=image_dir,
        metadata_csv=metadata_csv,
        source_name=args.source_name,
    )


if __name__ == "__main__":
    main()