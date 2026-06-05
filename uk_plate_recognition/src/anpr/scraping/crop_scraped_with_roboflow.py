from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import httpx

from anpr.preprocessing.image_ops import crop_image_from_roboflow_prediction, load_image_bgr


ALLOWED_DETECTION_CLASSES = {
    "number-plates",
    "number_plate",
    "number_plates",
    "license_plate",
    "license-plate",
    "plate",
}

REQUIRED_ROBOFLOW_BOX_KEYS = {"x", "y", "width", "height"}


@dataclass(frozen=True)
class RoboflowCropConfig:
    project_root: Path
    scraped_metadata_csv: Path
    output_crop_dir: Path
    output_metadata_csv: Path
    output_metadata_json: Path
    roboflow_api_key: str
    roboflow_detect_url: str
    roboflow_confidence_percent: int = 80
    roboflow_overlap_percent: int = 30
    min_detection_confidence: float = 0.80
    crop_padding: float = 0.05
    request_timeout_seconds: float = 30.0
    sleep_seconds: float = 0.2
    source_name: str | None = None
    limit: int | None = None


@dataclass
class ScrapedPlateCropRecord:
    source_name: str | None
    source_page_url: str | None
    source_image_url: str | None
    source_image_path: str
    crop_path: str
    crop_filename: str
    detection_index: int
    detection_class: str | None
    detection_confidence: float | None
    x: float
    y: float
    width: float
    height: float
    crop_padding: float
    crop_height: int
    crop_width: int
    status: str
    processed_at: str
    error: str | None = None


def normalise_class_name(value: str | None) -> str:
    return (
        str(value or "")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def class_is_allowed(class_name: str | None) -> bool:
    normalised_allowed = {normalise_class_name(name) for name in ALLOWED_DETECTION_CLASSES}
    return normalise_class_name(class_name) in normalised_allowed


def load_rows_from_csv(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def load_processed_source_paths(metadata_csv_path: Path) -> set[str]:
    if not metadata_csv_path.exists():
        return set()

    with metadata_csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return {
            row["source_image_path"]
            for row in reader
            if row.get("source_image_path") and row.get("status") in {"cropped", "no_detection", "failed"}
        }


def append_records_to_csv(records: list[ScrapedPlateCropRecord], csv_path: Path) -> None:
    if not records:
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [asdict(record) for record in records]
    fieldnames = list(rows[0].keys())

    file_missing_or_empty = not csv_path.exists() or csv_path.stat().st_size == 0

    with csv_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if file_missing_or_empty:
            writer.writeheader()

        writer.writerows(rows)


def append_records_to_json(records: list[ScrapedPlateCropRecord], json_path: Path) -> None:
    if not records:
        return

    json_path.parent.mkdir(parents=True, exist_ok=True)

    existing: list[dict[str, Any]] = []

    if json_path.exists():
        try:
            existing = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = []

    existing.extend(asdict(record) for record in records)

    json_path.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def make_crop_filename(
    source_image_path: str,
    detection_index: int,
    detection_confidence: float | None,
) -> str:
    key = f"{source_image_path}|{detection_index}|{detection_confidence}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"{digest}_det_{detection_index}.jpg"


def call_roboflow(
    image_path: Path,
    config: RoboflowCropConfig,
) -> dict[str, Any]:
    image_bytes = image_path.read_bytes()

    params = {
        "api_key": config.roboflow_api_key,
        "confidence": config.roboflow_confidence_percent,
        "overlap": config.roboflow_overlap_percent,
    }

    files = {
        "file": (
            image_path.name,
            image_bytes,
            "application/octet-stream",
        )
    }

    with httpx.Client(timeout=config.request_timeout_seconds) as client:
        response = client.post(
            config.roboflow_detect_url,
            params=params,
            files=files,
        )

    response.raise_for_status()
    return response.json()


def extract_valid_detections(
    roboflow_response: dict[str, Any],
    min_detection_confidence: float,
) -> list[dict[str, Any]]:
    predictions = roboflow_response.get("predictions", [])

    if not isinstance(predictions, list):
        return []

    valid: list[dict[str, Any]] = []

    for prediction in predictions:
        if not isinstance(prediction, dict):
            continue

        if not REQUIRED_ROBOFLOW_BOX_KEYS.issubset(prediction):
            continue

        confidence = prediction.get("confidence")

        if confidence is None:
            continue

        confidence = float(confidence)

        if confidence < min_detection_confidence:
            continue

        if not class_is_allowed(prediction.get("class")):
            continue

        width = float(prediction["width"])
        height = float(prediction["height"])

        if width <= 0 or height <= 0:
            continue

        valid.append(prediction)

    valid.sort(
        key=lambda item: float(item.get("confidence", 0.0)),
        reverse=True,
    )

    return valid


def resolve_source_image_path(project_root: Path, local_path: str) -> Path:
    path = Path(local_path)

    if path.is_absolute():
        return path

    return project_root / path


def make_base_failure_record(
    row: dict[str, str],
    source_image_path: Path,
    status: str,
    error: str | None,
) -> ScrapedPlateCropRecord:
    return ScrapedPlateCropRecord(
        source_name=row.get("source_name"),
        source_page_url=row.get("source_page_url"),
        source_image_url=row.get("image_url"),
        source_image_path=str(source_image_path),
        crop_path="",
        crop_filename="",
        detection_index=-1,
        detection_class=None,
        detection_confidence=None,
        x=0.0,
        y=0.0,
        width=0.0,
        height=0.0,
        crop_padding=0.0,
        crop_height=0,
        crop_width=0,
        status=status,
        processed_at=datetime.now(timezone.utc).isoformat(),
        error=error,
    )


def process_one_image(
    row: dict[str, str],
    config: RoboflowCropConfig,
) -> list[ScrapedPlateCropRecord]:
    local_path = row.get("local_path", "")

    if not local_path:
        return [
            make_base_failure_record(
                row=row,
                source_image_path=Path(""),
                status="failed",
                error="Missing local_path in scraped metadata.",
            )
        ]

    source_image_path = resolve_source_image_path(
        project_root=config.project_root,
        local_path=local_path,
    )

    if not source_image_path.exists():
        return [
            make_base_failure_record(
                row=row,
                source_image_path=source_image_path,
                status="failed",
                error=f"Source image does not exist: {source_image_path}",
            )
        ]

    try:
        roboflow_response = call_roboflow(
            image_path=source_image_path,
            config=config,
        )

        detections = extract_valid_detections(
            roboflow_response=roboflow_response,
            min_detection_confidence=config.min_detection_confidence,
        )

    except Exception as exc:
        return [
            make_base_failure_record(
                row=row,
                source_image_path=source_image_path,
                status="failed",
                error=f"Roboflow call failed: {exc}",
            )
        ]

    if not detections:
        return [
            make_base_failure_record(
                row=row,
                source_image_path=source_image_path,
                status="no_detection",
                error=None,
            )
        ]

    try:
        image = load_image_bgr(source_image_path)
    except Exception as exc:
        return [
            make_base_failure_record(
                row=row,
                source_image_path=source_image_path,
                status="failed",
                error=f"Could not load image for cropping: {exc}",
            )
        ]

    records: list[ScrapedPlateCropRecord] = []

    config.output_crop_dir.mkdir(parents=True, exist_ok=True)

    for detection_index, detection in enumerate(detections):
        try:
            crop = crop_image_from_roboflow_prediction(
                image=image,
                prediction=detection,
                padding=config.crop_padding,
            )

            crop_filename = make_crop_filename(
                source_image_path=str(source_image_path),
                detection_index=detection_index,
                detection_confidence=float(detection.get("confidence", 0.0)),
            )

            crop_path = config.output_crop_dir / crop_filename

            ok = cv2.imwrite(str(crop_path), crop)

            if not ok:
                raise ValueError(f"cv2.imwrite failed for {crop_path}")

            crop_height, crop_width = crop.shape[:2]

            record = ScrapedPlateCropRecord(
                source_name=row.get("source_name"),
                source_page_url=row.get("source_page_url"),
                source_image_url=row.get("image_url"),
                source_image_path=str(source_image_path),
                crop_path=str(crop_path),
                crop_filename=crop_filename,
                detection_index=detection_index,
                detection_class=detection.get("class"),
                detection_confidence=float(detection.get("confidence", 0.0)),
                x=float(detection["x"]),
                y=float(detection["y"]),
                width=float(detection["width"]),
                height=float(detection["height"]),
                crop_padding=config.crop_padding,
                crop_height=int(crop_height),
                crop_width=int(crop_width),
                status="cropped",
                processed_at=datetime.now(timezone.utc).isoformat(),
                error=None,
            )

        except Exception as exc:
            record = make_base_failure_record(
                row=row,
                source_image_path=source_image_path,
                status="failed",
                error=f"Crop failed: {exc}",
            )

        records.append(record)

    return records


def filter_input_rows(
    rows: list[dict[str, str]],
    source_name: str | None,
) -> list[dict[str, str]]:
    filtered = []

    for row in rows:
        if row.get("status") != "downloaded":
            continue

        if source_name is not None and row.get("source_name") != source_name:
            continue

        if not row.get("local_path"):
            continue

        filtered.append(row)

    return filtered


def crop_scraped_images_with_roboflow(config: RoboflowCropConfig) -> None:
    rows = load_rows_from_csv(config.scraped_metadata_csv)
    rows = filter_input_rows(rows, source_name=config.source_name)

    processed_source_paths = load_processed_source_paths(config.output_metadata_csv)

    remaining_rows = []

    for row in rows:
        source_image_path = resolve_source_image_path(
            project_root=config.project_root,
            local_path=row["local_path"],
        )

        if str(source_image_path) in processed_source_paths:
            continue

        remaining_rows.append(row)

    if config.limit is not None:
        remaining_rows = remaining_rows[: config.limit]

    print(f"Input downloaded rows: {len(rows)}")
    print(f"Already processed: {len(rows) - len(remaining_rows)}")
    print(f"To process this run: {len(remaining_rows)}")
    print(f"Output crop dir: {config.output_crop_dir}")

    batch_records: list[ScrapedPlateCropRecord] = []

    total_cropped = 0
    total_no_detection = 0
    total_failed = 0

    for index, row in enumerate(remaining_rows, start=1):
        print(f"\n[{index}/{len(remaining_rows)}] {row.get('local_path')}")

        records = process_one_image(row=row, config=config)

        for record in records:
            if record.status == "cropped":
                total_cropped += 1
            elif record.status == "no_detection":
                total_no_detection += 1
            elif record.status == "failed":
                total_failed += 1

        batch_records.extend(records)

        print(
            f"records={len(records)} | "
            f"cropped={total_cropped} | "
            f"no_detection={total_no_detection} | "
            f"failed={total_failed}"
        )

        if len(batch_records) >= 20:
            append_records_to_csv(batch_records, config.output_metadata_csv)
            append_records_to_json(batch_records, config.output_metadata_json)
            batch_records.clear()

        if config.sleep_seconds > 0:
            time.sleep(config.sleep_seconds)

    append_records_to_csv(batch_records, config.output_metadata_csv)
    append_records_to_json(batch_records, config.output_metadata_json)

    print("\nDone.")
    print(f"Cropped: {total_cropped}")
    print(f"No detection: {total_no_detection}")
    print(f"Failed: {total_failed}")
    print(f"Crop metadata CSV: {config.output_metadata_csv}")
    print(f"Crop metadata JSON: {config.output_metadata_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--scraped-metadata-csv",
        default="data/metadata/scraped_images.csv",
    )
    parser.add_argument(
        "--output-crop-dir",
        default="data/raw/plate_crops_scraped",
    )
    parser.add_argument(
        "--output-metadata-csv",
        default="data/metadata/scraped_plate_crops.csv",
    )
    parser.add_argument(
        "--output-metadata-json",
        default="data/metadata/scraped_plate_crops.json",
    )
    parser.add_argument("--source-name", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-detection-confidence", type=float, default=0.80)
    parser.add_argument("--roboflow-confidence-percent", type=int, default=80)
    parser.add_argument("--roboflow-overlap-percent", type=int, default=30)
    parser.add_argument("--crop-padding", type=float, default=0.05)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    api_key = os.getenv("ROBOFLOW_API_KEY")
    detect_url = os.getenv("ROBOFLOW_PLATE_DETECT_URL")

    if not api_key:
        raise RuntimeError("Missing ROBOFLOW_API_KEY environment variable.")

    if not detect_url:
        raise RuntimeError("Missing ROBOFLOW_PLATE_DETECT_URL environment variable.")

    project_root = Path(args.project_root).resolve()

    config = RoboflowCropConfig(
        project_root=project_root,
        scraped_metadata_csv=project_root / args.scraped_metadata_csv,
        output_crop_dir=project_root / args.output_crop_dir,
        output_metadata_csv=project_root / args.output_metadata_csv,
        output_metadata_json=project_root / args.output_metadata_json,
        roboflow_api_key=api_key,
        roboflow_detect_url=detect_url,
        roboflow_confidence_percent=args.roboflow_confidence_percent,
        roboflow_overlap_percent=args.roboflow_overlap_percent,
        min_detection_confidence=args.min_detection_confidence,
        crop_padding=args.crop_padding,
        source_name=args.source_name,
        limit=args.limit,
        sleep_seconds=args.sleep_seconds,
    )

    crop_scraped_images_with_roboflow(config)


if __name__ == "__main__":
    main()
