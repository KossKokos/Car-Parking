from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np

from open_image_models import LicensePlateDetector
from fast_plate_ocr import LicensePlateRecognizer


lp_detector = LicensePlateDetector(detection_model="yolo-v9-t-256-license-plate-end2end")
lp_reader = LicensePlateRecognizer('cct-s-v2-global-model')


UK_FIXED_PLATE_PATTERN = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{3}$")


@dataclass(frozen=True)
class PlateRecognizerCropConfig:
    project_root: Path
    scraped_metadata_csv: Path
    output_crop_dir: Path
    output_metadata_csv: Path
    output_metadata_json: Path
    # api_token: str
    # api_url: str = "https://api.platerecognizer.com/v1/plate-reader/"
    regions: tuple[str, ...] = ("gb",)
    min_detection_confidence: float = 0.90
    min_ocr_confidence: float = 0.98
    crop_padding: float = 0.05
    # request_timeout_seconds: float = 40.0
    # sleep_seconds: float = 1.1
    # max_retries: int = 2
    source_name: str | None = None
    limit: int | None = None
    require_uk_format: bool = False


@dataclass
class PlateRecognizerCropRecord:
    source_name: str | None
    source_page_url: str | None
    source_image_url: str | None
    source_image_path: str
    crop_path: str
    crop_filename: str

    result_index: int
    suggested_label: str | None
    suggested_label_clean: str | None
    is_uk_format: bool

    detection_confidence: float | None
    ocr_confidence: float | None
    region_code: str | None
    region_score: float | None

    box_xmin: int
    box_ymin: int
    box_xmax: int
    box_ymax: int
    crop_padding: float
    crop_height: int
    crop_width: int

    status: str
    processed_at: str
    error: str | None = None


def clean_plate_text(value: str | None) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def is_fixed_uk_plate(value: str | None) -> bool:
    return bool(UK_FIXED_PLATE_PATTERN.fullmatch(clean_plate_text(value)))


def load_rows_from_csv(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


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


def resolve_source_image_path(project_root: Path, local_path: str) -> Path:
    path = Path(local_path)

    if path.is_absolute():
        return path

    return project_root / path


def load_processed_source_paths(metadata_csv_path: Path) -> set[str]:
    if not metadata_csv_path.exists() or metadata_csv_path.stat().st_size == 0:
        return set()

    with metadata_csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames or "source_image_path" not in reader.fieldnames:
            return set()

        return {
            row["source_image_path"]
            for row in reader
            if row.get("source_image_path")
            and row.get("status") in {"cropped", "no_plate", "filtered"}
        }


def append_records_to_csv(
    records: list[PlateRecognizerCropRecord],
    csv_path: Path,
) -> None:
    if not records:
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    new_rows = [asdict(record) for record in records]
    new_fieldnames = list(new_rows[0].keys())

    file_missing_or_empty = not csv_path.exists() or csv_path.stat().st_size == 0

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


def append_records_to_json(
    records: list[PlateRecognizerCropRecord],
    json_path: Path,
) -> None:
    if not records:
        return

    json_path.parent.mkdir(parents=True, exist_ok=True)

    existing: list[dict[str, Any]] = []

    if json_path.exists():
        try:
            with open(json_path, 'r', encoding="utf-8") as f:
                existing = json.load(f)
        except json.JSONDecodeError:
            existing = []
        except PermissionError:
            return None


    existing.extend(asdict(record) for record in records)

    json_path.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def make_crop_filename(
    source_image_path: str,
    result_index: int,
    suggested_label_clean: str | None,
) -> str:
    label_part = suggested_label_clean or "unknown"

    key = f"{source_image_path}|{result_index}|{label_part}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]

    return f"{label_part}_{digest}_det_{result_index}.jpg"


def read_image_bgr(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    return image


def crop_xyxy_with_padding(
    image: np.ndarray,
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
    padding: float,
) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("image cannot be empty.")

    image_height, image_width = image.shape[:2]

    box_width = xmax - xmin
    box_height = ymax - ymin

    if box_width <= 0 or box_height <= 0:
        raise ValueError(f"Invalid box: {(xmin, ymin, xmax, ymax)}")

    pad_x = int(round(box_width * padding))
    pad_y = int(round(box_height * padding))

    x1 = max(0, xmin - pad_x)
    y1 = max(0, ymin - pad_y)
    x2 = min(image_width, xmax + pad_x)
    y2 = min(image_height, ymax + pad_y)

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid clipped box: {(x1, y1, x2, y2)}")

    return image[y1:y2, x1:x2].copy()


def call_plate_recognizer(
    image_path: Path,
    # config: PlateRecognizerCropConfig,
) -> dict[str, Any]:
    """
    Call plate recognizer and plate reader models.
    Returns response in format:
        {
        detection_payload: [],
        reader_payload: []
        }
    """

    # data = {
    #     "regions": ",".join(config.regions),
    # }
    full_response = {}
    
    try:
        detection_payload = lp_detector.predict(image_path)

        if len(detection_payload) <= 0:
            return full_response 
        detection_payload = detection_payload[0] # [0] used to extract first prediction in the responce from the list

        x1, x2 = detection_payload.bounding_box.x1, detection_payload.bounding_box.x2
        y1, y2 = detection_payload.bounding_box.y1, detection_payload.bounding_box.y2

        orig_img = cv2.imread(filename=image_path)
        cropped_img = orig_img[y1:y2, x1:x2]

        reader_payload = lp_reader.run(cropped_img, return_confidence=True)[0] # [0] used to extract first prediction in the responce from the list

        full_response["detection_payload"] = detection_payload
        full_response["reader_payload"] = reader_payload
        return full_response
    
    except Exception as err:
        print(err)
        return full_response

    # files = {\
    #     "upload": (
    #         image_path.name,
    #         image_bytes,
    #         "application/octet-stream",
    #     )
    # }

    # headers = {
    #     "Authorization": f"Token {config.api_token}",
    # }

    # last_error: Exception | None = None

    # for attempt in range(1, config.max_retries + 2):
    #     try:
    #         with httpx.Client(timeout=config.request_timeout_seconds) as client:
    #             response = client.post(
    #                 config.api_url,
    #                 data=data,
    #                 files=files,
    #                 headers=headers,
    #             )

    #         if response.status_code == 429:
    #             wait_seconds = max(config.sleep_seconds, 1.5 * attempt)
    #             print(f"Rate limited. Waiting {wait_seconds:.1f}s before retry.")
    #             time.sleep(wait_seconds)
    #             continue

    #         response.raise_for_status()
    #         return response.json()

    #     except Exception as exc:
    #         last_error = exc

    #         if attempt <= config.max_retries:
    #             wait_seconds = max(config.sleep_seconds, 1.5 * attempt)
    #             print(
    #                 f"Plate Recognizer call failed, retrying in "
    #                 f"{wait_seconds:.1f}s: {exc}"
    #             )
    #             time.sleep(wait_seconds)
    #             continue

    # raise RuntimeError(f"Plate Recognizer call failed after retries: {last_error}")


def extract_results(
    response: dict[str, Any],
    config: PlateRecognizerCropConfig,
) -> list[dict[str, Any]]:
    # raw_results = response.get("results", [])
    raw_response = deepcopy(response)
    valid_results = []

    detection_payload = raw_response.get("detection_payload", {})
    reader_payload = raw_response.get("reader_payload", {})
    
    if isinstance(detection_payload, dict):
        return valid_results 
    # if not isinstance(raw_results, list):
    #     return []

    # valid_results: list[dict[str, Any]] = []

    # for result in raw_results:
    #     if not isinstance(result, dict):
    #         continue

    plate_raw = reader_payload.plate
    plate_clean = clean_plate_text(plate_raw)

    x1, x2 = detection_payload.bounding_box.x1, detection_payload.bounding_box.x2
    y1, y2 = detection_payload.bounding_box.y1, detection_payload.bounding_box.y2
    # box = detection_payload.get("box") or {}
    box = {
        "xmin": x1,
        "xmax": x2,
        "ymin": y1,
        "ymax": y2
    }

    region_code = reader_payload.region
    region_score = reader_payload.region_prob
    region = {
        "code": region_code,
        "score": region_score  
    }

    # required_box_keys = {"x1", "y1", "x2", "y2"}

    # if not isinstance(box, dict) or not required_box_keys.issubset(box):
    #     continue

    dscore = detection_payload.confidence
    score = np.mean(reader_payload.char_probs)

    dscore = float(dscore)
    score = float(score)

    if dscore < config.min_detection_confidence:
        return valid_results

    if score < config.min_ocr_confidence:
        return valid_results

    if config.require_uk_format and not is_fixed_uk_plate(plate_clean):
        return valid_results

    valid_response = {
        "box": box,
        "plate": plate_raw, 
        "region": region,
    }
    valid_results.append(valid_response)
    return valid_results


def make_status_record(
    row: dict[str, str],
    source_image_path: Path,
    status: str,
    error: str | None,
) -> PlateRecognizerCropRecord:
    return PlateRecognizerCropRecord(
        source_name=row.get("source_name"),
        source_page_url=row.get("source_page_url"),
        source_image_url=row.get("image_url"),
        source_image_path=str(source_image_path),
        crop_path="",
        crop_filename="",
        result_index=-1,
        suggested_label=None,
        suggested_label_clean=None,
        is_uk_format=False,
        detection_confidence=None,
        ocr_confidence=None,
        region_code=None,
        region_score=None,
        box_xmin=0,
        box_ymin=0,
        box_xmax=0,
        box_ymax=0,
        crop_padding=0.0,
        crop_height=0,
        crop_width=0,
        status=status,
        processed_at=datetime.now(timezone.utc).isoformat(),
        error=error,
    )


def process_one_image(
    row: dict[str, str],
    config: PlateRecognizerCropConfig,
) -> list[PlateRecognizerCropRecord]:
    local_path = row.get("local_path", "")

    if not local_path:
        return [
            make_status_record(
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
            make_status_record(
                row=row,
                source_image_path=source_image_path,
                status="failed",
                error=f"Source image does not exist: {source_image_path}",
            )
        ]

    try:
        response = call_plate_recognizer(
            image_path=source_image_path,
            # config=config,
        )

        results = extract_results(
            response=response,
            config=config,
        )

    except Exception as exc:
        return [
            make_status_record(
                row=row,
                source_image_path=source_image_path,
                status="failed",
                error=str(exc),
            )
        ]

    if not results:
        return [
            make_status_record(
                row=row,
                source_image_path=source_image_path,
                status="no_plate",
                error=None,
            )
        ]

    try:
        image = read_image_bgr(source_image_path)
    except Exception as exc:
        return [
            make_status_record(
                row=row,
                source_image_path=source_image_path,
                status="failed",
                error=f"Could not load image for cropping: {exc}",
            )
        ]

    config.output_crop_dir.mkdir(parents=True, exist_ok=True)

    records: list[PlateRecognizerCropRecord] = []

    for result_index, result in enumerate(results):
        try:
            box = result["box"]

            xmin = int(box["xmin"])
            ymin = int(box["ymin"])
            xmax = int(box["xmax"])
            ymax = int(box["ymax"])

            plate_raw = result.get("plate")
            plate_clean = clean_plate_text(plate_raw)

            crop = crop_xyxy_with_padding(
                image=image,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
                padding=config.crop_padding,
            )

            crop_filename = make_crop_filename(
                source_image_path=str(source_image_path),
                result_index=result_index,
                suggested_label_clean=plate_clean,
            )

            crop_path = config.output_crop_dir / crop_filename

            ok = cv2.imwrite(str(crop_path), crop)

            if not ok:
                raise ValueError(f"cv2.imwrite failed for {crop_path}")

            crop_height, crop_width = crop.shape[:2]

            region = result.get("region") or {}

            record = PlateRecognizerCropRecord(
                source_name=row.get("source_name"),
                source_page_url=row.get("source_page_url"),
                source_image_url=row.get("image_url"),
                source_image_path=str(source_image_path),
                crop_path=str(crop_path),
                crop_filename=crop_filename,
                result_index=result_index,
                suggested_label=plate_raw,
                suggested_label_clean=plate_clean,
                is_uk_format=is_fixed_uk_plate(plate_clean),
                detection_confidence=float(result.get("dscore", 0.0)),
                ocr_confidence=float(result.get("score", 0.0)),
                region_code=region.get("code"),
                region_score=float(region["score"]) if region.get("score") is not None else None,
                box_xmin=xmin,
                box_ymin=ymin,
                box_xmax=xmax,
                box_ymax=ymax,
                crop_padding=config.crop_padding,
                crop_height=int(crop_height),
                crop_width=int(crop_width),
                status="cropped",
                processed_at=datetime.now(timezone.utc).isoformat(),
                error=None,
            )

        except Exception as exc:
            record = make_status_record(
                row=row,
                source_image_path=source_image_path,
                status="failed",
                error=f"Crop failed: {exc}",
            )

        records.append(record)

    return records


def crop_scraped_images_with_plate_recognizer(
    config: PlateRecognizerCropConfig,
) -> None:
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
    print(f"Regions: {config.regions}")
    print(f"Require UK fixed format: {config.require_uk_format}")

    batch_records: list[PlateRecognizerCropRecord] = []

    total_cropped = 0
    total_no_plate = 0
    total_failed = 0
    total_filtered = 0

    for index, row in enumerate(remaining_rows, start=1):
        print(f"\n[{index}/{len(remaining_rows)}] {row.get('local_path')}")

        records = process_one_image(row=row, config=config)

        for record in records:
            if record.status == "cropped":
                total_cropped += 1
            elif record.status == "no_plate":
                total_no_plate += 1
            elif record.status == "failed":
                total_failed += 1
            elif record.status == "filtered":
                total_filtered += 1

        batch_records.extend(records)

        print(
            f"records={len(records)} | "
            f"cropped={total_cropped} | "
            f"no_plate={total_no_plate} | "
            f"failed={total_failed} | "
            f"filtered={total_filtered}"
        )

        if len(batch_records) >= 20:
            append_records_to_csv(batch_records, config.output_metadata_csv)
            append_records_to_json(batch_records, config.output_metadata_json)
            batch_records.clear()

        # if config.sleep_seconds > 0:
        #     time.sleep(config.sleep_seconds)

    append_records_to_csv(batch_records, config.output_metadata_csv)
    append_records_to_json(batch_records, config.output_metadata_json)

    print("\nDone.")
    print(f"Cropped: {total_cropped}")
    print(f"No plate: {total_no_plate}")
    print(f"Failed: {total_failed}")
    print(f"Crop metadata CSV: {config.output_metadata_csv}")
    print(f"Crop metadata JSON: {config.output_metadata_json}")


def parse_regions(value: str) -> tuple[str, ...]:
    return tuple(
        item.strip()
        for item in value.split(",")
        if item.strip()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--scraped-metadata-csv",
        default="data/metadata/scraped_images.csv",
    )
    parser.add_argument(
        "--output-crop-dir",
        default="data/raw/plate_crops_plate_recognizer",
    )
    parser.add_argument(
        "--output-metadata-csv",
        default="data/metadata/plate_recognizer_crops.csv",
    )
    parser.add_argument(
        "--output-metadata-json",
        default="data/metadata/plate_recognizer_crops.json",
    )
    parser.add_argument("--source-name", default=None)
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--regions", default="gb")
    parser.add_argument("--min-detection-confidence", type=float, default=0.70)
    parser.add_argument("--min-ocr-confidence", type=float, default=0.70)
    parser.add_argument("--crop-padding", type=float, default=0.05)
    # parser.add_argument("--sleep-seconds", type=float, default=1.1)
    # parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--require-uk-format", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # api_token = os.getenv("PLATE_RECOGNIZER_API_TOKEN")
    # api_url = os.getenv(
    #     "PLATE_RECOGNIZER_API_URL",
    #     "https://api.platerecognizer.com/v1/plate-reader/",
    # )

    # if not api_token:
    #     raise RuntimeError("Missing PLATE_RECOGNIZER_API_TOKEN environment variable.")

    project_root = Path(args.project_root).resolve()

    config = PlateRecognizerCropConfig(
        project_root=project_root,
        scraped_metadata_csv=project_root / args.scraped_metadata_csv,
        output_crop_dir=project_root / args.output_crop_dir,
        output_metadata_csv=project_root / args.output_metadata_csv,
        output_metadata_json=project_root / args.output_metadata_json,
        # api_token=api_token,
        # api_url=api_url,
        regions=parse_regions(args.regions),
        min_detection_confidence=args.min_detection_confidence,
        min_ocr_confidence=args.min_ocr_confidence,
        crop_padding=args.crop_padding,
        # sleep_seconds=args.sleep_seconds,
        # max_retries=args.max_retries,
        source_name=args.source_name,
        limit=args.limit,
        require_uk_format=args.require_uk_format,
    )

    crop_scraped_images_with_plate_recognizer(config)


if __name__ == "__main__":
    main()