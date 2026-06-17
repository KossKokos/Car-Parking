from pathlib import Path
import argparse
import csv
import shutil

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def blur_score(image_path: Path) -> float:
    """Estimate sharpness using the variance of the Laplacian."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        return 0.0

    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def image_size(image_path: Path) -> tuple[int | None, int | None]:
    image = cv2.imread(str(image_path))

    if image is None:
        return None, None

    height, width = image.shape[:2]
    return width, height


def safe_move(source: Path, destination_dir: Path) -> Path:
    """Move a file without overwriting an existing destination filename."""
    destination_dir.mkdir(parents=True, exist_ok=True)

    destination = destination_dir / source.name

    if not destination.exists():
        shutil.move(str(source), str(destination))
        return destination

    stem = source.stem
    suffix = source.suffix
    counter = 2

    while True:
        new_destination = destination_dir / f"{stem}_duplicate_{counter}{suffix}"

        if not new_destination.exists():
            shutil.move(str(source), str(new_destination))
            return new_destination

        counter += 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move very small, badly shaped, or blurry plate crops into a low-quality folder."
    )

    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/raw/plate_crops_LLNNLLL_2"),
        help="Folder containing cleaned LLNNLLL plate crops.",
    )

    parser.add_argument(
        "--low-quality-dir",
        type=Path,
        default=Path("data/raw/plate_crops_LLNNLLL_low_quality"),
        help="Folder where low-quality crops will be moved.",
    )

    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("outputs/real_crop_dataset_report/low_quality_filter_report.csv"),
        help="CSV report path.",
    )

    parser.add_argument("--min-width", type=int, default=70)
    parser.add_argument("--min-height", type=int, default=22)
    parser.add_argument("--min-aspect-ratio", type=float, default=2.2)
    parser.add_argument("--max-aspect-ratio", type=float, default=7.5)

    parser.add_argument(
        "--min-blur-score",
        type=float,
        default=20.0,
        help="Lower value means blurrier image. Start with 20.0, then inspect results.",
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually move files. Without this, script only reports what would happen.",
    )

    args = parser.parse_args()

    if not args.source_dir.exists():
        raise FileNotFoundError(f"Source folder does not exist: {args.source_dir}")

    image_files = sorted(
        path
        for path in args.source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    args.report_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    low_quality_count = 0

    print("=" * 80)
    print("Low Quality Plate Crop Filter")
    print("=" * 80)
    print(f"Source folder: {args.source_dir}")
    print(f"Images found: {len(image_files):,}")
    print(f"Mode: {'APPLY' if args.apply else 'DRY RUN'}")
    print()

    for index, image_path in enumerate(image_files, start=1):
        width, height = image_size(image_path)
        score = blur_score(image_path)

        reasons = []

        if width is None or height is None:
            reasons.append("unreadable_image")
            aspect_ratio = None
        else:
            aspect_ratio = width / height if height > 0 else None

            if width < args.min_width:
                reasons.append(f"width<{args.min_width}")

            if height < args.min_height:
                reasons.append(f"height<{args.min_height}")

            if aspect_ratio is not None and aspect_ratio < args.min_aspect_ratio:
                reasons.append(f"aspect_ratio<{args.min_aspect_ratio}")

            if aspect_ratio is not None and aspect_ratio > args.max_aspect_ratio:
                reasons.append(f"aspect_ratio>{args.max_aspect_ratio}")

            if score < args.min_blur_score:
                reasons.append(f"blur_score<{args.min_blur_score}")

        is_low_quality = len(reasons) > 0
        action = "keep"

        if is_low_quality:
            low_quality_count += 1
            action = "would_move"

            if args.apply:
                safe_move(image_path, args.low_quality_dir)
                action = "moved"

        rows.append(
            {
                "filename": image_path.name,
                "width": width,
                "height": height,
                "aspect_ratio": round(aspect_ratio, 4) if aspect_ratio else "",
                "blur_score": round(score, 4),
                "low_quality": is_low_quality,
                "reasons": ", ".join(reasons),
                "action": action,
            }
        )

        if index % 1000 == 0:
            print(f"Checked {index:,}/{len(image_files):,} images...")

    with args.report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "width",
                "height",
                "aspect_ratio",
                "blur_score",
                "low_quality",
                "reasons",
                "action",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Images checked: {len(image_files):,}")
    print(f"Low-quality candidates: {low_quality_count:,}")
    print(f"Report saved to: {args.report_path}")

    if not args.apply:
        print()
        print("DRY RUN ONLY — no files were moved.")
        print("Open the report and inspect the low-quality candidates first.")
        print("Then run again with --apply.")
    else:
        print()
        print(f"Moved low-quality crops to: {args.low_quality_dir}")


if __name__ == "__main__":
    main()
