from pathlib import Path
import argparse
import csv
import hashlib
import shutil
from collections import defaultdict


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Create SHA256 hash for a file without loading the whole file into memory."""
    hasher = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)

    return hasher.hexdigest()


def find_image_files(source_dir: Path) -> list[Path]:
    return sorted(
        p for p in source_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def write_report(report_path: Path, duplicate_rows: list[dict]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "hash",
            "kept_file",
            "duplicate_file",
            "action",
            "duplicate_size_bytes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(duplicate_rows)


def remove_duplicate_files(
    source_dir: Path,
    duplicates_dir: Path,
    report_path: Path,
    apply_changes: bool,
    delete: bool,
) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source folder does not exist: {source_dir}")

    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source path is not a folder: {source_dir}")

    image_files = find_image_files(source_dir)

    print("=" * 80)
    print("Duplicate Plate Crop Cleaner")
    print("=" * 80)
    print(f"Source folder: {source_dir}")
    print(f"Image files found: {len(image_files):,}")
    print(f"Mode: {'APPLY' if apply_changes else 'DRY RUN'}")
    print(f"Action: {'DELETE duplicates' if delete else 'MOVE duplicates'}")
    print()

    hash_to_files: dict[str, list[Path]] = defaultdict(list)

    for index, image_path in enumerate(image_files, start=1):
        file_hash = sha256_file(image_path)
        hash_to_files[file_hash].append(image_path)

        if index % 500 == 0:
            print(f"Hashed {index:,}/{len(image_files):,} files...")

    duplicate_groups = {
        file_hash: files
        for file_hash, files in hash_to_files.items()
        if len(files) > 1
    }

    duplicate_rows = []
    duplicates_count = 0

    for file_hash, files in duplicate_groups.items():
        files = sorted(files)

        # Keep the first file by sorted filename/path.
        kept_file = files[0]
        duplicate_files = files[1:]

        for duplicate_file in duplicate_files:
            duplicates_count += 1

            action = "would_delete" if delete else "would_move"

            if apply_changes:
                if delete:
                    duplicate_file.unlink()
                    action = "deleted"
                else:
                    duplicates_dir.mkdir(parents=True, exist_ok=True)

                    destination = duplicates_dir / duplicate_file.name

                    # Avoid overwriting if same filename somehow already exists.
                    if destination.exists():
                        stem = destination.stem
                        suffix = destination.suffix
                        counter = 2

                        while True:
                            new_destination = duplicates_dir / f"{stem}_duplicate_{counter}{suffix}"
                            if not new_destination.exists():
                                destination = new_destination
                                break
                            counter += 1

                    shutil.move(str(duplicate_file), str(destination))
                    action = "moved"

            duplicate_rows.append(
                {
                    "hash": file_hash,
                    "kept_file": str(kept_file),
                    "duplicate_file": str(duplicate_file),
                    "action": action,
                    "duplicate_size_bytes": duplicate_file.stat().st_size if duplicate_file.exists() else "",
                }
            )

    write_report(report_path, duplicate_rows)

    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total images checked: {len(image_files):,}")
    print(f"Duplicate hash groups: {len(duplicate_groups):,}")
    print(f"Duplicate files found: {duplicates_count:,}")
    print(f"Report saved to: {report_path}")

    if not apply_changes:
        print()
        print("DRY RUN ONLY — no files were changed.")
        print("Run again with --apply to actually move/delete duplicates.")

    elif delete:
        print()
        print("Duplicates were permanently deleted.")

    else:
        print()
        print(f"Duplicates were moved to: {duplicates_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove exact duplicate cropped plate images by SHA256 hash."
    )

    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/raw/plate_crops_LLNNLLL_2"),
        help="Folder containing cleaned plate crops.",
    )

    parser.add_argument(
        "--duplicates-dir",
        type=Path,
        default=Path("data/raw/plate_crops_LLNNLLL_duplicates"),
        help="Folder where duplicate files are moved when not using --delete.",
    )

    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("outputs/real_crop_dataset_report/duplicate_removal_report.csv"),
        help="CSV report path.",
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually move/delete duplicate files. Without this, script only reports what would happen.",
    )

    parser.add_argument(
        "--delete",
        action="store_true",
        help="Permanently delete duplicates instead of moving them to duplicates folder.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    remove_duplicate_files(
        source_dir=args.source_dir,
        duplicates_dir=args.duplicates_dir,
        report_path=args.report_path,
        apply_changes=args.apply,
        delete=args.delete,
    )


if __name__ == "__main__":
    main()