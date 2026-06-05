from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


VALID_PLATE_PATTERN = re.compile(r"^([A-Z]{2}[0-9]{2}[A-Z]{3})", re.IGNORECASE)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def extract_plate_label(filename: str) -> str | None:
    match = VALID_PLATE_PATTERN.match(filename.upper())

    if not match:
        return None

    return match.group(1)


def make_unique_output_path(output_dir: Path, label: str, suffix: str) -> Path:
    """
    First file:
        AB12CDE.jpg

    Duplicates:
        AB12CDE_002.jpg
        AB12CDE_003.jpg
    """
    output_path = output_dir / f"{label}{suffix.lower()}"

    if not output_path.exists():
        return output_path

    counter = 2

    while True:
        output_path = output_dir / f"{label}_{counter:03d}{suffix.lower()}"

        if not output_path.exists():
            return output_path

        counter += 1


def find_image_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def clean_plate_crops(
    input_dir: Path,
    output_dir: Path | None,
    invalid_dir: Path | None,
    in_place: bool,
    invalid_action: str,
    apply: bool,
) -> None:
    image_paths = find_image_files(input_dir)

    valid_count = 0
    invalid_count = 0
    copied_or_renamed_count = 0
    deleted_count = 0
    moved_invalid_count = 0

    if not in_place and output_dir is None:
        raise ValueError("output_dir is required unless --in-place is used.")

    if invalid_action == "move" and invalid_dir is None:
        raise ValueError("invalid_dir is required when --invalid-action move is used.")

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    if invalid_dir is not None:
        invalid_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        label = extract_plate_label(image_path.name)

        if label is None:
            invalid_count += 1

            if invalid_action == "delete":
                print(f"[INVALID DELETE] {image_path}")

                if apply:
                    image_path.unlink()

                deleted_count += 1

            elif invalid_action == "move":
                target_path = invalid_dir / image_path.name
                print(f"[INVALID MOVE] {image_path} -> {target_path}")

                if apply:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(image_path), str(target_path))

                moved_invalid_count += 1

            else:
                print(f"[INVALID SKIP] {image_path}")

            continue

        valid_count += 1

        if in_place:
            target_path = make_unique_output_path(
                output_dir=image_path.parent,
                label=label,
                suffix=image_path.suffix,
            )

            # If file is already exactly correct, leave it.
            if image_path.resolve() == target_path.resolve():
                print(f"[VALID KEEP] {image_path}")
                continue

            print(f"[VALID RENAME] {image_path} -> {target_path}")

            if apply:
                image_path.rename(target_path)

            copied_or_renamed_count += 1

        else:
            assert output_dir is not None

            target_path = make_unique_output_path(
                output_dir=output_dir,
                label=label,
                suffix=image_path.suffix,
            )

            print(f"[VALID COPY] {image_path} -> {target_path}")

            if apply:
                shutil.copy2(image_path, target_path)

            copied_or_renamed_count += 1

    print("\nDone.")
    print(f"Total images found: {len(image_paths)}")
    print(f"Valid LLNNLLL images: {valid_count}")
    print(f"Invalid images: {invalid_count}")
    print(f"Copied/renamed valid images: {copied_or_renamed_count}")
    print(f"Deleted invalid images: {deleted_count}")
    print(f"Moved invalid images: {moved_invalid_count}")

    if not apply:
        print("\nDRY RUN ONLY. No files were changed.")
        print("Add --apply when the output looks correct.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing cropped plate images.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to copy cleaned labelled crops into.",
    )
    parser.add_argument(
        "--invalid-dir",
        default=None,
        help="Directory to move invalid images into if --invalid-action move is used.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Rename/delete/move files directly inside the input directory.",
    )
    parser.add_argument(
        "--invalid-action",
        choices=["skip", "move", "delete"],
        default="skip",
        help="What to do with files whose name does not start with LLNNLLL.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually modify files. Without this, the script only prints actions.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    clean_plate_crops(
        input_dir=Path(args.input_dir).resolve(),
        output_dir=Path(args.output_dir).resolve() if args.output_dir else None,
        invalid_dir=Path(args.invalid_dir).resolve() if args.invalid_dir else None,
        in_place=args.in_place,
        invalid_action=args.invalid_action,
        apply=args.apply,
    )


if __name__ == "__main__":
    main()