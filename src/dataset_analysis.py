from pathlib import Path
from collections import Counter
from PIL import Image
import hashlib


BASE_DIR = Path(__file__).resolve().parent.parent

WITH_DIR = BASE_DIR / "datasets" / "images" / "with"
WITHOUT_DIR = BASE_DIR / "datasets" / "images" / "without"


VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


def get_images(folder):
    if not folder.exists():
        return []

    return [
        file for file in folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in VALID_EXTENSIONS
    ]


def check_images(images):
    valid = 0
    corrupted = 0
    dimensions = Counter()
    extensions = Counter()

    for image_path in images:
        extensions[image_path.suffix.lower()] += 1

        try:
            with Image.open(image_path) as image:
                image.load()

                dimensions[image.size] += 1
                valid += 1

        except Exception:
            corrupted += 1

    return valid, corrupted, dimensions, extensions


def get_file_hash(image_path):
    hash_md5 = hashlib.md5()

    try:
        with open(image_path, "rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                hash_md5.update(chunk)

        return hash_md5.hexdigest()

    except Exception:
        return None


def find_duplicates(images):
    hashes = {}
    duplicates = []

    for image_path in images:
        file_hash = get_file_hash(image_path)

        if file_hash is None:
            continue

        if file_hash in hashes:
            duplicates.append(
                (image_path, hashes[file_hash])
            )
        else:
            hashes[file_hash] = image_path

    return duplicates


def analyze_class(name, folder):
    print("\n" + "=" * 50)
    print(f"ANALYZING: {name}")
    print("=" * 50)

    images = get_images(folder)

    print(f"Images found: {len(images)}")

    valid, corrupted, dimensions, extensions = check_images(images)

    print(f"Valid images: {valid}")
    print(f"Corrupted images: {corrupted}")

    print("\nFile extensions:")

    for extension, count in extensions.most_common():
        print(f"  {extension}: {count}")

    print("\nImage dimensions:")

    for dimension, count in dimensions.most_common(10):
        print(f"  {dimension[0]} x {dimension[1]}: {count}")

    print("\nChecking exact duplicate files...")

    duplicates = find_duplicates(images)

    print(f"Duplicate files found: {len(duplicates)}")

    return len(images), valid, corrupted, len(duplicates)


def main():
    print("=" * 50)
    print("SEAT BELT DATASET ANALYSIS")
    print("=" * 50)

    with_total, with_valid, with_corrupt, with_duplicates = (
        analyze_class("WITH SEAT BELT", WITH_DIR)
    )

    without_total, without_valid, without_corrupt, without_duplicates = (
        analyze_class("WITHOUT SEAT BELT", WITHOUT_DIR)
    )

    total = with_total + without_total

    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)

    print(f"Total images      : {total}")
    print(f"With seat belt    : {with_total}")
    print(f"Without seat belt : {without_total}")

    if total > 0:
        with_percentage = (with_total / total) * 100
        without_percentage = (without_total / total) * 100

        print(f"\nWith percentage   : {with_percentage:.2f}%")
        print(f"Without percentage: {without_percentage:.2f}%")

    print("\nCorrupted images:")
    print(f"  With    : {with_corrupt}")
    print(f"  Without : {without_corrupt}")

    print("\nExact duplicate files:")
    print(f"  With    : {with_duplicates}")
    print(f"  Without : {without_duplicates}")

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()