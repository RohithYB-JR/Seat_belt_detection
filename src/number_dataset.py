from pathlib import Path


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
    return sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in VALID_EXTENSIONS
        ],
        key=lambda path: path.name.lower()
    )


def number_images(folder, prefix):
    if not folder.exists():
        print(f"Folder not found: {folder}")
        return

    images = get_images(folder)

    print(f"\nFound {len(images)} images in {folder.name}")

    # First rename to temporary names
    # This prevents filename conflicts during renaming.
    temporary_files = []

    for index, image_path in enumerate(images, start=1):
        temporary_name = folder / f"__temp_{index:05d}{image_path.suffix.lower()}"

        image_path.rename(temporary_name)

        temporary_files.append(temporary_name)

    # Rename temporary files to final names
    for index, image_path in enumerate(temporary_files, start=1):
        final_name = folder / f"{prefix}_{index:05d}{image_path.suffix.lower()}"

        image_path.rename(final_name)

        if index % 1000 == 0:
            print(f"  Renamed {index}/{len(images)}")

    print(f"Completed: {len(images)} images")


def main():
    print("=" * 50)
    print("DATASET NUMBERING")
    print("=" * 50)

    number_images(WITH_DIR, "with")
    number_images(WITHOUT_DIR, "without")

    print("\n" + "=" * 50)
    print("NUMBERING COMPLETED")
    print("=" * 50)


if __name__ == "__main__":
    main()