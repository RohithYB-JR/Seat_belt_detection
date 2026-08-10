from pathlib import Path
import shutil

from PIL import Image


BASE_DIR = Path(__file__).resolve().parent.parent

WITH_DIR = BASE_DIR / "datasets" / "images" / "with"
WITHOUT_DIR = BASE_DIR / "datasets" / "images" / "without"
BIN_DIR = BASE_DIR / "datasets" / "Bin"


VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


def is_corrupt(image_path):
    try:
        with Image.open(image_path) as image:
            image.verify()

        return False, "Good"

    except Exception as error:
        return True, str(error)


def move_to_bin(image_path, reason):
    BIN_DIR.mkdir(parents=True, exist_ok=True)

    destination = BIN_DIR / image_path.name

    if destination.exists():
        destination = (
            BIN_DIR
            / f"{image_path.stem}_bad{image_path.suffix}"
        )

    shutil.move(str(image_path), str(destination))

    return destination


def clean_folder(folder):
    if not folder.exists():
        print(f"Folder not found: {folder}")
        return 0, 0

    checked = 0
    moved = 0

    images = [
        image_path
        for image_path in folder.iterdir()
        if image_path.is_file()
        and image_path.suffix.lower() in VALID_EXTENSIONS
    ]

    total = len(images)

    for image_path in images:
        checked += 1

        corrupt, reason = is_corrupt(image_path)

        if corrupt:
            destination = move_to_bin(image_path, reason)
            moved += 1

            print(
                f"[{checked}/{total}] "
                f"MOVED: {image_path.name}"
            )

        elif checked % 100 == 0:
            print(
                f"[{checked}/{total}] "
                f"Checking images..."
            )

    return checked, moved


def main():
    BIN_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("DATASET CLEANING")
    print("=" * 50)

    print("\nChecking 'with' images...")

    with_checked, with_moved = clean_folder(WITH_DIR)

    print("\nChecking 'without' images...")

    without_checked, without_moved = clean_folder(WITHOUT_DIR)

    total_checked = with_checked + without_checked
    total_moved = with_moved + without_moved
    total_kept = total_checked - total_moved

    print("\n" + "=" * 50)
    print("DATASET CLEANING COMPLETED")
    print("=" * 50)

    print(f"Images checked : {total_checked}")
    print(f"Images moved   : {total_moved}")
    print(f"Images kept    : {total_kept}")

    print("\nWITH:")
    print(f"  Checked : {with_checked}")
    print(f"  Moved   : {with_moved}")

    print("\nWITHOUT:")
    print(f"  Checked : {without_checked}")
    print(f"  Moved   : {without_moved}")


if __name__ == "__main__":
    main()