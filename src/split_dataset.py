from pathlib import Path
import random
import shutil


BASE_DIR = Path(__file__).resolve().parent.parent

SOURCE_DIR = BASE_DIR / "datasets" / "images"
OUTPUT_DIR = BASE_DIR / "datasets" / "classification"

CLASSES = ["with", "without"]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

RANDOM_SEED = 42

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


def get_images(folder):
    return [
        file for file in folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in VALID_EXTENSIONS
    ]


def create_folders():
    for split in ["train", "val", "test"]:
        for class_name in CLASSES:
            folder = OUTPUT_DIR / split / class_name
            folder.mkdir(parents=True, exist_ok=True)


def split_images(images):
    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    return train_images, val_images, test_images


def copy_images(images, destination):
    for image in images:
        shutil.copy2(image, destination / image.name)


def process_class(class_name):
    source_folder = SOURCE_DIR / class_name

    if not source_folder.exists():
        print(f"Folder not found: {source_folder}")
        return

    images = get_images(source_folder)

    print("\n" + "=" * 50)
    print(f"CLASS: {class_name}")
    print("=" * 50)

    print(f"Total images: {len(images)}")

    train_images, val_images, test_images = split_images(images)

    print(f"Training   : {len(train_images)}")
    print(f"Validation : {len(val_images)}")
    print(f"Testing    : {len(test_images)}")

    copy_images(
        train_images,
        OUTPUT_DIR / "train" / class_name
    )

    copy_images(
        val_images,
        OUTPUT_DIR / "val" / class_name
    )

    copy_images(
        test_images,
        OUTPUT_DIR / "test" / class_name
    )


def main():
    random.seed(RANDOM_SEED)

    create_folders()

    print("=" * 50)
    print("DATASET SPLITTING")
    print("=" * 50)

    process_class("with")
    process_class("without")

    print("\n" + "=" * 50)
    print("DATASET SPLIT COMPLETED")
    print("=" * 50)

    print("\nOriginal images were not modified.")
    print("Images were copied into the classification dataset.")


if __name__ == "__main__":
    main()