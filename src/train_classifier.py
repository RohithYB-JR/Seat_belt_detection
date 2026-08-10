from pathlib import Path
import shutil

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "datasets" / "classification"
MODEL_DIR = BASE_DIR / "models"
RUNS_DIR = BASE_DIR / "runs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)


MODEL_NAME = "yolo11n-cls.pt"

EPOCHS = 1
IMAGE_SIZE = 224
BATCH_SIZE = 16


def find_latest_best_model():
    model_files = list(
        RUNS_DIR.glob("seat_belt_classifier*/weights/best.pt")
    )

    if not model_files:
        return None

    model_files.sort(
        key=lambda path: path.stat().st_mtime,
        reverse=True
    )

    return model_files[0]


def main():

    print("=" * 50)
    print("SEAT BELT CLASSIFICATION TRAINING")
    print("=" * 50)

    print(f"Dataset    : {DATASET_DIR}")
    print(f"Model      : {MODEL_NAME}")
    print(f"Epochs     : {EPOCHS}")
    print(f"Image size : {IMAGE_SIZE}")
    print(f"Batch size : {BATCH_SIZE}")

    print("\nLoading YOLO11 classification model...")

    model = YOLO(MODEL_NAME)

    print("\nStarting training...")

    model.train(
        data=str(DATASET_DIR),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        project=str(RUNS_DIR),
        name="seat_belt_classifier",
        pretrained=True,
        patience=8,
        workers=2
    )

    print("\nTraining completed.")

    best_model = find_latest_best_model()

    if best_model is None:
        print("\nERROR: Trained best.pt was not found.")
        print("Check the runs folder.")
        return

    destination = MODEL_DIR / "best.pt"

    shutil.copy2(
        best_model,
        destination
    )

    print("\n" + "=" * 50)
    print("MODEL SAVED")
    print("=" * 50)

    print(f"Source      : {best_model}")
    print(f"Final model : {destination}")


if __name__ == "__main__":
    main()