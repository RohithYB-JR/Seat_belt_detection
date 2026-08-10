from pathlib import Path
import sys

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = BASE_DIR / "runs"
OUTPUT_DIR = BASE_DIR / "runs" / "inference"


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


def find_best_model():
    models = list(
        RUNS_DIR.glob("seat_belt_classifier*/weights/best.pt")
    )

    if not models:
        raise FileNotFoundError(
            "No trained best.pt model was found."
        )

    models.sort(
        key=lambda path: path.stat().st_mtime,
        reverse=True
    )

    return models[0]


def predict_image(model, image_path):

    print("\nRunning inference...")
    print(f"Image: {image_path.name}")

    results = model.predict(
        source=str(image_path),
        imgsz=224,
        verbose=False
    )

    result = results[0]

    probabilities = result.probs

    class_id = int(probabilities.top1)
    confidence = float(probabilities.top1conf)

    class_name = result.names[class_id]

    return class_name, confidence


def main():

    print("=" * 60)
    print("SEAT BELT IMAGE INFERENCE")
    print("=" * 60)

    if len(sys.argv) < 2:
        print("\nUsage:")
        print("python src\\inference.py <image_path>")
        return

    image_path = Path(sys.argv[1])

    if not image_path.is_absolute():
        image_path = BASE_DIR / image_path

    if not image_path.exists():
        print(f"\nERROR: Image not found:")
        print(image_path)
        return

    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        print("\nERROR: Unsupported image format.")
        return

    print("\nFinding trained model...")

    model_path = find_best_model()

    print(f"Model: {model_path}")

    print("\nLoading model...")

    model = YOLO(str(model_path))

    class_name, confidence = predict_image(
        model,
        image_path
    )

    print("\n" + "=" * 60)
    print("PREDICTION")
    print("=" * 60)

    print(f"Class      : {class_name}")
    print(f"Confidence : {confidence * 100:.2f}%")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results = model.predict(
        source=str(image_path),
        imgsz=224,
        save=True,
        project=str(OUTPUT_DIR),
        name="results",
        exist_ok=True,
        verbose=False
    )

    print("\nPrediction image saved to:")
    print(OUTPUT_DIR / "results")

    print("\nInference completed.")


if __name__ == "__main__":
    main()