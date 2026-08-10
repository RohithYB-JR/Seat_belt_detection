from pathlib import Path

import numpy as np
from ultralytics import YOLO
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent.parent

TEST_DIR = BASE_DIR / "datasets" / "classification" / "test"
MODEL_PATH = BASE_DIR / "models" / "best.pt"
RESULTS_DIR = BASE_DIR / "runs" / "evaluation"

IMAGE_SIZE = 224
BATCH_SIZE = 16


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    print(f"Model: {MODEL_PATH}")

    return YOLO(str(MODEL_PATH))


def load_test_images():

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    image_paths = []
    true_labels = []

    class_names = sorted(
        [
            folder.name
            for folder in TEST_DIR.iterdir()
            if folder.is_dir()
        ]
    )

    class_to_index = {
        name: index
        for index, name in enumerate(class_names)
    }

    for class_name in class_names:

        class_dir = TEST_DIR / class_name

        for image_path in class_dir.iterdir():

            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in image_extensions:
                continue

            image_paths.append(image_path)
            true_labels.append(
                class_to_index[class_name]
            )

    return image_paths, np.array(true_labels), class_names


def evaluate_model(model, image_paths):

    predicted_labels = []

    total_images = len(image_paths)

    print(f"\nRunning predictions on {total_images} images...")

    for index in range(0, total_images, BATCH_SIZE):

        batch_paths = image_paths[
            index:index + BATCH_SIZE
        ]

        results = model.predict(
            source=[
                str(path)
                for path in batch_paths
            ],
            imgsz=IMAGE_SIZE,
            verbose=False
        )

        for result in results:

            predicted_class = int(
                result.probs.top1
            )

            predicted_labels.append(
                predicted_class
            )

        processed = min(
            index + BATCH_SIZE,
            total_images
        )

        print(
            f"Processed: {processed}/{total_images}",
            end="\r"
        )

    print()

    return np.array(predicted_labels)


def save_confusion_matrix(
    cm,
    class_names
):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.imshow(cm)

    plt.title(
        "Seat Belt Detection Confusion Matrix"
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "True Label"
    )

    plt.xticks(
        range(len(class_names)),
        class_names,
        rotation=45
    )

    plt.yticks(
        range(len(class_names)),
        class_names
    )

    for i in range(len(class_names)):

        for j in range(len(class_names)):

            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.tight_layout()

    output_path = (
        RESULTS_DIR /
        "confusion_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"\nConfusion matrix saved to:"
    )

    print(output_path)


def save_report(
    model_path,
    accuracy,
    report,
    cm
):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report_path = (
        RESULTS_DIR /
        "classification_report.txt"
    )

    with open(
        report_path,
        "w"
    ) as file:

        file.write(
            "SEAT BELT MODEL EVALUATION\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        file.write(
            f"Model: {model_path.name}\n"
        )

        file.write(
            f"Test Accuracy: {accuracy:.4f}\n"
        )

        file.write(
            f"Test Accuracy: "
            f"{accuracy * 100:.2f}%\n\n"
        )

        file.write(
            "Classification Report:\n"
        )

        file.write(report)

        file.write(
            "\n\nConfusion Matrix:\n"
        )

        file.write(str(cm))

    print(
        "\nEvaluation report saved to:"
    )

    print(report_path)


def main():

    print("=" * 60)

    print(
        "SEAT BELT MODEL EVALUATION"
    )

    print("=" * 60)

    if not TEST_DIR.exists():

        raise FileNotFoundError(
            f"Test dataset not found:\n"
            f"{TEST_DIR}"
        )

    print("\nLoading model...")

    model = load_model()

    print("\nLoading test dataset...")

    image_paths, y_true, class_names = (
        load_test_images()
    )

    print(
        f"Classes: {class_names}"
    )

    print(
        f"Test images: {len(image_paths)}"
    )

    print("\nRunning predictions...")

    y_pred = evaluate_model(
        model,
        image_paths
    )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )

    print("\n" + "=" * 60)

    print(
        "EVALUATION RESULTS"
    )

    print("=" * 60)

    print(
        f"\nTest Accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"Test Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        "\nClassification Report:"
    )

    print(report)

    print(
        "\nConfusion Matrix:"
    )

    print(cm)

    save_report(
        MODEL_PATH,
        accuracy,
        report,
        cm
    )

    save_confusion_matrix(
        cm,
        class_names
    )

    print(
        "\nEvaluation completed."
    )


if __name__ == "__main__":
    main()