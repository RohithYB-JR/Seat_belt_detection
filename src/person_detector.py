from pathlib import Path

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "yolo11n.pt"

PERSON_CLASS_ID = 0
DEFAULT_CONFIDENCE = 0.5


class PersonDetector:

    def __init__(
        self,
        confidence=DEFAULT_CONFIDENCE
    ):
        print("\nLoading YOLO11 person detection model...")

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Person detection model not found:\n{MODEL_PATH}"
            )

        print(f"Model: {MODEL_PATH}")

        self.model = YOLO(str(MODEL_PATH))
        self.confidence = confidence

        print("Person detector loaded successfully.")

    def detect(self, frame):

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            verbose=False
        )

        detections = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                class_id = int(
                    box.cls[0].item()
                )

                if class_id != PERSON_CLASS_ID:
                    continue

                confidence = float(
                    box.conf[0].item()
                )

                coordinates = box.xyxy[0].tolist()

                x1, y1, x2, y2 = map(
                    int,
                    coordinates
                )

                detections.append(
                    {
                        "box": (
                            x1,
                            y1,
                            x2,
                            y2
                        ),
                        "confidence": confidence
                    }
                )

        return detections


def main():

    print("=" * 50)
    print("PERSON DETECTOR TEST")
    print("=" * 50)

    detector = PersonDetector()

    print("\nThe person detector is ready for integration.")
    print("Model loaded from:")
    print(MODEL_PATH)


if __name__ == "__main__":
    main()