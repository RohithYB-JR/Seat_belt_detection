from pathlib import Path

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best.pt"

DEFAULT_CONFIDENCE = 0.5


class SeatbeltDetector:

    def __init__(
        self,
        confidence=DEFAULT_CONFIDENCE
    ):
        print("\nLoading seat-belt classification model...")

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Seat-belt model not found:\n{MODEL_PATH}"
            )

        print(f"Model: {MODEL_PATH}")

        self.model = YOLO(str(MODEL_PATH))
        self.confidence = confidence

        print("Seat-belt classifier loaded successfully.")

    def predict(self, image):

        results = self.model.predict(
            source=image,
            conf=self.confidence,
            verbose=False
        )

        if not results:
            return None

        result = results[0]

        if result.probs is None:
            return None

        class_id = int(
            result.probs.top1
        )

        confidence = float(
            result.probs.top1conf
        )

        class_name = result.names[class_id]

        return {
            "class": class_name,
            "confidence": confidence
        }


def main():

    print("=" * 50)
    print("SEAT BELT DETECTOR TEST")
    print("=" * 50)

    detector = SeatbeltDetector()

    print("\nSeat-belt detector loaded successfully.")
    print("Ready for image/frame prediction.")


if __name__ == "__main__":
    main()