import cv2
import time

from src.camera import find_available_cameras, open_camera
from src.person_detector import PersonDetector
from src.seatbelt_detector import SeatbeltDetector


WINDOW_NAME = "Seat Belt Detection System"

PERSON_CONFIDENCE = 0.5
SEATBELT_CONFIDENCE = 0.5


def draw_person_result(
    frame,
    box,
    person_confidence,
    seatbelt_result
):
    x1, y1, x2, y2 = box

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    person_label = (
        f"PERSON {person_confidence * 100:.1f}%"
    )

    cv2.putText(
        frame,
        person_label,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    if seatbelt_result is None:
        seatbelt_label = "SEAT BELT: UNKNOWN"
        seatbelt_color = (0, 255, 255)

    else:
        class_name = seatbelt_result["class"]
        confidence = seatbelt_result["confidence"]

        seatbelt_label = (
            f"SEAT BELT: {class_name.upper()} "
            f"{confidence * 100:.1f}%"
        )

        if class_name.lower() == "with":
            seatbelt_color = (0, 255, 0)
        else:
            seatbelt_color = (0, 0, 255)

    cv2.putText(
        frame,
        seatbelt_label,
        (x1, min(y2 + 25, frame.shape[0] - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        seatbelt_color,
        2
    )


def select_camera():

    cameras = find_available_cameras()

    if not cameras:
        raise RuntimeError(
            "No available cameras were found."
        )

    print("\nAvailable cameras:")

    for index, camera_index in enumerate(cameras, start=1):
        print(f"{index}. Camera {camera_index}")

    while True:
        try:
            choice = int(
                input("\nSelect camera: ")
            )

            if 1 <= choice <= len(cameras):
                return cameras[choice - 1]

            print("Invalid selection.")

        except ValueError:
            print("Enter a valid number.")


def main():

    print("=" * 60)
    print("SEAT BELT DETECTION SYSTEM")
    print("=" * 60)

    print("\nLoading person detector...")

    person_detector = PersonDetector(
        confidence=PERSON_CONFIDENCE
    )

    print("\nLoading seat-belt detector...")

    seatbelt_detector = SeatbeltDetector(
        confidence=SEATBELT_CONFIDENCE
    )

    print("\nAll models loaded successfully.")

    camera_index = select_camera()

    camera = open_camera(camera_index)

    print("\nStarting live detection...")
    print("Press Q or ESC inside the camera window to exit.")

    previous_time = time.time()
    fps = 0

    while True:

        success, frame = camera.read()

        if not success:
            print(
                "Failed to read frame from camera."
            )
            break

        detections = person_detector.detect(frame)

        for detection in detections:

            box = detection["box"]
            person_confidence = detection["confidence"]

            x1, y1, x2, y2 = box

            height, width = frame.shape[:2]

            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))

            if x2 <= x1 or y2 <= y1:
                continue

            person_crop = frame[
                y1:y2,
                x1:x2
            ]

            if person_crop.size == 0:
                continue

            seatbelt_result = (
                seatbelt_detector.predict(
                    person_crop
                )
            )

            draw_person_result(
                frame,
                box,
                person_confidence,
                seatbelt_result
            )

        current_time = time.time()

        elapsed_time = (
            current_time - previous_time
        )

        if elapsed_time > 0:
            fps = 1 / elapsed_time

        previous_time = current_time

        fps_text = f"FPS: {fps:.1f}"

        cv2.putText(
            frame,
            fps_text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            WINDOW_NAME,
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == ord("Q"):
            break

        if key == 27:
            break

    camera.release()
    cv2.destroyAllWindows()

    print("\nDetection system stopped.")
    print("Camera released successfully.")


if __name__ == "__main__":
    main()