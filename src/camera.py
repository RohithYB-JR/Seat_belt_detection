import cv2


def find_available_cameras(max_cameras=5):
    available_cameras = []

    print("=" * 50)
    print("CHECKING AVAILABLE CAMERAS")
    print("=" * 50)

    for camera_index in range(max_cameras):
        camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

        if camera.isOpened():
            available_cameras.append(camera_index)
            print(f"Camera {camera_index}: Available")

        camera.release()

    if not available_cameras:
        print("No cameras found.")

    return available_cameras


def open_camera(camera_index):
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if not camera.isOpened():
        raise RuntimeError(
            f"Could not open camera {camera_index}."
        )

    print(f"\nCamera {camera_index} opened successfully.")
    print("Press Q to close the camera.")

    return camera


def run_camera(camera_index):
    camera = open_camera(camera_index)

    while True:
        success, frame = camera.read()

        if not success:
            print("Failed to read frame from camera.")
            break

        cv2.imshow("Seat Belt Detection Camera", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == ord("Q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    print("Camera closed.")


def main():
    cameras = find_available_cameras()

    if not cameras:
        return

    print("\nAvailable cameras:")

    for index, camera_index in enumerate(cameras, start=1):
        print(f"{index}. Camera {camera_index}")

    while True:
        try:
            choice = int(input("\nSelect camera: "))

            if 1 <= choice <= len(cameras):
                camera_index = cameras[choice - 1]
                break

            print("Invalid selection.")

        except ValueError:
            print("Enter a valid number.")

    run_camera(camera_index)


if __name__ == "__main__":
    main()