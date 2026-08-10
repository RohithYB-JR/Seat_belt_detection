from pathlib import Path
import cv2


BASE_DIR = Path(__file__).resolve().parent.parent

VIDEOS_DIR = BASE_DIR / "datasets" / "videos"
IMAGES_DIR = BASE_DIR / "datasets" / "images"

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm"
}

# Number of frames to extract per second.
# Change this value when needed.
EXTRACTION_FPS = 5


def extract_video(video_path, output_dir):
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        print(f"ERROR: Could not open {video_path.name}")
        return 0

    original_fps = capture.get(cv2.CAP_PROP_FPS)

    if original_fps <= 0:
        print(f"ERROR: Could not determine FPS: {video_path.name}")
        capture.release()
        return 0

    frame_interval = max(
        1,
        round(original_fps / EXTRACTION_FPS)
    )

    frame_number = 0
    saved_count = 0

    while True:
        success, frame = capture.read()

        if not success:
            break

        if frame_number % frame_interval == 0:

            filename = (
                f"{video_path.stem}_"
                f"{saved_count + 1:06d}.jpg"
            )

            output_path = output_dir / filename

            cv2.imwrite(
                str(output_path),
                frame
            )

            saved_count += 1

        frame_number += 1

    capture.release()

    return saved_count


def process_class(class_name):

    video_dir = VIDEOS_DIR / class_name
    image_dir = IMAGES_DIR / class_name

    image_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if not video_dir.exists():
        print(f"\nNo video folder found: {video_dir}")
        return

    videos = [
        file
        for file in video_dir.iterdir()
        if file.is_file()
        and file.suffix.lower() in VIDEO_EXTENSIONS
    ]

    if not videos:
        print(f"\nNo videos found in: {video_dir}")
        return

    print("\n" + "=" * 60)
    print(f"PROCESSING CLASS: {class_name}")
    print("=" * 60)

    total_frames = 0

    for video in videos:

        print(f"\nProcessing: {video.name}")

        frames = extract_video(
            video,
            image_dir
        )

        print(f"Frames extracted: {frames}")

        total_frames += frames

    print(f"\nTotal frames extracted: {total_frames}")


def main():

    print("=" * 60)
    print("VIDEO TO IMAGE EXTRACTION")
    print("=" * 60)

    print(f"\nExtraction FPS: {EXTRACTION_FPS}")

    process_class("with")
    process_class("without")

    print("\n" + "=" * 60)
    print("VIDEO EXTRACTION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()