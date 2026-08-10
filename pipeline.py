from pathlib import Path
import shutil
import subprocess
import sys


BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"

RUNS_DIR = BASE_DIR / "runs"
MODELS_DIR = BASE_DIR / "models"

STAGES = [
    ("Dataset Cleaning", "clean_dataset.py"),
    ("Dataset Analysis", "dataset_analysis.py"),
    ("Dataset Numbering", "number_dataset.py"),
    ("Train / Validation / Test Split", "split_dataset.py"),
    ("YOLO11 Classification Training", "train_classifier.py"),
    ("Model Evaluation", "evaluate_model.py"),
]


def print_header(title):

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def run_script(script_name):

    script_path = SRC_DIR / script_name

    if not script_path.exists():

        print("\nERROR: Script not found:")
        print(script_path)

        return False

    print_header(f"RUNNING: {script_name}")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=BASE_DIR
    )

    if result.returncode != 0:

        print("\nERROR: Stage failed.")
        print(f"Script: {script_name}")
        print(f"Exit code: {result.returncode}")

        return False

    print(f"\nCOMPLETED: {script_name}")

    return True


def ask_video_extraction():

    while True:

        choice = input(
            "\nDo you want to extract images from videos? (y/n): "
        ).strip().lower()

        if choice in ["y", "yes"]:
            return True

        if choice in ["n", "no"]:
            return False

        print("Please enter y or n.")


def find_latest_trained_model():

    if not RUNS_DIR.exists():
        return None

    model_files = list(
        RUNS_DIR.glob(
            "seat_belt_classifier*/weights/best.pt"
        )
    )

    if not model_files:
        return None

    model_files.sort(
        key=lambda path: path.stat().st_mtime,
        reverse=True
    )

    return model_files[0]


def save_trained_model():

    print_header("SAVING TRAINED MODEL")

    best_model = find_latest_trained_model()

    if best_model is None:

        print("ERROR: Trained best.pt model was not found.")

        print("\nExpected location:")
        print(
            RUNS_DIR
            / "seat_belt_classifier-X"
            / "weights"
            / "best.pt"
        )

        return False

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    destination = MODELS_DIR / "best.pt"

    shutil.copy2(
        best_model,
        destination
    )

    print(f"Source model:")
    print(best_model)

    print("\nModel copied to:")
    print(destination)

    return True


def main():

    print_header(
        "SEAT BELT DETECTION MASTER PIPELINE"
    )

    print("Project:")
    print(BASE_DIR)

    print("\nPython:")
    print(sys.executable)

    print("\nPipeline stages:")

    print("1. Video -> Images")
    print("2. Dataset Cleaning")
    print("3. Dataset Analysis")
    print("4. Dataset Numbering")
    print("5. Train / Validation / Test Split")
    print("6. YOLO11 Classification Training")
    print("7. Save best.pt to models/")
    print("8. Model Evaluation")

    print("\n" + "=" * 60)

    # -----------------------------------------------------
    # VIDEO TO IMAGE
    # -----------------------------------------------------

    extract_videos = ask_video_extraction()

    if extract_videos:

        success = run_script(
            "video_to_images.py"
        )

        if not success:

            print("\nPipeline stopped.")
            return

    else:

        print("\nVideo extraction skipped.")
        print("Using existing images dataset.")

    # -----------------------------------------------------
    # DATASET AND MODEL PIPELINE
    # -----------------------------------------------------

    for stage_name, script_name in STAGES:

        print_header(stage_name)

        success = run_script(
            script_name
        )

        if not success:

            print("\n" + "=" * 60)
            print("PIPELINE STOPPED")
            print("=" * 60)

            print(
                f"\nThe pipeline stopped at: {stage_name}"
            )

            print(
                f"Script: {script_name}"
            )

            return

        # Save trained model immediately after training
        if script_name == "train_classifier.py":

            success = save_trained_model()

            if not success:

                print("\n" + "=" * 60)
                print("PIPELINE STOPPED")
                print("=" * 60)

                print(
                    "\nThe trained model could not be saved."
                )

                return

    # -----------------------------------------------------
    # COMPLETED
    # -----------------------------------------------------

    print_header(
        "MASTER PIPELINE COMPLETED"
    )

    print("Dataset preparation completed.")
    print("Model training completed.")
    print("Best model saved.")
    print("Model evaluation completed.")

    print("\nModel location:")
    print(MODELS_DIR / "best.pt")

    print("\nYou can now run the live detection system using:")

    print("python main.py")


if __name__ == "__main__":
    main()