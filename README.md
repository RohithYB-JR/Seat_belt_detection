# Seat Belt Detection System

A modular and reusable seat-belt detection system built using Python, OpenCV, and YOLO11.

The system is designed to detect people in camera frames and classify whether a seat belt is being worn or not. It also provides a complete dataset preparation and model training pipeline so that users can train the system using their own datasets.

The project supports both direct image datasets and video-based datasets. Videos containing "with seat belt" and "without seat belt" examples can be automatically converted into images before continuing through the dataset preparation and training pipeline.

The system can be used for:

- Live camera-based seat-belt detection
- Custom dataset training
- Dataset cleaning and analysis
- Automatic dataset numbering
- Train, validation, and test splitting
- YOLO11 classification training
- Model evaluation
- Real-time inference
- Future integration with embedded and IoT systems

## Features

- Real-time seat-belt detection using a camera
- Person detection using YOLO11
- Seat-belt classification using a trained YOLO11 classification model
- Bounding boxes for detected persons
- Seat-belt status displayed as `WITH` or `WITHOUT`
- Confidence score displayed for person and seat-belt detection
- Support for multiple available cameras
- Direct image dataset support
- Optional video-to-image dataset extraction
- Automatic dataset cleaning
- Dataset analysis
- Automatic image numbering
- Automatic train, validation, and test splitting
- YOLO11 classification model training
- Model evaluation with accuracy, classification report, and confusion matrix
- Modular Python source files
- Master pipeline for automated dataset preparation and model training
- Reusable workflow for custom datasets
- Designed for future embedded and IoT integration

## How It Works

The system has two main workflows.

### 1. Model Training Workflow

A user can provide either images or videos containing two classes:

- `with` — person wearing a seat belt
- `without` — person not wearing a seat belt

If videos are provided, the optional video-to-image stage extracts frames from the videos.

The images then pass through the following stages:

```text
Videos (optional)
       |
       v
Video -> Image Extraction
       |
       v
Dataset Cleaning
       |
       v
Dataset Analysis
       |
       v
Dataset Numbering
       |
       v
Train / Validation / Test Split
       |
       v
YOLO11 Classification Training
       |
       v
Trained Model
       |
       v
Model Evaluation

## Project Structure

```text
Seat_belt_detection/
│
├── main.py
├── pipeline.py
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── datasets/
│   ├── images/
│   │   ├── with/
│   │   └── without/
│   │
│   ├── videos/
│   │   ├── with/
│   │   └── without/
│   │
│   └── classification/
│       ├── train/
│       │   ├── with/
│       │   └── without/
│       ├── val/
│       │   ├── with/
│       │   └── without/
│       └── test/
│           ├── with/
│           └── without/
│
├── models/
│   └── best.pt
│
├── runs/
│   ├── seat_belt_classifier/
│   └── evaluation/
│
└── src/
    ├── camera.py
    ├── video_to_images.py
    ├── clean_dataset.py
    ├── dataset_analysis.py
    ├── number_dataset.py
    ├── split_dataset.py
    ├── train_classifier.py
    ├── evaluate_model.py
    ├── person_detector.py
    └── seatbelt_detector.py

## Requirements

### Software

- Python 3.10 or newer
- pip
- Git
- OpenCV
- Ultralytics YOLO11
- TensorFlow
- NumPy
- scikit-learn
- Matplotlib
- Pillow

### Hardware

The project can run on a CPU-only system, but training and real-time inference performance depends heavily on the hardware.

#### Minimum

```text
RAM     : 8 GB
Storage : 10 GB+ recommended
CPU     : Multi-core processor
Camera  : USB or built-in camera
GPU     : Not required

#### Recommendation

```text
RAM     : 16 GB+
Storage : SSD
CPU     : Modern multi-core processor
GPU     : NVIDIA GPU with CUDA support
Camera  : USB or built-in camera

For embedded or IoT deployment, a GPU-enabled device such as an NVIDIA Jetson-class platform is more suitable for real-time inference than a low-power CPU-only system.

---

## Installation

### 1. Clone the Repository

Clone the repository using Git:

```bash
git clone <repository-url>

### 2. Create a Virtual Environment

Create a Python virtual environment:

python -m venv .venv

### 3. Activate the Virtual Environment

Windows PowerShell

.\.venv\Scripts\Activate.ps1

Windows Command Prompt

.venv\Scripts\activate

### 4. Install Dependencies

Install all required Python packages:

pip install -r requirements.txt

### 5. Verify the Installation

Check the Python version:

python --version

Check the Ultralytics installation:

python -c "import ultralytics; print(ultralytics.__version__)"

Check OpenCV:

python -c "import cv2; print(cv2.__version__)"

Check PyTorch:

python -c "import torch; print(torch.__version__)"

The project is now ready for dataset preparation, model training, evaluation, and live inference.

## Dataset Structure

The project supports two ways of providing a dataset:

1. Direct images
2. Videos that can be converted into images

The dataset contains two classes:

- `with` — person wearing a seat belt
- `without` — person not wearing a seat belt

### Option 1: Using Images

Place the images into:

```text
datasets/
└── images/
    ├── with/
    └── without/

### Option 2: Using Videos

If the user has videos instead of individual images, they can place the videos into:

```text
datasets/
└── videos/
    ├── with/
    └── without/

The videos are then converted into individual images by the video-to-image extraction stage.

The extracted images are placed into the corresponding image class:

datasets/videos/with/
        |
        v
Video-to-Image Extraction
        |
        v
datasets/images/with/

and:

datasets/videos/without/
        |
        v
Video-to-Image Extraction
        |
        v
datasets/images/without/

This allows users to build an image dataset directly from their recorded videos.

Selecting the Dataset Input Method

When the master pipeline is started:

python pipeline.py

the user is asked:

Do you want to extract images from videos? (y/n):

If the user enters:

y

the video-to-image extraction stage is executed.

If the user enters:

n

the video extraction stage is skipped and the existing image dataset is used.

Dataset Preparation Pipeline

After the image dataset is available, it passes through the following stages:

Image Dataset
      |
      v
Dataset Cleaning
      |
      v
Dataset Analysis
      |
      v
Dataset Numbering
      |
      v
Train / Validation / Test Split

The final classification dataset is organized as:

```text
datasets/
└── classification/
    ├── train/
    │   ├── with/
    │   └── without/
    │
    ├── val/
    │   ├── with/
    │   └── without/
    │
    └── test/
        ├── with/
        └── without/

This classification dataset is used by the YOLO11 classification training and evaluation stages.

## Master Pipeline

The project includes a master pipeline that automates the complete dataset preparation, model training, and evaluation workflow.

The pipeline can be started from the project root using:

```bash
python pipeline.py

## Model Training

The project uses YOLO11 classification for seat-belt classification.

The training script is:

```text
src/train_classifier.py

## Model Evaluation

After training, the model can be evaluated using the test dataset.

The evaluation script is:

```text
src/evaluate_model.py

## Live Camera Inference

The live detection system uses a camera to detect people and classify their seat-belt status in real time.

The main application is:

```text
main.py

## Performance and Hardware

The system supports CPU-only execution, but inference speed depends on the processor, image resolution, model size, and number of models being executed.

### CPU-Only Execution

The system can run without an NVIDIA GPU.

On low-power or older CPUs, live detection may produce a low FPS because the application performs multiple inference operations for each camera frame:

```text
Camera Frame
     |
     v
Person Detection
     |
     v
Person Crop
     |
     v
Seat-Belt Classification

### GPU Acceleration

For higher real-time performance, an NVIDIA GPU with CUDA support is recommended.

A GPU can accelerate the neural-network inference operations and provide substantially higher FPS compared with CPU-only execution.

## Model and File Management

The project uses multiple model files during training and inference. These files have different purposes and should not be treated as the same model.

### Trained Seat-Belt Model

The trained seat-belt classification model is stored as:

```text
models/
└── best.pt

## Running the Complete Project

The project can be used in two main ways:

1. Train the system using a custom dataset
2. Run the trained model for live camera detection

### Complete Training Workflow

After installing the requirements and preparing the dataset, run the master pipeline from the project root:

```bash
python pipeline.py

## Customization and Configuration

The project is designed to allow users to modify the dataset, training configuration, detection confidence, and other settings according to their requirements.

### Dataset Classes

The current classifier uses two classes:

```text
with
without

## Limitations

The current version of the project is a functional prototype and has some limitations.

### CPU Performance

The system can run on CPU-only hardware, but live inference performance can be low on older or low-power processors.

The live pipeline performs person detection followed by seat-belt classification, which requires two inference operations for each frame.

### Detection Accuracy

The model's performance depends heavily on the quality and diversity of the training dataset.

Performance may decrease when the system encounters conditions that are not sufficiently represented in the training data, such as:

- Different vehicle interiors
- Different camera positions
- Poor lighting
- Night-time conditions
- Occlusion
- Unusual seating positions
- Loose or partially visible seat belts
- Different clothing styles
- Low-resolution camera feeds

### Seat-Belt Classification

The current seat-belt model performs classification on the detected person's image crop.

It determines whether the detected person belongs to the `with` or `without` class.

The current implementation does not independently detect and draw a separate bounding box around the physical seat belt.

### Camera Conditions

The system is currently designed around a camera feed where the person is sufficiently visible.

Extreme camera angles, heavy occlusion, motion blur, or poor lighting can reduce detection performance.

### Hardware Dependency

Real-time performance depends on the hardware used for deployment.

A CPU-only system may provide significantly lower FPS than a GPU-accelerated system.

---

## Future Improvements

The project is designed to be extended beyond the current prototype.

### Improved Real-Time Performance

Future versions can improve inference speed through:

- GPU acceleration
- Model optimization
- Reduced input resolution
- Frame skipping
- Asynchronous inference
- Model quantization
- Hardware-specific optimization

### Improved Seat-Belt Detection

Future versions can use a dedicated seat-belt object-detection model instead of only classifying the person crop.

This could allow the system to directly detect the physical seat belt and display a bounding box around it.

A possible future architecture is:

```text
Camera
   |
   v
Person Detection
   |
   v
Seat-Belt Detection
   |
   v
Seat-Belt Status

## Project Status

The current version of the project includes:

- Dataset preparation pipeline
- Optional video-to-image extraction
- Dataset cleaning
- Dataset analysis
- Dataset numbering
- Train / validation / test splitting
- YOLO11 classification training
- Model evaluation
- Person detection
- Seat-belt classification
- Camera selection
- Live camera inference
- Person bounding boxes
- Seat-belt status and confidence display
- Master pipeline automation

### Current Status

The project is currently in the prototype and development stage.

The complete workflow from dataset preparation to live camera inference is functional.

Future development will focus on:

- Improving inference speed
- Improving detection accuracy
- Dedicated seat-belt object detection
- Multiple-person optimization
- Embedded deployment
- IoT integration
- Real-time alerts and event logging
- Hardware-specific model optimization

## License

This project is licensed under the MIT License.

See the [LICENSE](LICENSE) file for the complete license text.

The MIT License permits users to use, copy, modify, merge, publish, distribute, sublicense, and sell copies of the software, subject to the conditions specified in the license.

## Acknowledgements

This project uses and builds upon several open-source technologies and libraries.

### Technologies Used

- Python
- OpenCV
- Ultralytics YOLO11
- PyTorch
- TensorFlow
- NumPy
- scikit-learn
- Matplotlib
- Pillow

### Ultralytics YOLO

YOLO11 from Ultralytics is used for:

- Person detection
- Seat-belt classification

The pretrained YOLO11 model is used for person detection, while a custom-trained YOLO11 classification model is used for seat-belt classification.

More information about Ultralytics YOLO can be found in the official documentation.

### OpenCV

OpenCV is used for:

- Camera access
- Video frame processing
- Image processing
- Drawing detection results
- Displaying the live camera feed

### Python Libraries

Other Python libraries are used for dataset processing, model evaluation, numerical operations, and visualization.
