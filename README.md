# Object-Detection-System
This project is an Object Detection system using OpenCV's Deep Neural Network (DNN) module with a pre-trained SSD (Single Shot Multibox Detector) MobileNet model. The system detects and identifies objects in a video file, highlighting each detected object with a bounding box and label.

https://github.com/YashPise/Object-Detection-System/assets/99723328/ebfae052-8941-454e-8d4e-56ed484d2cd4


https://github.com/YashPise/Object-Detection-System/assets/99723328/4f5d0d93-cfae-49ff-a182-1cb4f5128e5b

Different Objects detected
## Overview

This repository contains an Object Detection System that utilizes advanced computer vision techniques to identify and locate objects within images or videos. The system is built using Python and leverages frameworks such as OpenCV and TensorFlow for real-time object detection.

## Features

- **Real-Time Detection**: Detects objects in real-time from webcam feed or video files.
- **Multiple Object Detection**: Identifies and classifies various objects within a single frame.
- **Pre-trained Models**: Uses pre-trained models like YOLO, SSD, and Faster R-CNN for accurate detection.
- **Custom Object Training**: Ability to train custom object detection models.
- **Visualization**: Draws bounding boxes and labels around detected objects.

## Getting Started

### Prerequisites

- Python 3.x
- OpenCV
- TensorFlow
- Numpy

### Installation

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/YashPise/Object-Detection-System.git
    cd Object-Detection-System
    ```

2. **Install Dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

### Usage

1. **Run the Object Detection Script**:
    ```sh
    python src/object_detection.py
    ```

2. **Options**:
    - **Webcam**: Detect objects from webcam feed.
    - **Video File**: Detect objects from a video file by specifying the file path.
    - **Image File**: Detect objects in a single image by specifying the image file path.

## Folder Structure

- `src/`: Contains the source code for the object detection system.
  - `object_detection.py`: Main script to run the object detection.
  - `utils.py`: Utility functions for processing.
- `models/`: Pre-trained models and configurations.
- `data/`: Dataset for training custom models.
- `results/`: Output images and videos with detected objects.
- `README.md`: Project overview and instructions.
- `requirements.txt`: List of dependencies.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.








