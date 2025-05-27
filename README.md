# Weapon Detection System

A deep learning-based system for detecting weapons (guns and knives) in images and video streams using YOLOv5 and YOLOv8.

## Features
- Real-time weapon detection
- Support for multiple weapon types:
  - Guns
  - Knives
- Uses state-of-the-art object detection models:
  - YOLOv5
  - YOLOv8

## Project Structure

```text
├── gun/                # Gun detection related files
│   ├── dataset/        # Training and testing datasets for guns
│   ├── yolov5/         # YOLOv5 implementation
│   └── app.py          # Gun detection application
├── knife/              # Knife detection related files
├── data/               # Model weights and configurations
├── gun_detection.py    # Main gun detection script
├── knife_detection.py  # Main knife detection script
└── requirements.txt    # Project dependencies
```


## Prerequisites
- Python
- CUDA-compatible GPU (recommended for faster inference)
- Install dependencies listed in `requirements.txt`

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd <repository-folder>
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Usage**:

  **Knife Detection**:
  Run the knife detection script:
```bash
python knife_detection.py
```

  **Gun Detection**:
  Run the gun detection script:
```bash
python gun_detection.py
```

  ## Dataset

  The project uses custom datasets for both gun and knife detection. The datasets should follow this structure:

```bash
dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

  **Images**: .jpg or .png

  **Labels**: YOLO format .txt

  ## Models

  **YOLOv5s**: Lightweight and fast version of YOLOv5.
  
  **YOLOv8n**: Ultralytics' latest model with improved performance and speed.
  
  Model weights should be stored in the data/ directory.


  ## Acknowledgments
  YOLOv5 by Ultralytics
  
  YOLOv8 by Ultralytics
  
  Dataset contributors and open-source community
