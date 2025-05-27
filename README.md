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
<pre lang="text"><code>```text . ├── gun/ # Gun detection related files │ ├── dataset/ # Training and testing datasets for guns │ ├── yolov5/ # YOLOv5 implementation │ └── app.py # Gun detection application ├── knife/ # Knife detection related files ├── data/ # Model weights and configurations ├── gun_detection.py # Main gun detection script ├── knife_detection.py # Main knife detection script └── requirements.txt # Project dependencies ``` </code></pre>
