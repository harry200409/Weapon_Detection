import cv2
import torch
from ultralytics import YOLO
import numpy as np
import os
import json
import time

# Create output directory
os.makedirs('detection_results', exist_ok=True)

# Initialize metrics
metrics = {
    'detection_accuracy': 0.0,
    'average_confidence': 0.0,
    'detection_time': 0.0
}

# Load the YOLOv5 model
print("Loading YOLOv5 model...")
model = YOLO('data/yolov5s.pt')

# Set the model to evaluation mode
model.eval()

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model.to(device)

# Open the webcam
print("Opening camera...")
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Set higher confidence threshold for better accuracy
CONFIDENCE_THRESHOLD = 0.5

# Define knife class ID (43 in COCO dataset)
KNIFE_CLASS_ID = 43

print("Starting knife detection. Press 'q' to quit...")

# Initialize detection statistics
total_frames = 0
knife_detections = 0
total_confidence = 0.0
start_time = time.time()

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    total_frames += 1
    
    # Run YOLOv5 inference
    results = model(frame, conf=CONFIDENCE_THRESHOLD)
    
    # Get the first result (since we're processing one frame at a time)
    result = results[0]
    
    # Get bounding boxes, confidence scores, and class IDs
    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    
    # Track knife detections in this frame
    frame_knife_detections = 0
    
    # Draw bounding boxes and labels only for knives
    for box, conf, class_id in zip(boxes, confidences, class_ids):
        if class_id == KNIFE_CLASS_ID:
            frame_knife_detections += 1
            total_confidence += float(conf)  # Convert numpy.float32 to Python float
            x1, y1, x2, y2 = map(int, box)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Create text label with confidence
            label = f'Knife: {conf:.2f}'
            
            # Calculate text size
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Draw semi-transparent background for text
            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 0, 255), -1)
            
            # Draw text
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    if frame_knife_detections > 0:
        knife_detections += 1
        # Save frame with detection
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        cv2.imwrite(f'detection_results/knife_detection_{timestamp}.jpg', frame)

    # Display the frame
    cv2.imshow('Knife Detection', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calculate metrics
end_time = time.time()
detection_time = float(end_time - start_time)  # Convert to Python float

if total_frames > 0:
    metrics['detection_accuracy'] = float(knife_detections / total_frames)  # Convert to Python float
if knife_detections > 0:
    metrics['average_confidence'] = float(total_confidence / knife_detections)  # Convert to Python float
metrics['detection_time'] = detection_time

# Save metrics
with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

# Release resources
print("Closing camera and cleaning up...")
cap.release()
cv2.destroyAllWindows() 