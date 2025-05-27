import cv2
from ultralytics import YOLO

# Load YOLO model
# model = YOLO(r"C:/gun/runs/detect/train9/weights/best.pt")
model = YOLO("runs/detect/train9/weights/best.pt")

# Open webcam
cap = cv2.VideoCapture(0)
CONFIDENCE_THRESHOLD = 0.6

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)[0]

    # Loop through detections
    for box in results.boxes:
        cls_id = int(box.cls.cpu().numpy())
        conf = float(box.conf.cpu().numpy())
        class_name = model.names[cls_id]

        if conf > CONFIDENCE_THRESHOLD and class_name.lower() == "gun":
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            label = f"{class_name} ({conf:.2f})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Gun Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()

# import cv2
# cap = cv2.VideoCapture(0)
# ret, frame = cap.read()
# cv2.imshow("Test", frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cap.release()
