import cv2
import math
import cvzone
from ultralytics import YOLO

# Initialize video capture
video_path = "Media/sample2.mp4"
cap = cv2.VideoCapture(video_path)

# Load YOLO model with custom weights
model = YOLO("Weights/best.pt")

# Define class names
classNames = ['Without Helmet', 'With Helmet']

while True:
    success, img = cap.read()
    if not success or img is None:
        print("Failed to read video frame.")
        break

    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            # Debug: Print detected class and confidence
            print(f"Class: {classNames[cls]}, Confidence: {conf}")
            
            if conf > 0.2:  # Confidence threshold
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', 
                                   (max(0, x1), max(35, y1)), scale=1, thickness=1)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
