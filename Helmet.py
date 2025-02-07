import numpy as np
import argparse
import imutils
import time
import cv2
import os

# Argument parser to handle command-line inputs
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="", 
                help="Path to input video file. Leave blank for webcam.")
ap.add_argument("-o", "--output", required=True, 
                help="Path to save the output video.")
ap.add_argument("-y", "--yolo", required=True, 
                help="Base path to YOLO directory (contains .cfg, .weights, .names).")
ap.add_argument("-c", "--confidence", type=float, default=0.5, 
                help="Minimum confidence to filter weak detections.")
ap.add_argument("-t", "--threshold", type=float, default=0.3, 
                help="Threshold for non-maxima suppression.")
args = vars(ap.parse_args())

# Load YOLO files
labelsPath = os.path.sep.join([args["yolo"], "cocohelmet.names"])
weightsPath = os.path.sep.join([args["yolo"], "yolov3-obj_2400.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3-obj.cfg"])

# Verify YOLO files
if not os.path.exists(labelsPath):
    raise FileNotFoundError(f"Class labels file not found: {labelsPath}")
if not os.path.exists(weightsPath):
    raise FileNotFoundError(f"Weights file not found: {weightsPath}")
if not os.path.exists(configPath):
    raise FileNotFoundError(f"Configuration file not found: {configPath}")

# Load labels, YOLO model
print("[INFO] Loading YOLO model...")
LABELS = open(labelsPath).read().strip().split("\n")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Initialize video stream and output writer
print("[INFO] Starting video stream...")
if args["input"] == "":
    vs = cv2.VideoCapture(0)  # Webcam
else:
    if not os.path.exists(args["input"]):
        raise FileNotFoundError(f"Input video file not found: {args['input']}")
    vs = cv2.VideoCapture(args["input"])  # Video file

writer = None
(W, H) = (None, None)

# Process each frame
while True:
    grabbed, frame = vs.read()
    if not grabbed:
        break

    # Resize frame for faster processing
    frame = imutils.resize(frame, width=700)
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # Create a blob and perform forward pass
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # Process YOLO outputs
    boxes, confidences, classIDs = [], [], []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > args["confidence"]:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Apply non-maxima suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

    # Draw detections
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in np.random.randint(0, 255, size=(3,), dtype="uint8")]
            text = f"{LABELS[classIDs[i]]}: {confidences[i]:.2f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Write the output frame
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(args["output"], fourcc, 30, (frame.shape[1], frame.shape[0]))
    writer.write(frame)

    # Show the frame (optional)
    cv2.imshow("Helmet Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video stream and writer
print("[INFO] Cleaning up...")
writer.release()
vs.release()
cv2.destroyAllWindows()
