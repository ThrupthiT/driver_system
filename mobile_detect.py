#DETECTION OF MOBILE PHONE USING YOLO

import cv2
import pyttsx3
import numpy as np

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

# Load YOLOv3 model files
yolo_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]

# Load COCO class labels (YOLO's trained dataset)
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Start capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Webcam not available.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 480))  # Resize for faster processing

    # Object detection using YOLOv3
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    detections = yolo_net.forward(output_layers)

    detected_phone = False  # Flag to track if a phone is detected

    for out in detections:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and classes[class_id] == "cell phone":  # Adjust confidence threshold if needed
                # Get the bounding box for the mobile phone
                center_x, center_y, w, h = (detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])).astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Draw the bounding box and label on the frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Mobile Phone Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # Announce through speech
                if not detected_phone:
                    engine.say("Mobile phone detected")
                    engine.runAndWait()
                    detected_phone = True
                break  # Stop after detecting the first phone

    # Display the video feed
    cv2.imshow("Mobile Phone Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
