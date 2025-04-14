from ultralytics import YOLO
import cv2

# Load models
custom_model = YOLO("models\\yolov8n_custom.pt")
pretrained_model = YOLO("models\\yolov8n.pt")

# Load video
video_path = "D:\\files\\projects\\ai\\walkingStick\\VID_20250406_143500.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Could not open video.")
    exit()

COLOR_CUSTOM = (0, 255, 0)       # Green for custom model
COLOR_PRETRAINED = (0, 0, 255)   # Red for pretrained model

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results_custom = custom_model.predict(frame, conf=0.35, imgsz=640, stream=False)[0]
    results_pretrained = pretrained_model.predict(frame, conf=0.35, imgsz=640, stream=False)[0]

    # Draw custom model results
    for box in results_custom.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = custom_model.names[cls]
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_CUSTOM, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_CUSTOM, 1)

    # Draw pretrained model results
    for box in results_pretrained.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = pretrained_model.names[cls]
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PRETRAINED, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_PRETRAINED, 1)

    # Show the result
    cv2.imshow("Combined Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
