import cv2
from ultralytics import YOLO

custom_model = YOLO("models\\yolov8n_custom.pt")
default_model = YOLO("models\\yolov8n.pt")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results_custom = custom_model.predict(source=frame, conf=0.4, stream=False, verbose=False)[0]
    results_default = default_model.predict(source=frame, conf=0.4, stream=False, verbose=False)[0]

    # Annotate original frame with both results
    frame_combined = frame.copy()
    frame_combined = results_custom.plot(img=frame_combined)
    frame_combined = results_default.plot(img=frame_combined)

    # Show or save the frame
    cv2.imshow("Combined Detection", frame_combined)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
