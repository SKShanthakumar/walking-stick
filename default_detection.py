import os
import cv2
import time
from ultralytics import YOLO
import pygame

# audio setup
audio_dir = "audio_files"
audio_map = {
    "person": "",
    "bicycle": "",
    "car": "",
    "motorcycle": "",
    "airplane": "",
    "bus": "",
    "train": "",
    "truck": "",
    "boat": "",
    "traffic light": "",
    "fire hydrant": "",
    "stop sign": "",
    "parking meter": "",
    "bench": "",
    "bird": "",
    "cat": "",
    "dog": "",
    "horse": "",
    "sheep": "",
    "cow": "",
    "elephant": "",
    "bear": "",
    "zebra": "",
    "giraffe": "",
    "backpack": "",
    "umbrella": "",
    "handbag": "",
    "tie": "",
    "suitcase": "",
    "frisbee": "",
    "skis": "",
    "snowboard": "",
    "sports ball": "",
    "kite": "",
    "baseball bat": "",
    "baseball glove": "",
    "skateboard": "",
    "surfboard": "",
    "tennis racket": "",
    "bottle": "",
    "wine glass": "",
    "cup": "",
    "fork": "",
    "knife": "",
    "spoon": "",
    "bowl": "",
    "banana": "",
    "apple": "",
    "sandwich": "",
    "orange": "",
    "broccoli": "",
    "carrot": "",
    "hot dog": "",
    "pizza": "",
    "donut": "",
    "cake": "",
    "chair": "",
    "couch": "",
    "potted plant": "",
    "bed": "",
    "dining table": "",
    "toilet": "",
    "tv": "",
    "laptop": "",
    "mouse": "",
    "remote": "",
    "keyboard": "",
    "cell phone": "",
    "microwave": "",
    "oven": "",
    "toaster": "",
    "sink": "",
    "refrigerator": "",
    "book": "",
    "clock": "",
    "vase": "",
    "scissors": "",
    "teddy bear": "",
    "hair drier": "",
    "toothbrush": "",
}

AUDIO_COOLDOWN = 5

pygame.mixer.init()
speaker_volume = 0.5  # 50%
pygame.mixer.music.set_volume(speaker_volume)

# model setup
model = YOLO("models\\yolov8n.pt")
cap = cv2.VideoCapture(0)

detected_objects = {}


def play_audio(object):
    if object not in audio_map:
        return
    filename = audio_map[object]
    try:
        full_path = os.path.join(audio_dir, filename)
        pygame.mixer.music.load(full_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy() == True:
            continue

    except FileNotFoundError:
        print(f"The directory '{audio_dir}' does not exist.")
    except PermissionError:
        print(f"Permission denied for accessing '{audio_dir}'.")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, stream=True, conf=0.4)

    current_time = time.time()
    current_frame_objects = set()

    for result in results:
        box = result.boxes
        classes = [int(x) for x in box.cls]

        for cls_id in classes:
            name = model.names[cls_id]
            current_frame_objects.add(name)

            # Only play audio if it's a new detection or cooldown has passed
            if name not in detected_objects or (
                current_time - detected_objects[name] > AUDIO_COOLDOWN
            ):
                play_audio(name)
                detected_objects[name] = current_time

    # Clean up: Remove objects no longer in frame
    for obj in list(detected_objects.keys()):
        if obj not in current_frame_objects:
            del detected_objects[obj]

    # Optional: Show annotated frame
    # annotated = result.plot()
    # cv2.imshow("Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
