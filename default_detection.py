import os
import cv2
import time
from ultralytics import YOLO
import pygame

# audio setup
audio_dir = "audio_files"
audio_map = {
    "airplane": "aeroplane.mp3",
    "backpack": "backpack.mp3",
    "bed": "bed.mp3",
    "bench": "bench.mp3",
    "bicycle": "bicycle.mp3",
    "bird": "bird.mp3",
    "boat": "boat.mp3",
    "book": "book.mp3",
    "bottle": "bottle.mp3",
    "bowl": "bowl.mp3",
    "bus": "bus.mp3",
    "car": "car.mp3",
    "cat": "cat.mp3",
    "cell phone": "cellphone.mp3",
    "chair": "chair.mp3",
    "clock": "clock.mp3",
    "couch": "couch.mp3",
    "cow": "cow.mp3",
    "cup": "cup.mp3",
    "dining table": "diningtable.mp3",
    "dog": "dog.mp3",
    "fire hydrant": "firehydrant.mp3",
    "handbag": "handbag.mp3",
    "horse": "horse.mp3",
    "keyboard": "keyboard.mp3",
    "knife": "knife.mp3",
    "laptop": "laptop.mp3",
    "microwave": "microwave.mp3",
    "motorcycle": "motorcycle.mp3",
    "mouse": "mouse.mp3",
    "oven": "oven.mp3",
    "parking meter": "parkingmeter.mp3",
    "person": "person.mp3",
    "potted plant": "pottedplant.mp3",
    "refrigerator": "refrigerator.mp3",
    "remote": "remote.mp3",
    "scissors": "scissors.mp3",
    "sheep": "sheep.mp3",
    "sink": "sink.mp3",
    "spoon": "spoon.mp3",
    "sports ball": "sportsball.mp3",
    "stop sign": "stopsign.mp3",
    "suitcase": "suitcase.mp3",
    "toaster": "toaster.mp3",
    "traffic light": "trafficlight.mp3",
    "train": "train.mp3",
    "truck": "truck.mp3",
    "umbrella": "umbrella.mp3",
    "vase": "vase.mp3",
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
    annotated = result.plot()
    cv2.imshow("Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
