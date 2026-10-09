import cv2
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "runs"
    / "airmouse"
    / "opening_seg_v2b"
    / "weights"
    / "best.pt"
)

print("Using V2B model:")
print(MODEL_PATH)

model = YOLO(str(MODEL_PATH))

CAMERA_URL = "http://127.0.0.1:18080/video"
cap = cv2.VideoCapture(CAMERA_URL)
if not cap.isOpened():
    print("Phone camera could not be opened")
    raise SystemExit

print("Phone USB camera connected")
if not cap.isOpened():
    print("Camera could not be opened")
    exit()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model.predict(
        frame,
        imgsz=640,
        conf=0.25,
        device=0,
        verbose=False
    )

    annotated = results[0].plot()

    cv2.imshow(
        "AirMouse V2B - Live Test",
        annotated
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()