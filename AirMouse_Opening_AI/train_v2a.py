from pathlib import Path
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent

V1_MODEL = (
    BASE_DIR
    / "runs"
    / "segment"
    / "runs"
    / "airmouse"
    / "opening_seg_v1-2"
    / "weights"
    / "best.pt"
)


print("Starting from V1:")
print(V1_MODEL)


model = YOLO(str(V1_MODEL))


model.train(
    data=str(BASE_DIR / "v2a_openings.yaml"),

    epochs=20,
    imgsz=640,
    batch=8,

    device=0,
    workers=0,

    patience=6,

    project=str(BASE_DIR / "runs" / "airmouse"),
    name="opening_seg_v2a"
)