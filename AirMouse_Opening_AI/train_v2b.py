from pathlib import Path
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent


MODEL_PATH = (
    BASE_DIR
    / "runs"
    / "airmouse"
    / "opening_seg_v2a"
    / "weights"
    / "best.pt"
)


print(
    "Fine-tuning from V2A:"
)

print(MODEL_PATH)


model = YOLO(
    str(MODEL_PATH)
)


model.train(

    data=str(
        BASE_DIR
        / "v2b_openings.yaml"
    ),

    epochs=20,

    imgsz=640,

    batch=8,

    device=0,

    workers=0,

    patience=7,

    lr0=0.001,

    project=str(
        BASE_DIR
        / "runs"
        / "airmouse"
    ),

    name="opening_seg_v2b"
)