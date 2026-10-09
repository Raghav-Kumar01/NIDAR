from ultralytics import YOLO

# -------------------------------------------------
# Load a PRETRAINED segmentation model
# -------------------------------------------------

model = YOLO("yolo11n-seg.pt")


# -------------------------------------------------
# Fine-tune it on our opening dataset
# -------------------------------------------------

model.train(
    data="airmouse_openings.yaml",

    epochs=30,

    imgsz=640,

    batch=8,

    device=0,

    workers=0,

    project="runs/airmouse",

    name="opening_seg_v1",

    patience=10
)