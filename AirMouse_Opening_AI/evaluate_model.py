from ultralytics import YOLO

model = YOLO(
    "runs/segment/runs/airmouse/opening_seg_v1-2/weights/best.pt"
)

results = model.val(
    data="airmouse_openings.yaml",
    split="test",
    device=0,
    workers=0
)