import cv2
import math
import csv
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent

DATASET = BASE / "datasets" / "v3_pilot"
FRAMES = DATASET / "frames"
OUTPUT = DATASET / "review_sheets"
MANIFEST = DATASET / "frames_manifest.csv"

OUTPUT.mkdir(parents=True, exist_ok=True)

# Layout
COLS = 4
ROWS = 5

THUMB_W = 320
THUMB_H = 180
LABEL_H = 35

PER_SHEET = COLS * ROWS

# Read the extraction manifest
with open(MANIFEST, newline="", encoding="utf-8") as f:
    records = list(csv.DictReader(f))

total_sheets = math.ceil(len(records) / PER_SHEET)

for sheet_no in range(total_sheets):

    sheet = np.full(
        (
            ROWS * (THUMB_H + LABEL_H),
            COLS * THUMB_W,
            3
        ),
        35,
        dtype=np.uint8
    )

    batch = records[
        sheet_no * PER_SHEET:
        (sheet_no + 1) * PER_SHEET
    ]

    for i, record in enumerate(batch):

        image_path = FRAMES / record["filename"]
        image = cv2.imread(str(image_path))

        if image is None:
            print("Cannot read:", image_path)
            continue

        image = cv2.resize(
            image,
            (THUMB_W, THUMB_H)
        )

        row = i // COLS
        col = i % COLS

        x = col * THUMB_W
        y = row * (THUMB_H + LABEL_H)

        sheet[
            y:y + THUMB_H,
            x:x + THUMB_W
        ] = image

        name = record["filename"]
        status = record["review_flag"]

        if status == "CHECK_BLUR":
            color = (0, 200, 255)
        else:
            color = (255, 255, 255)

        cv2.putText(
            sheet,
            name[:38],
            (x + 5, y + THUMB_H + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            color,
            1
        )

    output_path = OUTPUT / f"sheet_{sheet_no + 1:02d}.jpg"

    cv2.imwrite(str(output_path), sheet)

    print("Created:", output_path.name)

print("\nTotal frames:", len(records))
print("Total sheets:", total_sheets)
print("Output folder:", OUTPUT)