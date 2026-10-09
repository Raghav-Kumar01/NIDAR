import cv2
import csv
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / "datasets" / "v3_pilot"

FRAMES = DATA / "frames"
MANIFEST = DATA / "frames_manifest.csv"
OUTPUT = DATA / "review_v3.csv"

with open(MANIFEST, newline="", encoding="utf-8") as f:
    records = list(csv.DictReader(f))

# Resume previous review if available
reviews = {}

if OUTPUT.exists():
    with open(OUTPUT, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            reviews[row["filename"]] = row["label"]


def save_reviews():

    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow([
            "filename",
            "source_video",
            "label"
        ])

        for record in records:

            name = record["filename"]

            if name in reviews:
                writer.writerow([
                    name,
                    record["source_video"],
                    reviews[name]
                ])


cv2.namedWindow("V3 Dataset Review", cv2.WINDOW_NORMAL)

for index, record in enumerate(records, start=1):

    name = record["filename"]

    if name in reviews:
        continue

    image = cv2.imread(str(FRAMES / name))

    if image is None:
        print("Cannot read:", name)
        continue

    h, w = image.shape[:2]

    scale = min(1100 / w, 650 / h, 1.0)

    image = cv2.resize(
        image,
        (int(w * scale), int(h * scale))
    )

    # Add instructions above the image
    image = cv2.copyMakeBorder(
        image,
        80, 0, 0, 0,
        cv2.BORDER_CONSTANT,
        value=(30, 30, 30)
    )

    cv2.putText(
        image,
        f"{index}/{len(records)} - {name}",
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    cv2.putText(
        image,
        "P: Positive | N: Negative | A: Ambiguous | X: Reject | Q: Quit",
        (15, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 255, 255),
        1
    )

    cv2.imshow("V3 Dataset Review", image)

    while True:

        key = cv2.waitKey(0) & 0xFF
        label = chr(key).upper() if key < 128 else ""

        if label in ("P", "N", "A", "X"):

            reviews[name] = label

            save_reviews()

            print(f"{index}/{len(records)}: {name} -> {label}")
            break

        if label == "Q":

            save_reviews()
            cv2.destroyAllWindows()
            raise SystemExit


cv2.destroyAllWindows()

print("\nReview completed!")

for label in ("P", "N", "A", "X"):
    count = sum(v == label for v in reviews.values())
    print(f"{label}: {count}")

print("Saved:", OUTPUT)