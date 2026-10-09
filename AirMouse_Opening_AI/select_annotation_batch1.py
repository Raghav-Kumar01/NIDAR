from pathlib import Path
import csv
import shutil
import random

BASE_DIR = Path(__file__).resolve().parent

ROOT = BASE_DIR / "datasets" / "curated_v2"
PROVENANCE = ROOT / "provenance.csv"

TO_ANNOTATE = ROOT / "annotations_work" / "to_annotate" / "images"
BATCH1 = ROOT / "annotations_work" / "batch1" / "images"

BATCH1.mkdir(parents=True, exist_ok=True)

random.seed(42)

with open(PROVENANCE, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))


# Only reviewed positives which still need annotation
available = []

for row in rows:

    if (
        row["category"] == "positive"
        and not row["source"].startswith("monorace_")
    ):

        path = TO_ANNOTATE / row["new_filename"]

        if path.exists():
            available.append((row, path))


# Group by source
groups = {}

for row, path in available:

    groups.setdefault(
        row["source"],
        []
    ).append(path)


print("Sources found:")

for source, images in groups.items():
    print(source, len(images))


# Balanced batch
selected = []

for source, images in groups.items():

    random.shuffle(images)

    # Up to 40 from every positive source
    selected.extend(images[:40])


# Limit total batch to 80
random.shuffle(selected)
selected = selected[:80]


for image in selected:

    shutil.copy2(
        image,
        BATCH1 / image.name
    )


print("\nAnnotation Batch 1 ready!")
print("Images selected:", len(selected))
print("Folder:", BATCH1)