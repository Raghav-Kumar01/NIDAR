from pathlib import Path
import shutil
import random
import hashlib

BASE_DIR = Path(__file__).resolve().parent

CURATED = BASE_DIR / "datasets" / "curated_v2"

READY_IMAGES = CURATED / "annotations_work" / "ready" / "images"
READY_LABELS = CURATED / "annotations_work" / "ready" / "labels"

EXPORT_ROOT = CURATED / "annotations_work" / "batch1_export"

OUTPUT = BASE_DIR / "datasets" / "v2a_openings"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

random.seed(42)


# ------------------------------------------------
# Find Roboflow export automatically
# ------------------------------------------------

yaml_files = list(EXPORT_ROOT.rglob("data.yaml"))

if not yaml_files:
    raise RuntimeError("Roboflow data.yaml not found")

ROBO_ROOT = yaml_files[0].parent

print("Roboflow dataset:", ROBO_ROOT)


# ------------------------------------------------
# Clean V2A output
# ------------------------------------------------

if OUTPUT.exists():
    shutil.rmtree(OUTPUT)

for split in ["train", "val", "test"]:

    (OUTPUT / "images" / split).mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT / "labels" / split).mkdir(
        parents=True,
        exist_ok=True
    )


# ------------------------------------------------
# Collect ready images
# 93 MonoRace + 408 negatives
# ------------------------------------------------

samples = []

for image in READY_IMAGES.iterdir():

    if image.suffix.lower() not in IMAGE_EXTS:
        continue

    label = READY_LABELS / f"{image.stem}.txt"

    if label.exists():
        samples.append((image, label))


print("Existing ready samples:", len(samples))


# ------------------------------------------------
# Collect 74 Roboflow annotated positives
# ------------------------------------------------

roboflow_count = 0

for image in ROBO_ROOT.rglob("*"):

    if (
        not image.is_file()
        or image.suffix.lower() not in IMAGE_EXTS
    ):
        continue

    # Only actual image folders
    if image.parent.name.lower() != "images":
        continue

    label_folder = image.parent.parent / "labels"
    label = label_folder / f"{image.stem}.txt"

    if label.exists():

        samples.append((image, label))
        roboflow_count += 1


print("Roboflow annotated samples:", roboflow_count)
print("Total samples:", len(samples))


# ------------------------------------------------
# Remove exact duplicate images
# ------------------------------------------------

unique = []
seen_hashes = set()

for image, label in samples:

    sha = hashlib.sha256(
        image.read_bytes()
    ).hexdigest()

    if sha in seen_hashes:
        continue

    seen_hashes.add(sha)

    unique.append((image, label))


samples = unique

print("After exact dedup:", len(samples))


# ------------------------------------------------
# Deterministic shuffle + split
# ------------------------------------------------

random.shuffle(samples)

total = len(samples)

train_end = int(total * 0.80)
val_end = int(total * 0.90)

splits = {
    "train": samples[:train_end],
    "val": samples[train_end:val_end],
    "test": samples[val_end:]
}


# ------------------------------------------------
# Copy with safe unique names
# ------------------------------------------------

for split, items in splits.items():

    for index, (image, label) in enumerate(items):

        new_stem = f"v2a_{split}_{index:04d}"

        image_dest = (
            OUTPUT
            / "images"
            / split
            / f"{new_stem}{image.suffix.lower()}"
        )

        label_dest = (
            OUTPUT
            / "labels"
            / split
            / f"{new_stem}.txt"
        )

        shutil.copy2(image, image_dest)
        shutil.copy2(label, label_dest)


# ------------------------------------------------
# Create YOLO config
# ------------------------------------------------

yaml_text = f"""path: {OUTPUT.as_posix()}

train: images/train
val: images/val
test: images/test

names:
  0: traversable_opening
"""

(BASE_DIR / "v2a_openings.yaml").write_text(
    yaml_text,
    encoding="utf-8"
)


print("\n==============================")
print("V2A DATASET READY")
print("==============================")

for split in ["train", "val", "test"]:

    print(
        split,
        "=",
        len(splits[split])
    )

print("\nConfig: v2a_openings.yaml")