from pathlib import Path
import csv
import shutil
import random

BASE_DIR = Path(__file__).resolve().parent

CURATED = BASE_DIR / "datasets" / "curated_v2"

READY_IMAGES = (
    CURATED
    / "annotations_work"
    / "ready"
    / "images"
)

READY_LABELS = (
    CURATED
    / "annotations_work"
    / "ready"
    / "labels"
)

PROVENANCE = CURATED / "provenance.csv"

ROBOFLOW_ROOT = (
    CURATED
    / "annotations_work"
    / "batch1_export"
)

OUTPUT = BASE_DIR / "datasets" / "v2b_openings"

IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}

random.seed(42)


# =========================================================
# VALIDATE SEGMENTATION LABEL
# =========================================================

def valid_positive_label(label_path):

    if not label_path.exists():
        return False

    lines = label_path.read_text(
        encoding="utf-8"
    ).splitlines()

    # Positive cannot have empty label
    if not lines:
        return False

    found_polygon = False

    for line in lines:

        line = line.strip()

        if not line:
            continue

        values = line.split()

        if len(values) < 7:
            return False

        try:
            class_id = int(float(values[0]))
            coords = list(
                map(float, values[1:])
            )
        except Exception:
            return False

        if class_id != 0:
            return False

        # x,y pairs
        if len(coords) % 2 != 0:
            return False

        if len(coords) < 6:
            return False

        # Reject invalid normalized points
        if any(
            value < 0.0 or value > 1.0
            for value in coords
        ):
            return False

        found_polygon = True

    return found_polygon


# =========================================================
# LOAD PROVENANCE
# =========================================================

with open(
    PROVENANCE,
    "r",
    encoding="utf-8"
) as f:

    rows = list(
        csv.DictReader(f)
    )


row_by_filename = {
    row["new_filename"]: row
    for row in rows
}


# =========================================================
# COLLECT CLEAN MONORACE POSITIVES
# =========================================================

mono_positives = []

for image in READY_IMAGES.iterdir():

    if image.suffix.lower() not in IMAGE_EXTS:
        continue

    row = row_by_filename.get(
        image.name
    )

    if row is None:
        continue

    if row["category"] != "positive":
        continue

    if not row["source"].startswith(
        "monorace_"
    ):
        continue

    label = (
        READY_LABELS
        / f"{image.stem}.txt"
    )

    if valid_positive_label(label):

        mono_positives.append(
            (image, label, "mono_positive")
        )


print(
    "Clean MonoRace positives:",
    len(mono_positives)
)


# =========================================================
# COLLECT HARD NEGATIVES
# =========================================================

negatives = []

for image in READY_IMAGES.iterdir():

    if image.suffix.lower() not in IMAGE_EXTS:
        continue

    row = row_by_filename.get(
        image.name
    )

    if row is None:
        continue

    if row["category"] != "negative":
        continue

    label = (
        READY_LABELS
        / f"{image.stem}.txt"
    )

    if not label.exists():
        continue

    # Negative must have empty TXT
    if label.read_text(
        encoding="utf-8"
    ).strip():

        continue

    negatives.append(
        (image, label, "hard_negative")
    )


random.shuffle(negatives)

# Balance dataset instead of using all 408
negatives = negatives[:180]

print(
    "Hard negatives selected:",
    len(negatives)
)


# =========================================================
# FIND ROBOFLOW DATASET
# =========================================================

yaml_files = list(
    ROBOFLOW_ROOT.rglob(
        "data.yaml"
    )
)

if not yaml_files:

    raise RuntimeError(
        "Roboflow data.yaml not found"
    )

ROBO_ROOT = yaml_files[0].parent


# =========================================================
# COLLECT REAL POSITIVES
# =========================================================

real_positives = []

for image in ROBO_ROOT.rglob("*"):

    if (
        not image.is_file()
        or image.suffix.lower()
        not in IMAGE_EXTS
    ):
        continue

    if image.parent.name.lower() != "images":
        continue

    label = (
        image.parent.parent
        / "labels"
        / f"{image.stem}.txt"
    )

    if valid_positive_label(label):

        real_positives.append(
            (
                image,
                label,
                "real_positive"
            )
        )


print(
    "Real Roboflow positives:",
    len(real_positives)
)


# =========================================================
# BUILD CLEAN LIST
# =========================================================

positives = (
    mono_positives
    + real_positives
)

print(
    "Total clean positives:",
    len(positives)
)

print(
    "Total negatives:",
    len(negatives)
)


# =========================================================
# SPLIT POSITIVE AND NEGATIVE SEPARATELY
# =========================================================

random.shuffle(positives)
random.shuffle(negatives)


def split_group(items):

    split_point = int(
        len(items) * 0.85
    )

    return (
        items[:split_point],
        items[split_point:]
    )


pos_train, pos_val = split_group(
    positives
)

neg_train, neg_val = split_group(
    negatives
)


train = pos_train + neg_train
val = pos_val + neg_val

random.shuffle(train)
random.shuffle(val)


# =========================================================
# REBUILD OUTPUT
# =========================================================

if OUTPUT.exists():

    shutil.rmtree(
        OUTPUT
    )


for split in [
    "train",
    "val"
]:

    (
        OUTPUT
        / "images"
        / split
    ).mkdir(
        parents=True
    )

    (
        OUTPUT
        / "labels"
        / split
    ).mkdir(
        parents=True
    )


# =========================================================
# COPY
# =========================================================

def copy_samples(
    items,
    split
):

    for index, (
        image,
        label,
        sample_type
    ) in enumerate(items):

        stem = (
            f"v2b_{split}_"
            f"{index:04d}"
        )

        image_dest = (
            OUTPUT
            / "images"
            / split
            / f"{stem}{image.suffix.lower()}"
        )

        label_dest = (
            OUTPUT
            / "labels"
            / split
            / f"{stem}.txt"
        )

        shutil.copy2(
            image,
            image_dest
        )

        shutil.copy2(
            label,
            label_dest
        )


copy_samples(
    train,
    "train"
)

copy_samples(
    val,
    "val"
)


# =========================================================
# YAML
# =========================================================

yaml_text = f"""path: {OUTPUT.as_posix()}

train: images/train
val: images/val

names:
  0: traversable_opening
"""

(
    BASE_DIR
    / "v2b_openings.yaml"
).write_text(
    yaml_text,
    encoding="utf-8"
)


print("\n==============================")
print("V2B CLEAN DATASET READY")
print("==============================")

print(
    "Train:",
    len(train)
)

print(
    "Validation:",
    len(val)
)

print(
    "Positive:",
    len(positives)
)

print(
    "Negative:",
    len(negatives)
)

print(
    "\nYAML: v2b_openings.yaml"
)