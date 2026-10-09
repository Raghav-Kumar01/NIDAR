from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

EXPORT_ROOT = (
    BASE_DIR
    / "datasets"
    / "curated_v2"
    / "annotations_work"
    / "batch1_export"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


# Find Roboflow export folder automatically
dataset_roots = [
    p.parent
    for p in EXPORT_ROOT.rglob("data.yaml")
]

if not dataset_roots:
    print("ERROR: data.yaml not found")
    raise SystemExit

DATASET_ROOT = dataset_roots[0]

print("Dataset root:")
print(DATASET_ROOT)


images = []
labels = []

for file in DATASET_ROOT.rglob("*"):

    if not file.is_file():
        continue

    if file.suffix.lower() in IMAGE_EXTENSIONS:
        images.append(file)

    elif file.suffix.lower() == ".txt":

        # Ignore README txt files
        if "labels" in [p.lower() for p in file.parts]:
            labels.append(file)


print("\nImages:", len(images))
print("Labels:", len(labels))


# Validate label polygons
invalid_labels = 0
class_ids = set()

for label in labels:

    for line in label.read_text(
        encoding="utf-8"
    ).splitlines():

        line = line.strip()

        if not line:
            continue

        values = line.split()

        class_id = int(float(values[0]))
        class_ids.add(class_id)

        coords = list(map(float, values[1:]))

        # Segmentation needs x,y pairs
        if len(coords) < 6 or len(coords) % 2 != 0:

            print("Invalid polygon:", label)
            invalid_labels += 1
            continue

        if any(
            value < 0 or value > 1
            for value in coords
        ):

            print("Out-of-range coordinates:", label)
            invalid_labels += 1


print("\nClasses found:", class_ids)
print("Invalid polygons:", invalid_labels)