from pathlib import Path
import csv
import shutil


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

CURATED_ROOT = BASE_DIR / "datasets" / "curated_v2"

CANDIDATES = CURATED_ROOT / "candidates"

PROVENANCE = CURATED_ROOT / "provenance.csv"

WORK_ROOT = CURATED_ROOT / "annotations_work"


READY_IMAGES = WORK_ROOT / "ready" / "images"
READY_LABELS = WORK_ROOT / "ready" / "labels"

TO_ANNOTATE = WORK_ROOT / "to_annotate" / "images"


# =========================================================
# SAFETY CHECK
# =========================================================

if WORK_ROOT.exists() and any(WORK_ROOT.rglob("*")):

    print("annotations_work already contains files.")
    print("Stopping so existing annotation work is not overwritten.")
    print("Delete it manually ONLY if you intentionally want to rebuild it.")

    raise SystemExit


READY_IMAGES.mkdir(parents=True, exist_ok=True)
READY_LABELS.mkdir(parents=True, exist_ok=True)
TO_ANNOTATE.mkdir(parents=True, exist_ok=True)


# =========================================================
# LOAD PROVENANCE
# =========================================================

with open(
    PROVENANCE,
    "r",
    encoding="utf-8"
) as file:

    rows = list(csv.DictReader(file))


mono_count = 0
manual_positive_count = 0
negative_count = 0


# =========================================================
# PROCESS REVIEWED DATA
# =========================================================

for row in rows:

    category = row["category"]

    if category not in {
        "positive",
        "negative"
    }:
        continue


    filename = row["new_filename"]

    candidate_path = (
        CANDIDATES
        / category
        / filename
    )


    if not candidate_path.exists():

        print("Missing candidate:", filename)
        continue


    # -----------------------------------------------------
    # POSITIVE
    # -----------------------------------------------------

    if category == "positive":

        source = row["source"]


        # ================================================
        # MonoRaceGate already has segmentation labels
        # ================================================

        if source.startswith("monorace_"):

            original_image = Path(
                row["original_path"]
            )

            # Example:
            #
            # .../airmouse_openings/images/train/x.png
            #
            # becomes:
            #
            # .../airmouse_openings/labels/train/x.txt

            split_name = original_image.parent.name

            dataset_root = original_image.parents[2]

            original_label = (
                dataset_root
                / "labels"
                / split_name
                / f"{original_image.stem}.txt"
            )


            if not original_label.exists():

                print(
                    "Missing MonoRaceGate label:",
                    original_label
                )

                continue


            shutil.copy2(
                candidate_path,
                READY_IMAGES / filename
            )


            shutil.copy2(
                original_label,
                READY_LABELS
                / f"{Path(filename).stem}.txt"
            )


            mono_count += 1


        # ================================================
        # Real indoor positive → manual annotation needed
        # ================================================

        else:

            shutil.copy2(
                candidate_path,
                TO_ANNOTATE / filename
            )

            manual_positive_count += 1


    # -----------------------------------------------------
    # NEGATIVE
    # -----------------------------------------------------

    elif category == "negative":

        shutil.copy2(
            candidate_path,
            READY_IMAGES / filename
        )


        # Empty YOLO label = no object/opening
        empty_label = (
            READY_LABELS
            / f"{Path(filename).stem}.txt"
        )

        empty_label.touch()


        negative_count += 1


# =========================================================
# SUMMARY
# =========================================================

print("\n======================================")
print("V2 ANNOTATION PACK READY")
print("======================================")

print(
    "MonoRaceGate positives already labeled:",
    mono_count
)

print(
    "Real positives needing annotation:",
    manual_positive_count
)

print(
    "Negatives automatically ready:",
    negative_count
)

print(
    "\nAlready-ready training images:",
    mono_count + negative_count
)

print(
    "Images needing manual polygons:",
    manual_positive_count
)

print(
    "\nAnnotation work folder:"
)

print(WORK_ROOT)