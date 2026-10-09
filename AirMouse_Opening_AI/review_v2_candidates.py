from pathlib import Path
import cv2
import csv
import shutil


BASE_DIR = Path(__file__).resolve().parent

CURATED_ROOT = BASE_DIR / "datasets" / "curated_v2"
CANDIDATE_ROOT = CURATED_ROOT / "candidates"

POSITIVE_DIR = CANDIDATE_ROOT / "positive"
NEGATIVE_DIR = CANDIDATE_ROOT / "negative"
AMBIGUOUS_DIR = CANDIDATE_ROOT / "ambiguous"
REJECTED_DIR = CANDIDATE_ROOT / "rejected"

PROVENANCE_FILE = CURATED_ROOT / "provenance.csv"

REJECTED_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------
# Load provenance.csv
# -----------------------------------------------------

with open(PROVENANCE_FILE, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    rows = list(reader)

fieldnames = reader.fieldnames


# Quick lookup by filename
row_lookup = {
    row["new_filename"]: row
    for row in rows
}


# -----------------------------------------------------
# Collect images
# Review positives first, then ambiguous, then negatives
# -----------------------------------------------------

review_items = []

for category, folder in [
    ("positive", POSITIVE_DIR),
    ("ambiguous", AMBIGUOUS_DIR),
    ("negative", NEGATIVE_DIR),
]:

    for image_path in sorted(folder.iterdir()):

        if image_path.suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".bmp", ".webp"
        }:

            review_items.append(
                (category, image_path)
            )


print("Images to review:", len(review_items))

index = 0


# -----------------------------------------------------
# Save provenance after each decision
# -----------------------------------------------------

def save_provenance():

    with open(
        PROVENANCE_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)


# -----------------------------------------------------
# Move image + update provenance
# -----------------------------------------------------

def classify_image(image_path, new_category):

    destinations = {
        "positive": POSITIVE_DIR,
        "negative": NEGATIVE_DIR,
        "ambiguous": AMBIGUOUS_DIR,
        "rejected": REJECTED_DIR,
    }

    destination = destinations[new_category] / image_path.name

    # Only move if necessary
    if image_path.resolve() != destination.resolve():
        shutil.move(str(image_path), str(destination))

    row = row_lookup.get(image_path.name)

    if row is not None:

        row["category"] = new_category

        if new_category == "rejected":
            row["review_status"] = "rejected"
        else:
            row["review_status"] = "reviewed"

    save_provenance()


# -----------------------------------------------------
# Review loop
# -----------------------------------------------------

while index < len(review_items):

    original_category, image_path = review_items[index]

    # It might already have been moved during this session
    if not image_path.exists():
        index += 1
        continue

    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not open:", image_path)
        index += 1
        continue


    # Resize only for display
    h, w = image.shape[:2]

    max_width = 1100
    max_height = 700

    scale = min(
        max_width / w,
        max_height / h,
        1.0
    )

    display = cv2.resize(
        image,
        (
            int(w * scale),
            int(h * scale)
        )
    )


    # Add information bar
    text1 = f"{index + 1}/{len(review_items)} | Current: {original_category}"

    text2 = "P=Positive  N=Negative  A=Ambiguous  X=Reject  S=Skip  Q=Quit"

    cv2.putText(
        display,
        text1,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display,
        text2,
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    cv2.imshow(
        "AirMouse V2 Candidate Review",
        display
    )

    key = cv2.waitKey(0) & 0xFF


    if key == ord("p"):

        classify_image(
            image_path,
            "positive"
        )

        print("POSITIVE:", image_path.name)
        index += 1


    elif key == ord("n"):

        classify_image(
            image_path,
            "negative"
        )

        print("NEGATIVE:", image_path.name)
        index += 1


    elif key == ord("a"):

        classify_image(
            image_path,
            "ambiguous"
        )

        print("AMBIGUOUS:", image_path.name)
        index += 1


    elif key == ord("x"):

        classify_image(
            image_path,
            "rejected"
        )

        print("REJECTED:", image_path.name)
        index += 1


    elif key == ord("s"):

        print("SKIPPED:", image_path.name)
        index += 1


    elif key == ord("q"):

        print("Review stopped.")
        break


cv2.destroyAllWindows()

print("\nReview session finished.")