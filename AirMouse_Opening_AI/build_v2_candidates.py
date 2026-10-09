from pathlib import Path
import cv2
import hashlib
import shutil
import random
import csv


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

RAW_ROOT = Path(r"F:\AirMouseRaw")

CURATED_ROOT = BASE_DIR / "datasets" / "curated_v2"
CANDIDATE_ROOT = CURATED_ROOT / "candidates"

POSITIVE_DIR = CANDIDATE_ROOT / "positive"
NEGATIVE_DIR = CANDIDATE_ROOT / "negative"
AMBIGUOUS_DIR = CANDIDATE_ROOT / "ambiguous"

PROVENANCE_FILE = CURATED_ROOT / "provenance.csv"

V1_ROOT = BASE_DIR / "datasets" / "airmouse_openings"


for folder in [
    POSITIVE_DIR,
    NEGATIVE_DIR,
    AMBIGUOUS_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

random.seed(42)


# =========================================================
# DUPLICATE DETECTION
# =========================================================

used_sha256 = set()
used_dhashes = []


def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as file:

        while True:

            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def dhash_image(path):

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        return None

    image = cv2.resize(image, (9, 8))

    difference = image[:, 1:] > image[:, :-1]

    value = 0

    for bit in difference.flatten():

        value = (value << 1) | int(bit)

    return value


def hamming_distance(hash1, hash2):

    return (hash1 ^ hash2).bit_count()


def is_too_similar(image_path, threshold=5):

    exact_hash = sha256_file(image_path)

    # Exact duplicate
    if exact_hash in used_sha256:
        return True, None, None

    perceptual_hash = dhash_image(image_path)

    if perceptual_hash is None:
        return True, None, None

    # Near duplicate
    for previous_hash in used_dhashes:

        if hamming_distance(
            perceptual_hash,
            previous_hash
        ) <= threshold:

            return True, None, None

    return False, exact_hash, perceptual_hash


# =========================================================
# PROVENANCE
# =========================================================

provenance_rows = []


def save_candidate(
    source_name,
    image_path,
    category,
    original_label=""
):

    duplicate, sha_hash, d_hash = is_too_similar(
        image_path
    )

    if duplicate:
        return False

    destination_folder = {

        "positive": POSITIVE_DIR,
        "negative": NEGATIVE_DIR,
        "ambiguous": AMBIGUOUS_DIR

    }[category]


    # Unique name based on source + original name + hash
    short_hash = sha_hash[:8]

    safe_source = "".join(
        c if c.isalnum() or c in "_-" else "_"
        for c in source_name
)

    short_hash = sha_hash[:16]

    new_name = (
        f"{safe_source}__"
        f"{short_hash}"
        f"{image_path.suffix.lower()}"
    )

    destination = destination_folder / new_name

    shutil.copy2(
        image_path,
        destination
    )


    used_sha256.add(sha_hash)
    used_dhashes.append(d_hash)


    provenance_rows.append({

        "new_filename": new_name,
        "source": source_name,
        "original_path": str(image_path),
        "category": category,
        "original_label": str(original_label),
        "sha256": sha_hash,
        "dhash": str(d_hash),
        "review_status": "candidate"

    })

    return True


# =========================================================
# HELPER: diverse sample
# =========================================================

def sample_images(
    paths,
    amount,
    source_name,
    category,
    label_function=None
):

    paths = list(paths)

    random.shuffle(paths)

    saved = 0

    for image_path in paths:

        if saved >= amount:
            break

        original_label = ""

        if label_function is not None:
            original_label = label_function(image_path)

        success = save_candidate(
            source_name,
            image_path,
            category,
            original_label
        )

        if success:
            saved += 1

    print(
        f"{source_name:25} "
        f"{category:10} "
        f"{saved}/{amount}"
    )


# =========================================================
# 1. MONORACEGATE
# =========================================================

print("\n=== MONORACEGATE ===")

v1_images = V1_ROOT / "images"

flight_groups = {}


for split in ["train", "val", "test"]:

    split_folder = v1_images / split

    if not split_folder.exists():
        continue

    for image in split_folder.iterdir():

        if image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        # Example:
        # adnec_s1_f1_img_101.png
        #
        # flight =
        # adnec_s1_f1

        if "_img_" in image.name:

            flight = image.name.split("_img_")[0]

        else:
            flight = "unknown"

        flight_groups.setdefault(
            flight,
            []
        ).append(image)


# Sample evenly across flights
for flight, images in flight_groups.items():

    sample_images(
        images,
        15,
        f"monorace_{flight}",
        "positive"
    )


# =========================================================
# 2. DEEPDOORS2
# =========================================================

print("\n=== DEEPDOORS2 ===")

deepdoors = RAW_ROOT / "02_deepdoors2"


def find_rgb_class(class_name):

    result = []

    for image in deepdoors.rglob("*"):

        if (
            image.is_file()
            and image.suffix.lower()
            in IMAGE_EXTENSIONS
        ):

            parts_lower = [
                part.lower()
                for part in image.parts
            ]

            if (
                "rgb" in parts_lower
                and image.parent.name.lower()
                == class_name.lower()
            ):
                result.append(image)

    return result


deep_open = find_rgb_class("Open")
deep_closed = find_rgb_class("Closed")
deep_semi = find_rgb_class("Semi")


sample_images(
    deep_open,
    160,
    "deepdoors_open",
    "positive"
)

sample_images(
    deep_closed,
    160,
    "deepdoors_closed",
    "negative"
)

sample_images(
    deep_semi,
    80,
    "deepdoors_semi",
    "ambiguous"
)


# =========================================================
# 3. WINDOWS / OPEN DOOR
# =========================================================

print("\n=== WINDOWS / OPEN DOOR ===")

windows_source = (
    RAW_ROOT
    / "03_windows_open_door"
)


window_images = [

    file

    for file in windows_source.rglob("*")

    if (
        file.is_file()
        and file.suffix.lower()
        in IMAGE_EXTENSIONS
    )

]


sample_images(
    window_images,
    180,
    "windows_open_door",
    "positive"
)


# =========================================================
# 4. DATASET 04
# =========================================================

print("\n=== DOOR WINDOW SEG ===")

print(
    "Skipping dataset 04 for V2 candidate selection "
    "(overlaps DeepDoors and mask meaning is uncertain)."
)


# =========================================================
# 5. INDOOR ENVIRONMENT
# =========================================================

print("\n=== INDOOR ENVIRONMENT ===")

indoor_source = (
    RAW_ROOT
    / "05_indoor_environment"
)


def indoor_label_path(image_path):

    parts = list(image_path.parts)

    try:

        index = [
            p.lower()
            for p in parts
        ].index("images")

    except ValueError:

        return ""

    parts[index] = "labels"

    label_path = Path(*parts).with_suffix(".txt")

    return label_path


indoor_negative_candidates = []


for image in indoor_source.rglob("*"):

    if (
        not image.is_file()
        or image.suffix.lower()
        not in IMAGE_EXTENSIONS
    ):
        continue


    label_path = indoor_label_path(image)

    if not label_path.exists():
        continue


    class_ids = set()


    try:

        with open(
            label_path,
            "r",
            encoding="utf-8"
        ) as label_file:

            for line in label_file:

                line = line.strip()

                if not line:
                    continue

                first_value = line.split()[0]

                class_ids.add(
                    int(float(first_value))
                )

    except Exception:
        continue


    # Dataset classes:
    #
    # 0 = AC
    # 1 = Door
    # 2 = Monitor
    # 3 = Window
    #
    # Prefer images containing only AC / Monitor
    # or completely empty labels.
    #
    # Avoid automatically treating Door/Window
    # scenes as negative.

    if (
        len(class_ids) == 0
        or class_ids.issubset({0, 2})
    ):
        indoor_negative_candidates.append(
            image
        )


sample_images(
    indoor_negative_candidates,
    180,
    "indoor_hard_negative",
    "negative",
    label_function=indoor_label_path
)


# =========================================================
# 6. OPEN / CLOSE DOOR DATASET
# =========================================================

print("\n=== OPEN / CLOSE DOOR ===")

open_close_source = (
    RAW_ROOT
    / "06_open_close_door"
)


open_close_images = [

    file

    for file in open_close_source.rglob("*")

    if (
        file.is_file()
        and file.suffix.lower()
        in IMAGE_EXTENSIONS
    )

]


# Labels do not reliably encode open vs closed,
# therefore treat these as ambiguous candidates.

sample_images(
    open_close_images,
    100,
    "open_close_aux",
    "ambiguous"
)


# =========================================================
# WRITE PROVENANCE
# =========================================================

fieldnames = [
    "new_filename",
    "source",
    "original_path",
    "category",
    "original_label",
    "sha256",
    "dhash",
    "review_status"
]


with open(
    PROVENANCE_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as csv_file:

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        provenance_rows
    )


# =========================================================
# SUMMARY
# =========================================================

positive_count = len(
    list(POSITIVE_DIR.iterdir())
)

negative_count = len(
    list(NEGATIVE_DIR.iterdir())
)

ambiguous_count = len(
    list(AMBIGUOUS_DIR.iterdir())
)


print("\n======================================")
print("V2 CANDIDATE BUILD COMPLETED")
print("======================================")

print(
    "Positive candidates:",
    positive_count
)

print(
    "Negative candidates:",
    negative_count
)

print(
    "Ambiguous candidates:",
    ambiguous_count
)

print(
    "TOTAL:",
    positive_count
    + negative_count
    + ambiguous_count
)

print(
    "\nProvenance saved to:"
)

print(PROVENANCE_FILE)