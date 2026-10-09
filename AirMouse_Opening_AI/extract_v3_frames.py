import cv2
import csv
from pathlib import Path


# ==================================================
# FOLDERS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

VIDEO_DIR = BASE_DIR / "datasets" / "v3_pilot" / "videos"
FRAME_DIR = BASE_DIR / "datasets" / "v3_pilot" / "frames"

FRAME_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST = BASE_DIR / "datasets" / "v3_pilot" / "frames_manifest.csv"


# ==================================================
# SETTINGS
# ==================================================

SAMPLE_INTERVAL = 1.0     # One frame per second
MAX_FRAMES_PER_VIDEO = 35
DUPLICATE_THRESHOLD = 5

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# ==================================================
# PERCEPTUAL HASH
# ==================================================

def calculate_hash(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    small = cv2.resize(gray, (9, 8))

    differences = small[:, 1:] > small[:, :-1]

    hash_value = 0

    for bit in differences.flatten():
        hash_value = (hash_value << 1) | int(bit)

    return hash_value


def hash_distance(hash1, hash2):

    return (hash1 ^ hash2).bit_count()


# ==================================================
# FRAME EXTRACTION
# ==================================================

videos = sorted(
    p for p in VIDEO_DIR.iterdir()
    if p.suffix.lower() in VIDEO_EXTENSIONS
)

if not videos:
    print("No videos found in:", VIDEO_DIR)
    raise SystemExit

records = []

for video_path in videos:

    print(f"\nProcessing: {video_path.name}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("Could not open video. Skipping.")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0 or fps != fps:
        fps = 30.0

    sample_every = max(1, round(fps * SAMPLE_INTERVAL))

    frame_number = 0
    saved = 0
    previous_hash = None

    while saved < MAX_FRAMES_PER_VIDEO:

        success, frame = cap.read()

        if not success:
            break

        current_frame = frame_number
        frame_number += 1

        # Sample approximately once per second
        if current_frame % sample_every != 0:
            continue

        # Calculate perceptual hash
        current_hash = calculate_hash(frame)

        # Skip frames that are too similar
        if previous_hash is not None:

            difference = hash_distance(
                previous_hash,
                current_hash
            )

            if difference <= DUPLICATE_THRESHOLD:
                continue

        # Check image sharpness
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        blur_score = cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()

        review_flag = (
            "CHECK_BLUR"
            if blur_score < 45
            else "OK"
        )

        # Filename contains source and frame number
        filename = (
            f"{video_path.stem}_"
            f"{video_path.suffix[1:]}_"
            f"{current_frame:07d}.jpg"
        )

        output_path = FRAME_DIR / filename

        written = cv2.imwrite(
            str(output_path),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 90]
        )

        if not written:
            print("Could not save:", filename)
            continue

        records.append({
            "filename": filename,
            "source_video": video_path.name,
            "time_seconds": round(
                current_frame / fps, 2
            ),
            "blur_score": round(
                float(blur_score), 2
            ),
            "review_flag": review_flag
        })

        previous_hash = current_hash
        saved += 1

    cap.release()

    print(f"Saved {saved} frames")


# ==================================================
# SAVE PROVENANCE
# ==================================================

with open(
    MANIFEST,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "filename",
            "source_video",
            "time_seconds",
            "blur_score",
            "review_flag"
        ]
    )

    writer.writeheader()
    writer.writerows(records)


print("\n----------------------------")
print("Extraction complete")
print("Videos processed:", len(videos))
print("Frames saved:", len(records))
print("Manifest:", MANIFEST)
print("----------------------------")