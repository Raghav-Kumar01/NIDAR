from pathlib import Path
from collections import Counter


# We extracted datasets here
RAW_ROOT = Path(r"F:\AirMouseRaw")

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


print("\nAIRMOUSE RAW DATASET INVENTORY")
print("=" * 60)


for source in sorted(RAW_ROOT.iterdir()):

    if not source.is_dir():
        continue

    image_files = []
    extension_counter = Counter()

    for file in source.rglob("*"):

        if not file.is_file():
            continue

        ext = file.suffix.lower()

        if ext in IMAGE_EXTENSIONS:
            image_files.append(file)
            extension_counter[ext] += 1


    print(f"\nDataset: {source.name}")
    print(f"Images found: {len(image_files)}")
    print(f"Formats: {dict(extension_counter)}")