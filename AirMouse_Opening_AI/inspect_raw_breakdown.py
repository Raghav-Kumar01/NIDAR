from pathlib import Path
from collections import Counter

RAW_ROOT = Path(r"F:\AirMouseRaw")

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

LABEL_EXTENSIONS = {
    ".txt",
    ".csv",
    ".yaml",
    ".yml",
    ".json"
}


print("\nAIRMOUSE RAW DATASET BREAKDOWN")
print("=" * 70)


for source in sorted(RAW_ROOT.iterdir()):

    if not source.is_dir():
        continue

    print(f"\n\nDATASET: {source.name}")
    print("-" * 70)

    folder_stats = {}


    for file in source.rglob("*"):

        if not file.is_file():
            continue

        ext = file.suffix.lower()

        if ext not in IMAGE_EXTENSIONS and ext not in LABEL_EXTENSIONS:
            continue


        relative_parent = file.parent.relative_to(source)

        key = str(relative_parent)

        if key not in folder_stats:

            folder_stats[key] = {
                "images": 0,
                "labels": 0,
                "formats": Counter()
            }


        if ext in IMAGE_EXTENSIONS:

            folder_stats[key]["images"] += 1
            folder_stats[key]["formats"][ext] += 1

        elif ext in LABEL_EXTENSIONS:

            folder_stats[key]["labels"] += 1


    for folder, stats in sorted(folder_stats.items()):

        if stats["images"] == 0 and stats["labels"] == 0:
            continue

        print(
            f"{folder}\n"
            f"    Images: {stats['images']}\n"
            f"    Labels/configs: {stats['labels']}\n"
            f"    Formats: {dict(stats['formats'])}"
        )