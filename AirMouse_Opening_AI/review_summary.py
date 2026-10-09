from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

ROOT = BASE_DIR / "datasets" / "curated_v2" / "candidates"

for name in [
    "positive",
    "negative",
    "ambiguous",
    "rejected"
]:
    folder = ROOT / name

    count = 0

    if folder.exists():
        count = len([
            f for f in folder.iterdir()
            if f.suffix.lower() in {
                ".jpg", ".jpeg", ".png", ".webp"
            }
        ])

    print(f"{name:10} : {count}")