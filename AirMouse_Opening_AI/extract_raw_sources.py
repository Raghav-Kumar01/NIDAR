from pathlib import Path
import zipfile

BASE_DIR = Path(__file__).resolve().parent

RAW_ROOT = BASE_DIR / "datasets" / "raw_sources"

# Short external extraction location
EXTRACT_ROOT = Path(r"F:\AirMouseRaw")

EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)


for source_folder in RAW_ROOT.iterdir():

    if not source_folder.is_dir():
        continue

    print("\n--------------------------------")
    print("Checking:", source_folder.name)
    print("--------------------------------")

    zip_files = list(source_folder.glob("*.zip"))

    if not zip_files:
        print("No ZIP file found.")
        continue

    for zip_path in zip_files:

        destination = EXTRACT_ROOT / source_folder.name

        complete_marker = destination / ".extraction_complete"

        # Only skip if extraction actually finished previously
        if complete_marker.exists():
            print("Already completely extracted:", zip_path.name)
            continue

        print("Extracting:", zip_path.name)
        print("Destination:", destination)

        destination.mkdir(
            parents=True,
            exist_ok=True
        )

        try:

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(destination)

            # Marker proves extraction finished
            complete_marker.touch()

            print("Done!")

        except Exception as error:

            print("Extraction failed:")
            print(error)

            print(
                "\nPartial files may exist inside:",
                destination
            )

            break