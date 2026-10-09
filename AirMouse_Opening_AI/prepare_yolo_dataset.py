import os
import csv
import shutil
import cv2
from collections import defaultdict


# ---------------------------------------------------------
# 1. Where the original MonoRaceGate dataset is located
# ---------------------------------------------------------

SOURCE_ROOT = "datasets/MonoRaceGate/data"

# We will create our own converted dataset here.
OUTPUT_ROOT = "datasets/airmouse_openings"


# ---------------------------------------------------------
# 2. Split whole flight sequences, NOT individual frames
# ---------------------------------------------------------

SPLITS = {

    "train": [
        "adnec_s1_f1",
        "adnec_s1_f2",
        "dhl_s1_f1",
        "red_s1_f1",
        "red_s1_f2",
    ],

    "val": [
        "marina_s1_f1",
    ],

    "test": [
        "red_s1_f3",
    ],
}


# ---------------------------------------------------------
# 3. Create required YOLO folders
# ---------------------------------------------------------

for split in ["train", "val", "test"]:

    os.makedirs(
        os.path.join(OUTPUT_ROOT, "images", split),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(OUTPUT_ROOT, "labels", split),
        exist_ok=True
    )


# ---------------------------------------------------------
# 4. Process one flight sequence
# ---------------------------------------------------------

def process_sequence(sequence_name, split):

    sequence_path = os.path.join(SOURCE_ROOT, sequence_name)
    corners_path = os.path.join(sequence_path, "corners.csv")

    print(f"\nProcessing {sequence_name} → {split}")

    # Store all gates belonging to each image.
    #
    # Example:
    #
    # img_101.png:
    #     gate 1 corners
    #     gate 2 corners
    #
    annotations = defaultdict(list)

    with open(corners_path, "r") as file:

        reader = csv.reader(file)

        for row in reader:

            image_name = row[0]

            values = list(map(float, row[1:]))

            points = [
                (values[0], values[1]),
                (values[2], values[3]),
                (values[4], values[5]),
                (values[6], values[7]),
            ]

            annotations[image_name].append(points)


    # Find every PNG image in this sequence.
    image_files = [
        file
        for file in os.listdir(sequence_path)
        if file.lower().endswith((".png", ".jpg", ".jpeg"))
    ]


    for image_name in image_files:

        source_image = os.path.join(sequence_path, image_name)

        image = cv2.imread(source_image)

        if image is None:
            print("Could not read:", source_image)
            continue

        height, width = image.shape[:2]


        # -------------------------------------------------
        # Give every image a UNIQUE filename.
        #
        # Different sequences can both contain img_1.png.
        # Without renaming, one would overwrite another.
        # -------------------------------------------------

        new_image_name = f"{sequence_name}_{image_name}"

        output_image = os.path.join(
            OUTPUT_ROOT,
            "images",
            split,
            new_image_name
        )

        shutil.copy2(source_image, output_image)


        # Corresponding YOLO label file.
        label_name = os.path.splitext(new_image_name)[0] + ".txt"

        output_label = os.path.join(
            OUTPUT_ROOT,
            "labels",
            split,
            label_name
        )


        # -------------------------------------------------
        # Convert each gate polygon to YOLO format
        # -------------------------------------------------

        with open(output_label, "w") as label_file:

            for points in annotations.get(image_name, []):

                normalized_points = []

                for x, y in points:

                    normalized_x = x / width
                    normalized_y = y / height

                    normalized_points.extend([
                        normalized_x,
                        normalized_y
                    ])


                # We have only one class:
                #
                # class 0 = traversable_opening
                #
                line = "0 " + " ".join(
                    f"{value:.6f}"
                    for value in normalized_points
                )

                label_file.write(line + "\n")


        print(f"Converted: {new_image_name}")


# ---------------------------------------------------------
# 5. Run conversion for every split
# ---------------------------------------------------------

for split, sequences in SPLITS.items():

    for sequence in sequences:

        process_sequence(sequence, split)


print("\n====================================")
print("YOLO dataset conversion completed!")
print("====================================")
