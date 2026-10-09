import cv2
import os
import numpy as np
import random

IMAGE_FOLDER = "datasets/airmouse_openings/images/train"
LABEL_FOLDER = "datasets/airmouse_openings/labels/train"

# Pick a random image
image_files = [
    f for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

image_name = random.choice(image_files)

image_path = os.path.join(IMAGE_FOLDER, image_name)

label_name = os.path.splitext(image_name)[0] + ".txt"
label_path = os.path.join(LABEL_FOLDER, label_name)

image = cv2.imread(image_path)

height, width = image.shape[:2]

overlay = image.copy()

with open(label_path, "r") as file:

    for line in file:

        values = line.strip().split()

        class_id = int(values[0])

        coordinates = list(map(float, values[1:]))

        points = []

        for i in range(0, len(coordinates), 2):

            x_normalized = coordinates[i]
            y_normalized = coordinates[i + 1]

            x = int(x_normalized * width)
            y = int(y_normalized * height)

            points.append([x, y])

        points = np.array(points, dtype=np.int32)

        # Fill the opening region
        cv2.fillPoly(
            overlay,
            [points],
            (0, 255, 0)
        )

        # Draw boundary
        cv2.polylines(
            image,
            [points],
            True,
            (0, 255, 0),
            2
        )


# Make filled mask transparent
image = cv2.addWeighted(
    overlay,
    0.35,
    image,
    0.65,
    0
)

cv2.putText(
    image,
    image_name,
    (20, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255, 255, 255),
    2
)

cv2.imshow("YOLO Segmentation Label Verification", image)

print("Showing:", image_name)

cv2.waitKey(0)
cv2.destroyAllWindows()