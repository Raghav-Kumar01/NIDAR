import cv2
import csv
import os

DATASET_FOLDER = "datasets/MonoRaceGate/data/adnec_s1_f1"
IMAGE_NAME = "img_101.png"

image_path = os.path.join(DATASET_FOLDER, IMAGE_NAME)
csv_path = os.path.join(DATASET_FOLDER, "corners.csv")

image = cv2.imread(image_path)

with open(csv_path, "r") as file:
    reader = csv.reader(file)

    for row in reader:

        if row[0] != IMAGE_NAME:
            continue

        numbers = list(map(int, row[1:]))

        points = [
            (numbers[0], numbers[1]),
            (numbers[2], numbers[3]),
            (numbers[4], numbers[5]),
            (numbers[6], numbers[7])
        ]

        # Draw the four corner points
        for point in points:
            cv2.circle(image, point, 6, (0, 255, 0), -1)

        # Connect the corners
        for i in range(4):
            start = points[i]
            end = points[(i + 1) % 4]

            cv2.line(image, start, end, (0, 255, 0), 2)

cv2.imshow("Gate Corner Verification", image)
cv2.waitKey(0)
cv2.destroyAllWindows()