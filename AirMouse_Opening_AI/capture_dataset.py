import cv2
import os

os.makedirs("dataset_raw", exist_ok=True)

cap = cv2.VideoCapture(0)

count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.putText(
        frame,
        "SPACE = Save Image | Q = Quit",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.imshow("AirMouse Dataset Collector", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):
        filename = f"dataset_raw/opening_{count:04d}.jpg"
        cv2.imwrite(filename, frame)

        print("Saved:", filename)

        count += 1

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()