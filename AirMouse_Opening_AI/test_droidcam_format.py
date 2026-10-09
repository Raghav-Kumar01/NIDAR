import cv2
import numpy as np

CAMERA_INDEX = 1

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_MSMF
)

# IMPORTANT:
# OpenCV ko automatic RGB conversion mat karne do
cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

if not cap.isOpened():
    print("Camera could not be opened")
    raise SystemExit

print("Camera opened")

while True:

    ret, frame = cap.read()

    if not ret or frame is None:
        print("Frame read failed")
        continue

    print(
        "Shape:", frame.shape,
        "dtype:", frame.dtype,
        "min:", frame.min(),
        "max:", frame.max()
    )

    output = None

    # -------------------------------------------------
    # Most common DroidCam / virtual webcam format:
    # YUY2 / YUYV
    # -------------------------------------------------

    if len(frame.shape) == 3 and frame.shape[2] == 2:

        try:
            output = cv2.cvtColor(
                frame,
                cv2.COLOR_YUV2BGR_YUY2
            )

        except Exception:
            pass


    # If backend already gives 3-channel data,
    # show it directly for inspection
    elif len(frame.shape) == 3 and frame.shape[2] == 3:

        output = frame


    if output is not None:

        cv2.imshow(
            "DroidCam Decoded",
            output
        )

    else:

        print(
            "Unexpected frame format:",
            frame.shape
        )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()