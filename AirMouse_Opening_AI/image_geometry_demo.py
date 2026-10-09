import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tkinter import Tk, filedialog


# =========================================================
# MODEL
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "runs"
    / "airmouse"
    / "opening_seg_v2b"
    / "weights"
    / "best.pt"
)

print("Loading model:")
print(MODEL_PATH)

if not MODEL_PATH.exists():
    print("ERROR: Model not found")
    raise SystemExit

model = YOLO(str(MODEL_PATH))


# =========================================================
# SETTINGS
# =========================================================

CONFIDENCE = 0.12

X_TOLERANCE_RATIO = 0.08
Y_TOLERANCE_RATIO = 0.08

MIN_AREA_RATIO = 0.02

WINDOW_NAME = "AirMouse - Image Geometry Demo"


# =========================================================
# FILE PICKER
# =========================================================

root = Tk()
root.withdraw()

image_paths = filedialog.askopenfilenames(
    title="Select Gate / Opening Images",
    filetypes=[
        (
            "Images",
            "*.jpg *.jpeg *.png *.bmp *.webp"
        )
    ]
)

root.destroy()


if not image_paths:

    print("No images selected.")
    raise SystemExit


print()
print("Selected images:", len(image_paths))
print()


# =========================================================
# WINDOW
# =========================================================

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    WINDOW_NAME,
    1100,
    700
)


# =========================================================
# PROCESS EACH IMAGE
# =========================================================

for image_number, image_path in enumerate(
    image_paths,
    start=1
):

    print(
        f"[{image_number}/{len(image_paths)}]",
        image_path
    )


    # -----------------------------------------------------
    # LOAD IMAGE
    # -----------------------------------------------------

    frame = cv2.imread(
        image_path
    )

    if frame is None:

        print("Could not load image.")
        continue


    frame_h, frame_w = frame.shape[:2]


    # =====================================================
    # IMAGE / CAMERA CENTRE
    # =====================================================

    camera_cx = frame_w // 2
    camera_cy = frame_h // 2


    x_tolerance = int(
        frame_w * X_TOLERANCE_RATIO
    )

    y_tolerance = int(
        frame_h * Y_TOLERANCE_RATIO
    )


    # =====================================================
    # YOLO PREDICTION
    # =====================================================

    results = model.predict(
        frame,
        imgsz=640,
        conf=CONFIDENCE,
        device=0,
        verbose=False
    )

    result = results[0]


    guidance = "NO OPENING"

    chosen_confidence = 0.0


    # =====================================================
    # CHECK DETECTION
    # =====================================================

    if (
        result.masks is not None
        and result.boxes is not None
        and len(result.boxes) > 0
        and len(result.masks.xy) > 0
    ):

        confidences = (
            result.boxes.conf
            .detach()
            .cpu()
            .numpy()
        )


        # Choose highest-confidence detected opening
        best_index = int(
            np.argmax(confidences)
        )


        chosen_confidence = float(
            confidences[best_index]
        )


        # =================================================
        # ORIGINAL IMAGE POLYGON
        # =================================================

        polygon = result.masks.xy[
            best_index
        ]


        if (
            polygon is not None
            and len(polygon) >= 3
        ):

            binary_mask = np.zeros(
                (
                    frame_h,
                    frame_w
                ),
                dtype=np.uint8
            )


            points = np.round(
                polygon
            ).astype(np.int32)


            points[:, 0] = np.clip(
                points[:, 0],
                0,
                frame_w - 1
            )

            points[:, 1] = np.clip(
                points[:, 1],
                0,
                frame_h - 1
            )


            cv2.fillPoly(
                binary_mask,
                [points],
                255
            )


            # =============================================
            # FIND CONTOUR
            # =============================================

            contours, _ = cv2.findContours(
                binary_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )


            if contours:

                contour = max(
                    contours,
                    key=cv2.contourArea
                )


                area = cv2.contourArea(
                    contour
                )


                min_area = (
                    frame_w
                    * frame_h
                    * MIN_AREA_RATIO
                )


                if area >= min_area:


                    # =====================================
                    # BOUNDING GEOMETRY
                    # =====================================

                    x, y, w, h = cv2.boundingRect(
                        contour
                    )


                    # =====================================
                    # OPENING CENTRE
                    # =====================================

                    moments = cv2.moments(
                        contour
                    )


                    if moments["m00"] != 0:

                        opening_cx = int(
                            moments["m10"]
                            / moments["m00"]
                        )

                        opening_cy = int(
                            moments["m01"]
                            / moments["m00"]
                        )

                    else:

                        opening_cx = (
                            x + w // 2
                        )

                        opening_cy = (
                            y + h // 2
                        )


                    # =====================================
                    # ALIGNMENT ERROR
                    # =====================================

                    error_x = (
                        opening_cx
                        - camera_cx
                    )

                    error_y = (
                        opening_cy
                        - camera_cy
                    )


                    # =====================================
                    # DIRECTION
                    # =====================================

                    horizontal = ""
                    vertical = ""


                    if error_x < -x_tolerance:

                        horizontal = "LEFT"

                    elif error_x > x_tolerance:

                        horizontal = "RIGHT"


                    if error_y < -y_tolerance:

                        vertical = "UP"

                    elif error_y > y_tolerance:

                        vertical = "DOWN"


                    commands = []


                    if horizontal:

                        commands.append(
                            horizontal
                        )


                    if vertical:

                        commands.append(
                            vertical
                        )


                    if not commands:

                        guidance = (
                            "ALIGNED -> FORWARD"
                        )

                    else:

                        guidance = (
                            "MOVE "
                            + " + ".join(commands)
                        )


                    # =====================================
                    # GREEN SEGMENTATION MASK
                    # =====================================

                    overlay = frame.copy()

                    overlay[
                        binary_mask > 0
                    ] = (
                        0,
                        255,
                        0
                    )


                    frame = cv2.addWeighted(
                        overlay,
                        0.25,
                        frame,
                        0.75,
                        0
                    )


                    # =====================================
                    # OPENING CONTOUR
                    # =====================================

                    cv2.drawContours(
                        frame,
                        [contour],
                        -1,
                        (
                            0,
                            255,
                            0
                        ),
                        3
                    )


                    # =====================================
                    # BOUNDING BOX
                    # =====================================

                    cv2.rectangle(
                        frame,
                        (
                            x,
                            y
                        ),
                        (
                            x + w,
                            y + h
                        ),
                        (
                            0,
                            255,
                            0
                        ),
                        2
                    )


                    # =====================================
                    # OPENING CENTRE
                    # RED DOT
                    # =====================================

                    cv2.circle(
                        frame,
                        (
                            opening_cx,
                            opening_cy
                        ),
                        8,
                        (
                            0,
                            0,
                            255
                        ),
                        -1
                    )


                    # =====================================
                    # CENTRE ERROR LINE
                    # =====================================

                    cv2.line(
                        frame,
                        (
                            camera_cx,
                            camera_cy
                        ),
                        (
                            opening_cx,
                            opening_cy
                        ),
                        (
                            0,
                            255,
                            255
                        ),
                        2
                    )


                    # =====================================
                    # GEOMETRY INFORMATION
                    # =====================================

                    cv2.putText(
                        frame,
                        f"Opening: {w}px x {h}px",
                        (
                            20,
                            90
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (
                            255,
                            255,
                            255
                        ),
                        2
                    )


                    cv2.putText(
                        frame,
                        f"Error X: {error_x}px | Y: {error_y}px",
                        (
                            20,
                            120
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (
                            255,
                            255,
                            255
                        ),
                        2
                    )


                    cv2.putText(
                        frame,
                        f"Confidence: {chosen_confidence:.2f}",
                        (
                            20,
                            150
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (
                            255,
                            255,
                            255
                        ),
                        2
                    )


    # =====================================================
    # IMAGE CENTRE ALIGNMENT BOX
    # =====================================================

    cv2.rectangle(
        frame,
        (
            camera_cx - x_tolerance,
            camera_cy - y_tolerance
        ),
        (
            camera_cx + x_tolerance,
            camera_cy + y_tolerance
        ),
        (
            255,
            255,
            255
        ),
        3
    )


    # White image centre
    cv2.circle(
        frame,
        (
            camera_cx,
            camera_cy
        ),
        7,
        (
            255,
            255,
            255
        ),
        -1
    )


    # =====================================================
    # IMAGE NAME
    # =====================================================

    filename = Path(
        image_path
    ).name


    cv2.putText(
        frame,
        f"Image: {filename}",
        (
            20,
            35
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (
            255,
            255,
            255
        ),
        2
    )


    # =====================================================
    # GUIDANCE
    # =====================================================

    if guidance == "ALIGNED -> FORWARD":

        guidance_color = (
            0,
            255,
            0
        )

    else:

        guidance_color = (
            0,
            255,
            255
        )


    cv2.putText(
        frame,
        guidance,
        (
            20,
            frame_h - 35
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        guidance_color,
        3
    )


    # =====================================================
    # CONTROLS
    # =====================================================

    cv2.putText(
        frame,
        "SPACE/N = Next | Q = Quit",
        (
            20,
            frame_h - 75
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (
            255,
            255,
            255
        ),
        2
    )


    # =====================================================
    # SHOW RESULT
    # =====================================================

    cv2.imshow(
        WINDOW_NAME,
        frame
    )


    # Wait until user presses key
    while True:

        key = (
            cv2.waitKey(0)
            & 0xFF
        )


        # Q
        if key == ord("q"):

            cv2.destroyAllWindows()
            raise SystemExit


        # N or SPACE
        elif (
            key == ord("n")
            or key == 32
        ):

            break


# =========================================================
# END
# =========================================================

cv2.destroyAllWindows()

print()
print("All selected images processed.")