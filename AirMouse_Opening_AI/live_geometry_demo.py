import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import time
import threading


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
    print("ERROR: V2B best.pt not found")
    raise SystemExit

model = YOLO(str(MODEL_PATH))


# =========================================================
# LOW-LATENCY PHONE CAMERA
# =========================================================

class LatestFrameCamera:
    """
    Continuously reads the IP Webcam stream in a background
    thread and keeps ONLY the newest frame.

    This prevents old frames from building up and causing
    several seconds of latency.
    """

    def __init__(self, source):

        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(
                "Phone camera could not be opened"
            )

        # Some backends ignore this, but request minimum buffer.
        self.cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

        self.frame = None
        self.frame_id = 0

        self.lock = threading.Lock()
        self.running = True

        self.thread = threading.Thread(
            target=self._reader,
            daemon=True
        )

        self.thread.start()


    def _reader(self):

        while self.running:

            ret, frame = self.cap.read()

            if not ret or frame is None:
                time.sleep(0.005)
                continue

            # Replace the previous frame.
            # Never build a queue.
            with self.lock:

                self.frame = frame
                self.frame_id += 1


    def read(self):

        with self.lock:

            if self.frame is None:
                return False, None, -1

            return (
                True,
                self.frame.copy(),
                self.frame_id
            )


    def release(self):

        self.running = False

        self.cap.release()

        if self.thread.is_alive():
            self.thread.join(
                timeout=1.0
            )


# =========================================================
# CAMERA
# =========================================================

# IP Webcam phone server:
# phone side = port 8080
#
# ADB:
# adb.exe forward tcp:18080 tcp:8080

CAMERA_URL = 0

print("\nConnecting to phone camera...")

try:

    camera = LatestFrameCamera(
        CAMERA_URL
    )

except RuntimeError as error:

    print(error)
    raise SystemExit

print("Phone USB camera connected")


# =========================================================
# SETTINGS
# =========================================================

CONFIDENCE = 0.15

# Alignment tolerance around camera centre
X_TOLERANCE_RATIO = 0.08
Y_TOLERANCE_RATIO = 0.08

# Ignore extremely tiny predicted openings
MIN_AREA_RATIO = 0.02

WINDOW_NAME = "AirMouse - Opening Geometry Demo"


# =========================================================
# WINDOW
# =========================================================

# WINDOW_NORMAL lets Windows scale the image so the whole
# frame fits on screen without changing geometry calculations.

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    WINDOW_NAME,
    1100,
    650
)


# =========================================================
# FPS
# =========================================================

previous_time = time.time()

last_frame_id = -1


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    # -----------------------------------------------------
    # GET LATEST FRAME
    # -----------------------------------------------------

    ret, frame, frame_id = camera.read()

    if not ret:
        time.sleep(0.005)
        continue


    # Don't run YOLO repeatedly on the exact same frame
    if frame_id == last_frame_id:
        time.sleep(0.001)
        continue

    last_frame_id = frame_id


    # -----------------------------------------------------
    # FRAME SIZE
    # -----------------------------------------------------

    frame_h, frame_w = frame.shape[:2]


    # Actual centre of CURRENT frame
    camera_cx = frame_w // 2
    camera_cy = frame_h // 2


    # -----------------------------------------------------
    # ALIGNMENT TOLERANCE
    # -----------------------------------------------------

    x_tolerance = int(
        frame_w * X_TOLERANCE_RATIO
    )

    y_tolerance = int(
        frame_h * Y_TOLERANCE_RATIO
    )


    # -----------------------------------------------------
    # YOLO V2B INFERENCE
    # -----------------------------------------------------

    results = model.predict(
        frame,
        imgsz=640,
        conf=CONFIDENCE,
        device=0,
        verbose=False
    )

    result = results[0]


    # -----------------------------------------------------
    # DEFAULT STATE
    # -----------------------------------------------------

    guidance = "NO OPENING"

    chosen_confidence = 0.0


    # =====================================================
    # SEGMENTATION DETECTION
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


        # -------------------------------------------------
        # DEMO:
        # choose highest-confidence opening
        # -------------------------------------------------

        best_index = int(
            np.argmax(confidences)
        )

        chosen_confidence = float(
            confidences[best_index]
        )


        # -------------------------------------------------
        # CORRECT ORIGINAL-FRAME POLYGON
        # -------------------------------------------------
        #
        # masks.xy gives coordinates in the original
        # camera frame, so no manual mask stretching.
        # -------------------------------------------------

        polygon = result.masks.xy[
            best_index
        ]


        if (
            polygon is not None
            and len(polygon) >= 3
        ):

            # ---------------------------------------------
            # BUILD BINARY MASK AT ORIGINAL RESOLUTION
            # ---------------------------------------------

            binary_mask = np.zeros(
                (frame_h, frame_w),
                dtype=np.uint8
            )


            points = np.round(
                polygon
            ).astype(np.int32)


            # Keep all points inside image
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
            # FIND OPENING CONTOUR
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


                # =========================================
                # IGNORE VERY SMALL FALSE DETECTIONS
                # =========================================

                if area >= min_area:


                    # =====================================
                    # OPENING WIDTH / HEIGHT
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
                    # DIRECTION LOGIC
                    # =====================================

                    horizontal = ""
                    vertical = ""


                    # Horizontal
                    if error_x < -x_tolerance:

                        horizontal = "LEFT"

                    elif error_x > x_tolerance:

                        horizontal = "RIGHT"


                    # Vertical
                    if error_y < -y_tolerance:

                        vertical = "UP"

                    elif error_y > y_tolerance:

                        vertical = "DOWN"


                    # =====================================
                    # FINAL GUIDANCE
                    # =====================================

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
                    # GREEN OPENING MASK
                    # =====================================

                    overlay = frame.copy()

                    overlay[
                        binary_mask > 0
                    ] = (0, 255, 0)


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
                        (0, 255, 0),
                        3
                    )


                    # =====================================
                    # OPENING BOUNDING RECTANGLE
                    # =====================================

                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        (0, 255, 0),
                        2
                    )


                    # =====================================
                    # OPENING CENTRE = RED
                    # =====================================

                    cv2.circle(
                        frame,
                        (
                            opening_cx,
                            opening_cy
                        ),
                        8,
                        (0, 0, 255),
                        -1
                    )


                    # =====================================
                    # LINE:
                    # CAMERA CENTRE -> OPENING CENTRE
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
                        (0, 255, 255),
                        2
                    )


                    # =====================================
                    # GEOMETRY INFO
                    # =====================================

                    cv2.putText(
                        frame,
                        f"Opening: {w}px x {h}px",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )


                    cv2.putText(
                        frame,
                        f"Error X: {error_x}px | Y: {error_y}px",
                        (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )


                    cv2.putText(
                        frame,
                        f"Confidence: {chosen_confidence:.2f}",
                        (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )


    # =====================================================
    # CAMERA CENTRE / ALIGNMENT BOX
    # =====================================================
    #
    # Always calculated from the real current frame:
    #
    # center_x = frame width / 2
    # center_y = frame height / 2
    #
    # So it should always be geometrically centred.
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
        (255, 255, 255),
        3
    )


    cv2.circle(
        frame,
        (
            camera_cx,
            camera_cy
        ),
        7,
        (255, 255, 255),
        -1
    )


    # =====================================================
    # FPS
    # =====================================================

    current_time = time.time()

    fps = 1.0 / max(
        current_time - previous_time,
        0.0001
    )

    previous_time = current_time


    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # =====================================================
    # GUIDANCE TEXT
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
            frame_h - 30
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        guidance_color,
        3
    )


    # =====================================================
    # DISPLAY
    # =====================================================

    cv2.imshow(
        WINDOW_NAME,
        frame
    )


    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================================================
# CLEANUP
# =========================================================

camera.release()

cv2.destroyAllWindows()

print("Geometry demo closed")