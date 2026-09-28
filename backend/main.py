from pathlib import Path
import base64

import cv2
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ultralytics import YOLO


# =========================================================
# PATH CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PERSON_MODEL_PATH = BASE_DIR / "yolov8n.pt"
HELMET_MODEL_PATH = BASE_DIR / "models" / "helmet.pt"


# =========================================================
# MODEL CONFIGURATION
# =========================================================

PERSON_CONF = 0.45
HELMET_CONF = 0.05

HEAD_REGION_RATIO = 0.45


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="VisionGuard AI",
    description="AI-powered Helmet Detection System",
    version="3.0.0"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# LOAD MODELS
# =========================================================

print()
print("=" * 60)
print("VISIONGUARD AI")
print("=" * 60)
print("Loading AI models...")
print()

if not PERSON_MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Person model not found: {PERSON_MODEL_PATH}"
    )


if not HELMET_MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Helmet model not found: {HELMET_MODEL_PATH}"
    )


person_model = YOLO(
    str(PERSON_MODEL_PATH)
)

helmet_model = YOLO(
    str(HELMET_MODEL_PATH)
)


print("Person model loaded:")
print(PERSON_MODEL_PATH)

print()

print("Helmet model loaded:")
print(HELMET_MODEL_PATH)

print()

print("Helmet classes:")
print(helmet_model.names)

print()
print("=" * 60)
print("VISIONGUARD AI BACKEND READY")
print("=" * 60)
print()


# =========================================================
# HELMET DETECTION
# =========================================================

def detect_helmet(head_crop):

    if head_crop is None:
        return None

    if head_crop.size == 0:
        return None


    results = helmet_model(
        head_crop,
        conf=HELMET_CONF,
        iou=0.45,
        imgsz=640,
        max_det=3,
        verbose=False
    )


    result = results[0]


    if result.boxes is None:
        return None


    if len(result.boxes) == 0:
        return None


    best_detection = None


    for box in result.boxes:

        class_id = int(
            box.cls[0]
        )

        confidence = float(
            box.conf[0]
        )


        if class_id not in (0, 1):
            continue


        detection = {
            "class": class_id,
            "confidence": confidence
        }


        if (
            best_detection is None
            or confidence > best_detection["confidence"]
        ):

            best_detection = detection


    return best_detection


# =========================================================
# PROCESS FRAME
# =========================================================

def process_frame(frame):

    total_persons = 0
    with_helmet = 0
    without_helmet = 0
    checking = 0

    detections = []


    # -----------------------------------------------------
    # PERSON DETECTION
    # -----------------------------------------------------

    results = person_model.predict(
        frame,
        classes=[0],
        conf=PERSON_CONF,
        imgsz=640,
        verbose=False
    )


    result = results[0]


    # -----------------------------------------------------
    # PROCESS DETECTED PERSONS
    # -----------------------------------------------------

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )


            # COCO class 0 = person
            if class_id != 0:
                continue


            person_confidence = float(
                box.conf[0]
            )


            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            # Keep coordinates inside image

            x1 = max(
                0,
                x1
            )

            y1 = max(
                0,
                y1
            )

            x2 = min(
                frame.shape[1],
                x2
            )

            y2 = min(
                frame.shape[0],
                y2
            )


            width = x2 - x1
            height = y2 - y1


            if width <= 0 or height <= 0:
                continue


            total_persons += 1


            # -------------------------------------------------
            # HEAD REGION
            # -------------------------------------------------

            head_y2 = y1 + int(
                height * HEAD_REGION_RATIO
            )


            head_y2 = min(
                head_y2,
                frame.shape[0]
            )


            head_crop = frame[
                y1:head_y2,
                x1:x2
            ]


            # -------------------------------------------------
            # HELMET DETECTION
            # -------------------------------------------------

            helmet_detection = detect_helmet(
                head_crop
            )


            status = "Checking..."

            helmet_confidence = 0.0


            if helmet_detection is not None:

                helmet_class = (
                    helmet_detection["class"]
                )


                helmet_confidence = (
                    helmet_detection["confidence"]
                )


                # Class 0 = With Helmet
                if helmet_class == 0:

                    status = "With Helmet"

                    with_helmet += 1


                # Class 1 = Without Helmet
                elif helmet_class == 1:

                    status = "Without Helmet"

                    without_helmet += 1


            else:

                checking += 1


            # -------------------------------------------------
            # BOX COLOR
            # -------------------------------------------------

            if status == "With Helmet":

                box_color = (
                    0,
                    200,
                    0
                )


            elif status == "Without Helmet":

                box_color = (
                    0,
                    0,
                    255
                )


            else:

                box_color = (
                    0,
                    165,
                    255
                )


            # -------------------------------------------------
            # PERSON BOX
            # -------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                2
            )


            # -------------------------------------------------
            # LABEL
            # -------------------------------------------------

            label = (
                f"{status} | "
                f"{helmet_confidence:.2f}"
            )


            label_y = max(
                25,
                y1 - 10
            )


            cv2.putText(
                frame,
                label,
                (
                    x1,
                    label_y
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                box_color,
                2
            )


            # -------------------------------------------------
            # SAVE DETECTION INFORMATION
            # -------------------------------------------------

            detections.append({

                "person_confidence": round(
                    person_confidence,
                    3
                ),

                "status": status,

                "helmet_confidence": round(
                    helmet_confidence,
                    3
                ),

                "box": [
                    x1,
                    y1,
                    x2,
                    y2
                ]

            })


    # =====================================================
    # COMPLIANCE
    # =====================================================

    if total_persons > 0:

        compliance = round(
            (
                with_helmet
                / total_persons
            ) * 100,
            2
        )

    else:

        compliance = 0


    # =====================================================
    # RETURN
    # =====================================================

    return (
        frame,
        {
            "total_persons": total_persons,

            "with_helmet": with_helmet,

            "without_helmet": without_helmet,

            "checking": checking,

            "compliance": compliance
        },

        detections
    )


# =========================================================
# ENCODE IMAGE
# =========================================================

def encode_image(frame):

    success, encoded_image = cv2.imencode(
        ".jpg",
        frame,
        [
            int(
                cv2.IMWRITE_JPEG_QUALITY
            ),
            80
        ]
    )


    if not success:

        raise HTTPException(
            status_code=500,
            detail="Could not encode result image."
        )


    image_base64 = base64.b64encode(
        encoded_image.tobytes()
    ).decode("utf-8")


    return (
        "data:image/jpeg;base64,"
        + image_base64
    )


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():

    return {

        "message":
            "VisionGuard AI Backend is running",

        "status":
            "online",

        "version":
            "3.0.0"

    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():

    return {

        "status":
            "healthy",

        "person_model":
            PERSON_MODEL_PATH.exists(),

        "helmet_model":
            HELMET_MODEL_PATH.exists()

    }


# =========================================================
# MODEL INFORMATION
# =========================================================

@app.get("/model-info")
def model_info():

    return {

        "person_model":
            "YOLOv8n",

        "helmet_model":
            "helmet.pt",

        "helmet_classes":
            helmet_model.names

    }


# =========================================================
# IMAGE DETECTION
# =========================================================

@app.post("/detect")
async def detect(
    file: UploadFile = File(...)
):

    allowed_types = [
        "image/jpeg",
        "image/png",
        "image/jpg",
        "image/webp"
    ]


    if file.content_type not in allowed_types:

        raise HTTPException(
            status_code=400,
            detail=
                "Please upload JPG, PNG or WEBP image."
        )


    contents = await file.read()


    if not contents:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )


    image_array = np.frombuffer(
        contents,
        dtype=np.uint8
    )


    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )


    if frame is None:

        raise HTTPException(
            status_code=400,
            detail="Could not read image."
        )


    # Process image

    processed_frame, statistics, detections = (
        process_frame(frame)
    )


    # Encode result

    image = encode_image(
        processed_frame
    )


    return {

        "success": True,

        "statistics":
            statistics,

        "detections":
            detections,

        "image":
            image

    }


# =========================================================
# LIVE CAMERA DETECTION
# =========================================================

@app.post("/live-detect")
async def live_detect(
    file: UploadFile = File(...)
):

    allowed_types = [
        "image/jpeg",
        "image/png",
        "image/jpg",
        "image/webp"
    ]


    if file.content_type not in allowed_types:

        raise HTTPException(
            status_code=400,
            detail="Invalid image format."
        )


    contents = await file.read()


    if not contents:

        raise HTTPException(
            status_code=400,
            detail="Empty camera frame received."
        )


    image_array = np.frombuffer(
        contents,
        dtype=np.uint8
    )


    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )


    if frame is None:

        raise HTTPException(
            status_code=400,
            detail="Could not decode camera frame."
        )


    # Process camera frame

    processed_frame, statistics, detections = (
        process_frame(frame)
    )


    # Encode processed frame

    image = encode_image(
        processed_frame
    )


    return {

        "success": True,

        "statistics":
            statistics,

        "detections":
            detections,

        "image":
            image

    }
from backend.quality_api import router as quality_router
from backend.quality_db import init_db

init_db()w
app.include_router(quality_router)
