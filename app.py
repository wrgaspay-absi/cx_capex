"""
Signature Detection API (FastAPI)
- Loads a trained YOLO model at startup.
- Accepts image files or PDFs, converts the first PDF page to an image.
- Runs simple preprocessing, runs YOLO detection, and returns bounding boxes + confidence.
- Includes helpful comments and small helper utilities to make future edits easier.
"""

import os
import tempfile
import logging
from typing import List, Dict, Any, Tuple

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import cv2
from pdf2image import convert_from_bytes
from ultralytics import YOLO

# ---------------------------
# Configuration / Constants
# ---------------------------
DEFAULT_PORT = int(os.environ.get("PORT", 7860))  # default for local or HF Spaces
BEST_MODEL_PATH = os.environ.get("BEST_MODEL_PATH", "best.pt")
PDF_DPI = 300  # DPI used when converting PDFs to images
YOLO_CONF_THRESHOLD = float(os.environ.get("YOLO_CONF_THRESHOLD", 0.6))
REGION_BELOW_SIG_Y_OFFSET = 10   # start y offset below signature bounding box
REGION_BELOW_SIG_HEIGHT = 100    # height of the region below signature to crop for OCR

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("signature_detection_api")

# ---------------------------
# App initialization
# ---------------------------
app = FastAPI(title="Signature Detection API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------
# Helper utilities
# ---------------------------
def clamp(v: int, lo: int, hi: int) -> int:
    """Clamp integer v to the inclusive range [lo, hi]."""
    return max(lo, min(v, hi))


def box_xyxy_from_ultralytics(box) -> Tuple[int, int, int, int]:
    """
    Convert an ultralytics box object to Python ints for x_min, y_min, x_max, y_max.
    Note: this expects a single box (one row) and returns ints.
    """
    xyxy = box.xyxy.cpu().numpy().astype(int)[0]  # shape (4,)
    x_min, y_min, x_max, y_max = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
    return x_min, y_min, x_max, y_max


# ---------------------------
# Model lifecycle events
# ---------------------------
@app.on_event("startup")
def load_model_at_startup():
    """
    Load the YOLO model once at application startup and store it in app.state
    so endpoints can reference it easily.
    """
    try:
        logger.info("Loading YOLO model from: %s", BEST_MODEL_PATH)
        model = YOLO(BEST_MODEL_PATH)
        # store the model instance on the app state for future reference
        app.state.model = model
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.exception("Failed to load model at startup: %s", e)
        # re-raise so the server fails fast if the model can't be loaded
        raise RuntimeError(f"Failed to load model: {e}") from e


# ---------------------------
# Routes / Endpoints
# ---------------------------
@app.get("/")
def read_root() -> Dict[str, str]:
    """Simple health endpoint."""
    return {"status": "ok", "message": "Signature Detection API is running"}


@app.post("/detect")
async def detect_signatures(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Main detection endpoint.
    Accepts:
      - image file (png/jpg/jpeg, etc.)
      - PDF (first page is used)
    Returns JSON with:
      - filename
      - signatures_detected (count)
      - detections: list of { signature_box: [x_min,y_min,x_max,y_max], confidence: float }
    """
    # Ensure model is loaded
    model: YOLO = getattr(app.state, "model", None)
    if model is None:
        logger.error("Model not found in app state.")
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Read input bytes
    file_bytes = await file.read()
    file_ext = (file.filename or "").split(".")[-1].lower()

    image = None

    # --- Step 1: handle PDF or image ---
    try:
        if file_ext == "pdf":
            # Convert first page of PDF to an image (PIL.Image), then save/load into OpenCV
            pages = convert_from_bytes(file_bytes, dpi=PDF_DPI)
            if len(pages) == 0:
                raise ValueError("PDF contains no pages")
            # Save first page to a temporary PNG file then read with cv2
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
                pages[0].save(tmp_path, "PNG")
            image = cv2.imread(tmp_path)
            # remove the temp file
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        else:
            # Treat as image bytes (jpg/png)
            np_arr = np.frombuffer(file_bytes, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.exception("Failed to convert file to image: %s", e)
        raise HTTPException(status_code=400, detail=f"Unable to process uploaded file: {e}")

    if image is None:
        logger.error("No image could be read from the uploaded file.")
        raise HTTPException(status_code=400, detail="Unable to read image from uploaded file")

    img_h, img_w = image.shape[0], image.shape[1]

    # --- Step 2: Preprocessing (grayscale + contrast enhancement) ---
    # Keep a color copy for bbox indices; run preprocessing for detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Contrast / brightness tweak - tuned by alpha/beta
    alpha = 1.5  # contrast
    beta = 30    # brightness
    enhanced = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    preprocessed_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    # --- Step 3: Run YOLO detection ---
    try:
        # call the model with the preprocessed image and confidence threshold
        results = model(preprocessed_bgr, conf=YOLO_CONF_THRESHOLD)
    except Exception as e:
        logger.exception("YOLO model inference failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Detection failed: {e}")

    detection_data: List[Dict[str, Any]] = []

    # results[0].boxes contains the detected boxes for the first (and only) image
    boxes = getattr(results[0], "boxes", None)
    if boxes is None:
        boxes = []

    for box in boxes:
        try:
            x_min, y_min, x_max, y_max = box_xyxy_from_ultralytics(box)
            # clamp to image bounds to avoid indexing errors when cropping
            x_min = clamp(x_min, 0, img_w - 1)
            x_max = clamp(x_max, 0, img_w - 1)
            y_min = clamp(y_min, 0, img_h - 1)
            y_max = clamp(y_max, 0, img_h - 1)

            # Prepare a small region below the detected signature box (useful for OCR)
            region_y_start = clamp(y_max + REGION_BELOW_SIG_Y_OFFSET, 0, img_h - 1)
            region_y_end = clamp(region_y_start + REGION_BELOW_SIG_HEIGHT, 0, img_h)
            region_x_start = x_min
            region_x_end = x_max

            # Ensure the crop region is valid; if not, produce an empty crop
            if region_y_start >= region_y_end or region_x_start >= region_x_end:
                cropped_region = np.zeros((0, 0, 3), dtype=np.uint8)
            else:
                cropped_region = image[region_y_start:region_y_end, region_x_start:region_x_end]

            # Optionally, you could run pytesseract here on cropped_region to extract text.
            # For now we only return coordinates + confidence to keep endpoint fast and predictable.

            detection_data.append({
                "signature_box": [int(x_min), int(y_min), int(x_max), int(y_max)],
                "confidence": float(getattr(box, "conf", 0.0)),
                # describe region used for OCR (useful for downstream processing)
                "ocr_region": [int(region_x_start), int(region_y_start), int(region_x_end), int(region_y_end)],
            })
        except Exception as e:
            # continue processing other boxes but log the failure
            logger.exception("Failed processing a detection box: %s", e)

    return {
        "filename": file.filename,
        "image_width": img_w,
        "image_height": img_h,
        "signatures_detected": len(detection_data),
        "detections": detection_data
    }


# ---------------------------
# Standalone run (development)
# ---------------------------
if __name__ == "__main__":
    # Development: start uvicorn programmatically.
    # In production you might run `uvicorn app:app --host 0.0.0.0 --port $PORT`
    uvicorn.run("app:app", host="0.0.0.0", port=DEFAULT_PORT, reload=False)