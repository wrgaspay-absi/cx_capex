"""
Signature Detection API
-----------------------
FastAPI service for detecting signatures in images or PDFs using a trained YOLO model.

Features:
- Supports image files (JPG, PNG, etc.) and PDFs (auto-converts first page to image)
- Preprocessing (contrast + brightness enhancement) to improve detection
- YOLO detection of signatures with bounding boxes + confidence scores
- Future-ready: OCR integration and multi-page PDF support
- CORS enabled for easy integration with web frontends
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
import os
import tempfile
import uvicorn
import numpy as np
import cv2
import torch
import pytesseract
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pdf2image import convert_from_bytes
from ultralytics import YOLO


# -----------------------------------------------------------------------------
# APP INITIALIZATION
# -----------------------------------------------------------------------------
# Create a FastAPI instance
app = FastAPI(title="Signature Detection API", version="1.0.0")

# Enable CORS so the API can be accessed from browsers or other domains
# For production, replace ["*"] with ["https://your-frontend-domain.com"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# MODEL LOADING
# -----------------------------------------------------------------------------
# Load the trained YOLO model (make sure best.pt exists in the same folder)
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)


# -----------------------------------------------------------------------------
# ROOT ENDPOINT (health check)
# -----------------------------------------------------------------------------
@app.get("/")
def read_root():
    """
    Health check endpoint.
    Useful for verifying that the API is running.
    """
    return {"status": "ok", "message": "API is running"}


# -----------------------------------------------------------------------------
# SIGNATURE DETECTION ENDPOINT
# -----------------------------------------------------------------------------
@app.post("/detect")
async def detect_signatures(file: UploadFile = File(...)):
    """
    Detect signatures from an uploaded file.
    
    Steps:
    1. If PDF → convert first page into an image
    2. If image → read directly
    3. Preprocess the image (contrast + brightness enhancement)
    4. Run YOLO model to detect signatures
    5. Return bounding boxes, confidence scores, and file info
    """

    # Create a temporary directory for processing
    with tempfile.TemporaryDirectory() as tmpdir:
        # Read the uploaded file as bytes
        file_bytes = await file.read()
        file_ext = file.filename.split(".")[-1].lower()

        image = None

        # ---------------------------------------------------------------
        # Step 1: Handle PDFs (convert first page to image)
        # ---------------------------------------------------------------
        if file_ext == "pdf":
            pages = convert_from_bytes(file_bytes, dpi=300)   # Convert PDF → images
            img_path = os.path.join(tmpdir, "page1.png")      # Save first page
            pages[0].save(img_path, "PNG")
            image = cv2.imread(img_path)                      # Load into OpenCV

        # ---------------------------------------------------------------
        # Step 2: Handle image files (JPG, PNG, etc.)
        # ---------------------------------------------------------------
        else:
            np_arr = np.frombuffer(file_bytes, np.uint8)      # Convert to numpy array
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)    # Decode image

        # If file is unreadable, return error
        if image is None:
            return {"error": "Unable to process file"}

        # ---------------------------------------------------------------
        # Step 3: Preprocessing (improves detection performance)
        # ---------------------------------------------------------------
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)        # Convert to grayscale
        alpha = 1.5   # Contrast factor
        beta = 30     # Brightness adjustment
        enhanced = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
        preprocessed_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        # ---------------------------------------------------------------
        # Step 4: YOLO detection
        # ---------------------------------------------------------------
        results = model(preprocessed_bgr, conf=0.6)  # Run detection with threshold

        detection_data = []

        # Loop over detected bounding boxes
        for box in results[0].boxes:
            # Extract bounding box coordinates
            xyxy = box.xyxy.cpu().numpy().astype(int)[0]  
            x_min, y_min, x_max, y_max = xyxy

            # OPTIONAL: Crop region below the detected signature for OCR
            # (useful if signatures are followed by names or labels)
            region_y_start = y_max + 10
            region_y_end = y_max + 100
            region_x_start = x_min
            region_x_end = x_max
            cropped_region = image[region_y_start:region_y_end, region_x_start:region_x_end]

            # OCR placeholder (future use: extract text below signature)
            # text_below = pytesseract.image_to_string(cropped_region)

            # Append detection details
            detection_data.append({
                "signature_box": xyxy.tolist(),     # [x_min, y_min, x_max, y_max]
                "confidence": float(box.conf),      # Detection confidence
                # "ocr_text_below": text_below.strip() if text_below else None
            })

        # ---------------------------------------------------------------
        # Step 5: Return structured response
        # ---------------------------------------------------------------
        return {
            "filename": file.filename,
            "signatures_detected": len(detection_data),
            "detections": detection_data
        }


# -----------------------------------------------------------------------------
# MAIN ENTRY POINT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Port defaults to 7860 (commonly used in Hugging Face Spaces)
    # Can be overridden by setting PORT in environment variables
    port = int(os.environ.get("PORT", 7860))
    
    # Start the server
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)