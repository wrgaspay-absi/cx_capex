import uvicorn
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import cv2
import numpy as np
import os
from pdf2image import convert_from_bytes
import pytesseract
import tempfile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Signature Detection API")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "API is running"}



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your trained YOLO model
model = YOLO("best.pt")

@app.post("/detect")
async def detect_signatures(file: UploadFile = File(...)):
    # Create a temporary directory to store files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Read uploaded file as bytes
        file_bytes = await file.read()
        file_ext = file.filename.split(".")[-1].lower()

        image = None

        # ✅ Step 1: If PDF → Convert first page to image
        if file_ext == "pdf":
            pages = convert_from_bytes(file_bytes, dpi=300)
            img_path = os.path.join(tmpdir, "page1.png")
            pages[0].save(img_path, "PNG")
            image = cv2.imread(img_path)
        else:
            # Assume it's an image (PNG, JPG)
            np_arr = np.frombuffer(file_bytes, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return {"error": "Unable to process file"}

        # ✅ Step 2: Preprocessing (Grayscale + Contrast Enhancement)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        alpha = 1.5  # Contrast factor
        beta = 30    # Brightness
        enhanced = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
        preprocessed_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        # ✅ Step 3: YOLO detection
        results = model(preprocessed_bgr, conf=0.6)

        detection_data = []
        for box in results[0].boxes:
            xyxy = box.xyxy.cpu().numpy().astype(int)[0]
            x_min, y_min, x_max, y_max = xyxy

            # Region below the signature for OCR
            region_y_start = y_max + 10
            region_y_end = y_max + 100
            region_x_start = x_min
            region_x_end = x_max

            cropped_region = image[region_y_start:region_y_end, region_x_start:region_x_end]


            detection_data.append({
                "signature_box": xyxy.tolist(),
                "confidence": float(box.conf),
            })

        return {
            "filename": file.filename,
            "signatures_detected": len(detection_data),
            "detections": detection_data
        }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))  # For HF Spaces or local run
    uvicorn.run("app:app", host="0.0.0.0", port=port)
