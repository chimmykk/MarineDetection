import os
import sys
import uuid
import base64
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import List, Optional
import io

# Add parent to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from pipeline.run_pipeline import UnderwaterPipeline, PipelineConfig

app = FastAPI(title="Underwater Processing API")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline (global)
# Using defaults, can be re-initialized per request if needed
DEFAULT_MODEL = str(Path(__file__).parent.parent / "results/underwater_test/weights/best.pt")
if not os.path.exists(DEFAULT_MODEL):
    DEFAULT_MODEL = "yolov5su.pt" # Fallback

pipeline_config = PipelineConfig(
    detection_model=DEFAULT_MODEL,
    enhancement_method="combined",
    detection_enabled=True,
    detection_confidence=0.25
)
pipeline = UnderwaterPipeline(pipeline_config)

@app.get("/")
async def root():
    return {"message": "API is running"}

@app.post("/process")
async def process_image(
    file: UploadFile = File(...),
    enhancement: str = Form("combined"),
    confidence: float = Form(0.25),
    detection: bool = Form(True)
):
    # 1. Read uploaded file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return {"error": "Invalid image file"}

    # 2. Update pipeline config for this request
    pipeline.config.enhancement_method = enhancement
    pipeline.config.detection_confidence = confidence
    pipeline.config.detection_enabled = detection

    # 3. Process image
    result = pipeline.process_image(image)

    # 4. Prepare response
    # Convert enhanced image to base64
    _, enhanced_buffer = cv2.imencode('.jpg', result.enhanced_image)
    enhanced_base64 = base64.b64encode(enhanced_buffer).decode('utf-8')

    # Convert annotated image to base64
    annotated_base64 = None
    if result.annotated_image is not None:
        _, annotated_buffer = cv2.imencode('.jpg', result.annotated_image)
        annotated_base64 = base64.b64encode(annotated_buffer).decode('utf-8')

    return {
        "enhanced_image": f"data:image/jpeg;base64,{enhanced_base64}",
        "annotated_image": f"data:image/jpeg;base64,{annotated_base64}" if annotated_base64 else None,
        "detections": result.detections,
        "processing_time": result.processing_time
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
