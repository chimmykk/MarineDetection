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
from detection.unified_detector import UnifiedMarineDetector, DetectionCategory

app = FastAPI(title="Underwater Processing API")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model paths in priority order
MODEL_PATHS = [
    str(Path(__file__).parent.parent / "models/unified_marine_detector.pt"),
    str(Path(__file__).parent.parent / "runs/train/unified_marine_detector/weights/best.pt"),
    str(Path(__file__).parent.parent / "results/underwater_test/weights/best.pt"),
    "yolov5su.pt"
]

# Find first available model
DEFAULT_MODEL = None
for path in MODEL_PATHS:
    if os.path.exists(path):
        DEFAULT_MODEL = path
        break

if DEFAULT_MODEL is None:
    DEFAULT_MODEL = "yolov5su.pt"

print(f"Using detection model: {DEFAULT_MODEL}")

pipeline_config = PipelineConfig(
    detection_model=DEFAULT_MODEL,
    enhancement_method="combined",
    detection_enabled=True,
    detection_confidence=0.25
)
pipeline = UnderwaterPipeline(pipeline_config)

# Initialize unified detector for category-based detection
unified_detector = None
try:
    unified_detector = UnifiedMarineDetector(
        model_path=DEFAULT_MODEL,
        confidence_threshold=0.25
    )
    print("Unified detector initialized successfully")
except Exception as e:
    print(f"Warning: Could not initialize unified detector: {e}")

@app.get("/")
async def root():
    return {"message": "API is running"}

@app.post("/process")
async def process_image(
    file: UploadFile = File(...),
    enhancement: str = Form("combined"),
    confidence: float = Form(0.25),
    detection: bool = Form(True),
    category: str = Form("all")  # New: filter by category (all, marine, species, disease)
):
    """
    Process an underwater image with enhancement and detection.
    
    Args:
        file: Image file to process
        enhancement: Enhancement method (combined, clahe, dehaze, white_balance)
        confidence: Detection confidence threshold
        detection: Enable/disable detection
        category: Filter detections by category (all, marine, species, disease)
    """
    import time
    start_time = time.time()
    
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

    # 3. Process image (enhancement)
    result = pipeline.process_image(image)
    
    # 4. If unified detector available and detection enabled, use it for category filtering
    detections = result.detections
    annotated_image = result.annotated_image
    
    if detection and unified_detector is not None:
        # Map category string to enum
        category_map = {
            'all': DetectionCategory.ALL,
            'marine': DetectionCategory.MARINE,
            'species': DetectionCategory.SPECIES,
            'disease': DetectionCategory.DISEASE,
        }
        cat_filter = category_map.get(category, DetectionCategory.ALL)
        
        # Run unified detection on enhanced image
        unified_detector.conf_threshold = confidence
        unified_detections, unified_annotated = unified_detector.detect(
            result.enhanced_image,
            category_filter=cat_filter,
            visualize=True
        )
        
        # Use unified results
        detections = unified_detections
        annotated_image = unified_annotated

    # 5. Prepare response
    # Convert enhanced image to base64
    _, enhanced_buffer = cv2.imencode('.jpg', result.enhanced_image)
    enhanced_base64 = base64.b64encode(enhanced_buffer).decode('utf-8')

    # Convert annotated image to base64
    annotated_base64 = None
    if annotated_image is not None:
        _, annotated_buffer = cv2.imencode('.jpg', annotated_image)
        annotated_base64 = base64.b64encode(annotated_buffer).decode('utf-8')

    processing_time = time.time() - start_time

    return {
        "enhanced_image": f"data:image/jpeg;base64,{enhanced_base64}",
        "annotated_image": f"data:image/jpeg;base64,{annotated_base64}" if annotated_base64 else None,
        "detections": detections,
        "processing_time": processing_time,
        "category_filter": category,
        "model": DEFAULT_MODEL
    }


@app.get("/categories")
async def get_categories():
    """Get available detection categories and their classes."""
    from detection.unified_detector import UNIFIED_CLASSES
    
    categories = {
        'marine': [],
        'species': [],
        'disease': []
    }
    
    for class_id, info in UNIFIED_CLASSES.items():
        cat = info['category']
        if cat in categories:
            categories[cat].append({
                'id': class_id,
                'name': info['name']
            })
    
    return {
        'categories': categories,
        'total_classes': len(UNIFIED_CLASSES)
    }


@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model."""
    return {
        'model_path': DEFAULT_MODEL,
        'unified_detector_loaded': unified_detector is not None,
        'available_categories': ['all', 'marine', 'species', 'disease']
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
