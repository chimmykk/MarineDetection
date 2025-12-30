# Underwater Image Enhancement + Marine Life Detection

A complete AI system for underwater image processing with two stages:

1. **Image Enhancement**: Denoising, color correction, dehazing
2. **Marine Life Detection**: YOLOv5-based object detection

> **Note**: This system is optimized for **CPU-only** environments. All PyTorch operations will automatically use CPU.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Stage 1: Image Enhancement](#stage-1-image-enhancement)
- [Stage 2: Marine Life Detection](#stage-2-marine-life-detection)
- [Full Pipeline](#full-pipeline)
- [Training Your Own Models](#training-your-own-models)
- [API Reference](#api-reference)

## Installation

### Requirements

- Python 3.8+
- macOS, Linux, or Windows

### Setup

```bash
# Clone/navigate to project
cd underwater-ai

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Process a Single Image

```python
from pipeline.run_pipeline import UnderwaterPipeline

# Initialize pipeline
pipeline = UnderwaterPipeline()

# Process image
result = pipeline.process_image('input.jpg', 'output/')

print(f"Found {len(result.detections)} marine life objects")
```

### Command Line

```bash
# Process single image
python -m pipeline.run_pipeline input.jpg output/

# Process directory
python -m pipeline.run_pipeline data/raw/ data/enhanced/

# Process video
python -m pipeline.run_pipeline underwater_video.mp4 enhanced_video.mp4
```

## Project Structure

```
underwater-ai/
├── data/
│   ├── raw/                  # Original underwater images
│   ├── enhanced/             # Enhanced outputs
│   └── detection/            # YOLOv5 training data
├── denoising/                # Image enhancement modules
│   ├── clahe.py             # CLAHE enhancement
│   ├── white_balance.py     # Color correction
│   ├── dehazing.py          # Underwater dehazing
│   └── unet/                # Deep learning enhancement
├── detection/                # Object detection modules
│   ├── train.py             # YOLOv5 training
│   ├── detect.py            # Inference
│   └── utils.py             # Utilities
├── pipeline/                 # End-to-end pipeline
├── configs/                  # Configuration files
└── notebooks/               # Jupyter notebooks
```

## Stage 1: Image Enhancement

### Traditional Methods

```python
from denoising import apply_clahe, gray_world, underwater_dehaze

# Load image with OpenCV
import cv2
image = cv2.imread('underwater.jpg')

# Apply CLAHE (contrast enhancement)
enhanced = apply_clahe(image, clip_limit=2.0)

# Apply white balance correction
balanced = gray_world(image)

# Apply dehazing
dehazed = underwater_dehaze(image)
```

### Combined Enhancement

```python
from denoising.clahe import adaptive_clahe
from denoising.white_balance import combined_white_balance
from denoising.dehazing import underwater_dehaze

# Apply full enhancement pipeline
image = cv2.imread('underwater.jpg')

# 1. White balance
image = combined_white_balance(image, method='auto')

# 2. Adaptive CLAHE
image = adaptive_clahe(image)

# 3. Dehazing
enhanced = underwater_dehaze(image)

cv2.imwrite('enhanced.jpg', enhanced)
```

### Batch Processing

```bash
# CLAHE on directory
python -m denoising.clahe input_dir/ output_dir/ --clip-limit 2.5

# White balance on directory
python -m denoising.white_balance input_dir/ output_dir/ --method auto

# Dehazing on directory
python -m denoising.dehazing input_dir/ output_dir/ --method underwater
```

## Stage 2: Marine Life Detection

### Using Pretrained Model

```python
from detection import MarineDetector

# Initialize detector (uses pretrained YOLOv5s if no custom model)
detector = MarineDetector(confidence_threshold=0.3)

# Detect on single image
detections, annotated_image = detector.detect('enhanced.jpg')

for det in detections:
    print(f"{det['class_name']}: {det['confidence']:.2f}")

# Save annotated image
cv2.imwrite('detected.jpg', annotated_image)
```

### Detection on Directory

```python
# Detect on all images in directory
detector.detect_directory(
    input_dir='data/enhanced/',
    output_dir='data/detected/',
    save_json=True,
    save_csv=True
)
```

### Supported Classes

| ID  | Class Name | Description                 |
| --- | ---------- | --------------------------- |
| 0   | fish       | General fish category       |
| 1   | shark      | All shark species           |
| 2   | jellyfish  | Jellyfish and related       |
| 3   | starfish   | Starfish and sea urchins    |
| 4   | coral      | Healthy coral formations    |
| 5   | diseased   | Diseased/bleached organisms |
| 6   | damaged    | Environmental damage        |

## Full Pipeline

### Python API

```python
from pipeline.run_pipeline import UnderwaterPipeline, PipelineConfig

# Custom configuration
config = PipelineConfig(
    enhancement_method='combined',
    detection_enabled=True,
    detection_confidence=0.25,
    save_enhanced=True,
    save_annotated=True,
    save_json=True
)

# Initialize pipeline
pipeline = UnderwaterPipeline(config)

# Process single image
result = pipeline.process_image('underwater.jpg', 'output/')

# Process directory
results = pipeline.process_directory('input_dir/', 'output_dir/')

# Process video
detections = pipeline.process_video('video.mp4', 'output_video.mp4')
```

### Command Line

```bash
# Full pipeline on image
python -m pipeline.run_pipeline input.jpg output/ \
    --enhancement combined \
    --conf 0.3

# Enhancement only (no detection)
python -m pipeline.run_pipeline input.jpg output/ --no-detection

# Video processing
python -m pipeline.run_pipeline video.mp4 output.mp4 --skip-frames 2
```

## 🏋 Training Your Own Models

### U-Net Enhancement

```bash
# Prepare paired dataset (degraded -> clean images)
# data/train/input/  - degraded images
# data/train/target/ - clean reference images

# Train U-Net
python -m denoising.unet.train \
    --train-input data/train/input \
    --train-target data/train/target \
    --val-input data/val/input \
    --val-target data/val/target \
    --epochs 100 \
    --batch-size 4 \
    --model-type standard
```

### YOLOv5 Detection

```bash
# Prepare YOLO format dataset
# data/detection/images/train/
# data/detection/images/val/
# data/detection/labels/train/
# data/detection/labels/val/

# Train YOLOv5
python -m detection.train \
    --data data/detection \
    --epochs 100 \
    --batch-size 8 \
    --model n  # Use nano model for CPU
```

### Dataset Preparation

```bash
# Convert COCO to YOLO format
python -m detection.utils coco2yolo \
    --json annotations.json \
    --images images/ \
    --output data/detection/

# Validate labels
python -m detection.utils validate \
    --labels data/detection/labels/train \
    --images data/detection/images/train \
    --num-classes 7

# Split dataset
python -m detection.utils split \
    --images data/all/images \
    --labels data/all/labels \
    --output data/detection \
    --train 0.8 --val 0.1
```

## 📚 API Reference

### Enhancement Functions

```python
# CLAHE
apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8), color_space='LAB')
adaptive_clahe(image)  # Auto-adjusts based on brightness
batch_clahe(input_dir, output_dir)

# White Balance
gray_world(image)
shades_of_gray(image, p=6.0)
white_patch_retinex(image, percentile=99.0)
combined_white_balance(image, method='auto')

# Dehazing
dark_channel_prior(image, omega=0.95, patch_size=15)
underwater_dehaze(image, enhance_red=True)
simple_dehaze(image, strength=0.5)
```

### Detection Classes

```python
from detection import MarineDetector

detector = MarineDetector(
    model_path=None,          # Custom model path
    classes=['fish', ...],    # Class names
    confidence_threshold=0.25,
    iou_threshold=0.45,
    device='cpu'
)

detections, annotated = detector.detect(image)
detector.detect_directory(input_dir, output_dir)
detector.detect_video(video_path, output_path)
```

### Pipeline Configuration

```python
from pipeline.run_pipeline import PipelineConfig

config = PipelineConfig(
    # Enhancement
    enhancement_method='combined',  # clahe, white_balance, dehaze, unet, combined
    clahe_clip_limit=2.0,
    dehaze_enabled=True,

    # Detection
    detection_enabled=True,
    detection_model=None,  # Custom model path
    detection_confidence=0.25,

    # Output
    save_enhanced=True,
    save_annotated=True,
    save_json=True,
    output_format='jpg',

    # Processing
    target_size=(640, 640),
    preserve_original_size=True
)
```

// to train more

```bash
# 1. Activate the environment
source venv/bin/activate

# 2. Start high-epoch training
python -m detection.train \
    --data data/detection \
    --epochs 50 \
    --batch-size 4 \
    --img-size 320 \
    --model s \
    --output runs/marine_life
```

## Web Dashboard (Next.js + FastAPI)

A web interface for image enhancement and detection.

### 1. Start the Backend API

Run this from the `underwater-ai/` root directory:

```bash
source venv/bin/activate
python -m uvicorn backend.main:app --port 8000
```

### 2. Start the Frontend UI

Run this from the `underwater-ai/frontend-ui/` directory:

```bash
cd frontend-ui
npm install    # (Only if not done already)
npm run dev
```

Then open [http://localhost:3000](http://localhost:3000) in your browser.

## CPU Optimization Tips

1. **Use nano/small models**: YOLOv5n or YOLOv5s for faster inference
2. **Reduce batch size**: Use batch size 2-4 for training
3. **Cache enhanced images**: Save enhanced images to avoid re-processing
4. **Resize inputs**: Smaller images process faster (640x640 recommended)
5. **Skip video frames**: Use `--skip-frames` for faster video processing

## License

MIT License / ISC
All right Reserved

## Contributing

No this is a hobby project pls don't consider i would not continuing maintaining this project
in whatsoever manner
