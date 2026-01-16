# Unified Marine Detection Model

This document describes the unified detection model that combines:
- **Marine Life Detection** (7 classes)
- **Fish Species Identification** (12 classes)
- **Fish Disease Detection** (4 classes)

## Total Classes: 23

### Marine Life (Classes 0-6)
| ID | Class Name | Description |
|----|------------|-------------|
| 0 | fish | Generic fish |
| 1 | jellyfish | Jellyfish |
| 2 | penguin | Penguin |
| 3 | puffin | Puffin |
| 4 | shark | Shark (all species) |
| 5 | starfish | Starfish |
| 6 | stingray | Stingray |

### Fish Species (Classes 7-18)
| ID | Class Name | Scientific Family |
|----|------------|-------------------|
| 7 | surgeonfish | Acanthuridae |
| 8 | triggerfish | Balistidae |
| 9 | jack | Carangidae |
| 10 | spadefish | Ephippidae |
| 11 | wrasse | Labridae |
| 12 | snapper | Lutjanidae |
| 13 | angelfish | Pomacanthidae |
| 14 | damselfish | Pomacentridae |
| 15 | parrotfish | Scaridae |
| 16 | tuna | Scombridae |
| 17 | grouper | Serranidae |
| 18 | moorish_idol | Zanclidae |

### Fish Disease (Classes 19-22)
| ID | Class Name | Description |
|----|------------|-------------|
| 19 | bacterial_gill_disease | Bacterial gill infection |
| 20 | bacterial_red_disease | Bacterial red disease |
| 21 | bacterial_disease | General bacterial disease |
| 22 | healthy_fish | Healthy fish (no disease) |

## Dataset Statistics

- **Training Images**: ~3,000
- **Validation Images**: ~550
- **Sources**:
  - Existing marine dataset
  - Roboflow Fish Species dataset
  - Roboflow Fish Disease dataset

## Training the Model

### Quick Start (CPU)
```bash
python scripts/train_unified.py --epochs 50 --batch-size 4 --device cpu
```

### GPU Training (Recommended)
```bash
python scripts/train_unified.py --epochs 100 --batch-size 16 --device 0
```

### Training Options
```
--epochs      Number of training epochs (default: 100)
--batch-size  Batch size (default: 16, use 4-8 for CPU)
--img-size    Image size (default: 640)
--model       Model size: n, s, m, l, x (default: s)
--device      Device: cpu, 0, 0,1 (default: cpu)
--resume      Resume from last checkpoint
```

## Using the Model

### Python API
```python
from detection.unified_detector import UnifiedMarineDetector, DetectionCategory

# Initialize detector
detector = UnifiedMarineDetector(
    model_path="models/unified_marine_detector.pt",
    confidence_threshold=0.25
)

# Detect all categories
detections, annotated = detector.detect("image.jpg")

# Filter by category
marine_only, _ = detector.detect("image.jpg", category_filter=DetectionCategory.MARINE)
species_only, _ = detector.detect("image.jpg", category_filter=DetectionCategory.SPECIES)
disease_only, _ = detector.detect("image.jpg", category_filter=DetectionCategory.DISEASE)
```

### Command Line
```bash
# Detect all
python detection/unified_detector.py image.jpg --output result.jpg

# Filter by category
python detection/unified_detector.py image.jpg --category species --output result.jpg
```

### REST API
```bash
# Start backend
python backend/main.py

# Process image with category filter
curl -X POST "http://localhost:8000/process" \
  -F "file=@image.jpg" \
  -F "category=species" \
  -F "confidence=0.25"
```

## Frontend Integration

The frontend can now filter detections by category:

```typescript
const formData = new FormData();
formData.append("file", file);
formData.append("category", "species"); // all, marine, species, disease
formData.append("confidence", "0.25");

const response = await fetch("http://localhost:8000/process", {
  method: "POST",
  body: formData,
});
```

## Model Files

After training, copy the best weights:
```bash
cp runs/train/unified_marine_detector/weights/best.pt models/unified_marine_detector.pt
```

The backend will automatically detect and use the unified model from:
1. `models/unified_marine_detector.pt`
2. `runs/train/unified_marine_detector/weights/best.pt`
3. `yolov5su.pt` (fallback)
