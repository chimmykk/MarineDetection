"""
Unified Marine Detector

Supports detection of:
- Marine life (fish, jellyfish, penguin, puffin, shark, starfish, stingray)
- Fish species (13 species)
- Fish diseases (4 disease states)

Provides category-based filtering and grouping for frontend display.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple
from enum import Enum


class DetectionCategory(Enum):
    """Detection category for filtering results."""
    MARINE = "marine"
    SPECIES = "species"
    DISEASE = "disease"
    ALL = "all"


# Unified class definitions
UNIFIED_CLASSES = {
    # Marine life (0-6)
    0: {'name': 'fish', 'category': 'marine', 'color': (255, 128, 0)},
    1: {'name': 'jellyfish', 'category': 'marine', 'color': (255, 0, 255)},
    2: {'name': 'penguin', 'category': 'marine', 'color': (0, 255, 255)},
    3: {'name': 'puffin', 'category': 'marine', 'color': (128, 255, 0)},
    4: {'name': 'shark', 'category': 'marine', 'color': (0, 0, 255)},
    5: {'name': 'starfish', 'category': 'marine', 'color': (255, 255, 0)},
    6: {'name': 'stingray', 'category': 'marine', 'color': (128, 128, 255)},
    
    # Fish species (7-18)
    7: {'name': 'surgeonfish', 'category': 'species', 'color': (0, 200, 200)},
    8: {'name': 'triggerfish', 'category': 'species', 'color': (200, 100, 0)},
    9: {'name': 'jack', 'category': 'species', 'color': (100, 200, 100)},
    10: {'name': 'spadefish', 'category': 'species', 'color': (200, 200, 0)},
    11: {'name': 'wrasse', 'category': 'species', 'color': (0, 100, 200)},
    12: {'name': 'snapper', 'category': 'species', 'color': (200, 0, 100)},
    13: {'name': 'angelfish', 'category': 'species', 'color': (100, 0, 200)},
    14: {'name': 'damselfish', 'category': 'species', 'color': (0, 200, 100)},
    15: {'name': 'parrotfish', 'category': 'species', 'color': (200, 100, 200)},
    16: {'name': 'tuna', 'category': 'species', 'color': (100, 200, 200)},
    17: {'name': 'grouper', 'category': 'species', 'color': (200, 200, 100)},
    18: {'name': 'moorish_idol', 'category': 'species', 'color': (150, 150, 0)},
    
    # Fish disease (19-22)
    19: {'name': 'bacterial_gill_disease', 'category': 'disease', 'color': (0, 0, 200)},
    20: {'name': 'bacterial_red_disease', 'category': 'disease', 'color': (0, 50, 255)},
    21: {'name': 'bacterial_disease', 'category': 'disease', 'color': (50, 0, 200)},
    22: {'name': 'healthy_fish', 'category': 'disease', 'color': (0, 255, 0)},
}


class UnifiedMarineDetector:
    """
    Unified detector for marine life, fish species, and fish diseases.
    
    Uses a single YOLOv5 model trained on merged datasets.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = 'cpu'
    ):
        """
        Initialize the unified detector.
        
        Args:
            model_path: Path to unified model weights
            confidence_threshold: Detection confidence threshold
            iou_threshold: NMS IoU threshold
            device: Inference device ('cpu' or 'cuda:0')
        """
        import torch
        
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.classes = UNIFIED_CLASSES
        
        # Load model
        if model_path is None:
            # Try to find unified model
            default_paths = [
                'models/unified_marine_detector.pt',
                'runs/train/unified_marine_detector/weights/best.pt',
                'yolov5su.pt'  # Fallback to existing
            ]
            for path in default_paths:
                if Path(path).exists():
                    model_path = path
                    break
        
        if model_path is None:
            print("Warning: No unified model found, using pretrained YOLOv5s")
            model_path = 'yolov5s.pt'
        
        print(f"Loading model from {model_path}...")
        
        # Load using torch hub (works with YOLOv5 repo trained models)
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
        self.model.conf = confidence_threshold
        self.model.iou = iou_threshold
    
    def detect(
        self,
        image: Union[np.ndarray, str],
        category_filter: DetectionCategory = DetectionCategory.ALL,
        visualize: bool = True
    ) -> Tuple[List[Dict], Optional[np.ndarray]]:
        """
        Detect objects in an image.
        
        Args:
            image: Input image (array or path)
            category_filter: Filter results by category
            visualize: Whether to draw bounding boxes
        
        Returns:
            Tuple of (detections list, annotated image or None)
        """
        # Load image if path
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Could not load image: {image}")
        
        # Run inference using torch hub model
        results = self.model(image)
        
        # Parse detections from pandas dataframe
        detections = []
        df = results.pandas().xyxy[0]  # Get detections as pandas dataframe
        
        for _, row in df.iterrows():
            class_id = int(row['class'])
            
            # Get class info
            if class_id in self.classes:
                class_info = self.classes[class_id]
            else:
                # Fallback for unknown classes
                class_info = {
                    'name': row['name'] if 'name' in row else f'class_{class_id}',
                    'category': 'unknown',
                    'color': (128, 128, 128)
                }
            
            # Apply category filter
            if category_filter != DetectionCategory.ALL:
                if class_info['category'] != category_filter.value:
                    continue
            
            detection = {
                'class_id': class_id,
                'class_name': class_info['name'],
                'category': class_info['category'],
                'confidence': float(row['confidence']),
                'bbox': {
                    'x1': float(row['xmin']),
                    'y1': float(row['ymin']),
                    'x2': float(row['xmax']),
                    'y2': float(row['ymax']),
                },
                'color': class_info['color']
            }
            
            # Add center and dimensions
            detection['bbox']['center_x'] = (detection['bbox']['x1'] + detection['bbox']['x2']) / 2
            detection['bbox']['center_y'] = (detection['bbox']['y1'] + detection['bbox']['y2']) / 2
            detection['bbox']['width'] = detection['bbox']['x2'] - detection['bbox']['x1']
            detection['bbox']['height'] = detection['bbox']['y2'] - detection['bbox']['y1']
            
            detections.append(detection)
        
        # Visualize if requested
        annotated = None
        if visualize:
            annotated = self.draw_detections(image.copy(), detections)
        
        return detections, annotated
    
    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Dict],
        thickness: int = 2,
        font_scale: float = 0.6
    ) -> np.ndarray:
        """Draw detection bounding boxes on image."""
        for det in detections:
            x1 = int(det['bbox']['x1'])
            y1 = int(det['bbox']['y1'])
            x2 = int(det['bbox']['x2'])
            y2 = int(det['bbox']['y2'])
            
            color = det.get('color', (255, 255, 255))
            
            # Draw box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label with category
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            category_label = f"[{det['category']}]"
            
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            
            # Label background
            cv2.rectangle(
                image,
                (x1, y1 - label_h - 10),
                (x1 + label_w + 10, y1),
                color, -1
            )
            
            # Label text
            cv2.putText(
                image, label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness
            )
        
        return image
    
    def get_summary(self, detections: List[Dict]) -> Dict:
        """
        Get a summary of detections grouped by category.
        
        Args:
            detections: List of detection dictionaries
        
        Returns:
            Summary dictionary with counts and details by category
        """
        summary = {
            'total': len(detections),
            'by_category': {
                'marine': [],
                'species': [],
                'disease': [],
            },
            'counts': {
                'marine': 0,
                'species': 0,
                'disease': 0,
            }
        }
        
        for det in detections:
            category = det.get('category', 'unknown')
            if category in summary['by_category']:
                summary['by_category'][category].append(det)
                summary['counts'][category] += 1
        
        return summary


# Convenience function for quick detection
def detect_unified(
    image_path: str,
    model_path: Optional[str] = None,
    category: str = "all",
    confidence: float = 0.25
) -> Tuple[List[Dict], np.ndarray]:
    """
    Quick detection function.
    
    Args:
        image_path: Path to image
        model_path: Path to model weights
        category: Filter category ('all', 'marine', 'species', 'disease')
        confidence: Confidence threshold
    
    Returns:
        Tuple of (detections, annotated_image)
    """
    detector = UnifiedMarineDetector(
        model_path=model_path,
        confidence_threshold=confidence
    )
    
    category_map = {
        'all': DetectionCategory.ALL,
        'marine': DetectionCategory.MARINE,
        'species': DetectionCategory.SPECIES,
        'disease': DetectionCategory.DISEASE,
    }
    
    return detector.detect(
        image_path,
        category_filter=category_map.get(category, DetectionCategory.ALL)
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Unified marine detection')
    parser.add_argument('input', help='Input image path')
    parser.add_argument('--model', help='Model weights path')
    parser.add_argument('--output', help='Output image path')
    parser.add_argument('--category', choices=['all', 'marine', 'species', 'disease'],
                        default='all', help='Filter by category')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    
    args = parser.parse_args()
    
    detections, annotated = detect_unified(
        args.input,
        model_path=args.model,
        category=args.category,
        confidence=args.conf
    )
    
    # Print results
    print(f"\nFound {len(detections)} detections:")
    for det in detections:
        print(f"  [{det['category']}] {det['class_name']}: {det['confidence']:.2f}")
    
    # Save output
    if args.output and annotated is not None:
        cv2.imwrite(args.output, annotated)
        print(f"\nSaved annotated image to {args.output}")
