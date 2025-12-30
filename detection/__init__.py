"""
Detection Module for Marine Life Detection

Provides YOLOv5-based object detection for underwater marine life,
including training, inference, and utility functions.
"""

from .detect import MarineDetector
from .utils import convert_coco_to_yolo, validate_labels

__all__ = [
    'MarineDetector',
    'convert_coco_to_yolo',
    'validate_labels',
]
