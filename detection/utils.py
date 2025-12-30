"""
Detection Utilities

Helper functions for dataset conversion, label validation,
and metrics calculation.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict


def convert_coco_to_yolo(
    coco_json: str,
    images_dir: str,
    output_dir: str,
    class_mapping: Optional[Dict[str, int]] = None
) -> None:
    """
    Convert COCO format annotations to YOLO format.
    
    COCO format: JSON with images, annotations (bbox as [x, y, width, height])
    YOLO format: TXT files with [class_id, x_center, y_center, width, height] (normalized)
    
    Args:
        coco_json: Path to COCO JSON annotation file
        images_dir: Directory containing images
        output_dir: Output directory for YOLO format
        class_mapping: Optional mapping from COCO category names to class IDs
    """
    output_dir = Path(output_dir)
    images_output = output_dir / 'images'
    labels_output = output_dir / 'labels'
    
    images_output.mkdir(parents=True, exist_ok=True)
    labels_output.mkdir(parents=True, exist_ok=True)
    
    # Load COCO annotations
    with open(coco_json, 'r') as f:
        coco = json.load(f)
    
    # Build mappings
    image_info = {img['id']: img for img in coco['images']}
    
    # Category mapping
    if class_mapping is None:
        # Create mapping from COCO categories
        class_mapping = {
            cat['name']: idx for idx, cat in enumerate(coco['categories'])
        }
    
    cat_id_to_class = {
        cat['id']: class_mapping.get(cat['name'], 0)
        for cat in coco['categories']
    }
    
    # Group annotations by image
    annotations_by_image = defaultdict(list)
    for ann in coco['annotations']:
        annotations_by_image[ann['image_id']].append(ann)
    
    # Convert each image
    for img_id, img_data in image_info.items():
        img_filename = img_data['file_name']
        img_width = img_data['width']
        img_height = img_data['height']
        
        # Copy image
        src_img = Path(images_dir) / img_filename
        if src_img.exists():
            shutil.copy2(src_img, images_output / img_filename)
        
        # Create label file
        label_filename = Path(img_filename).stem + '.txt'
        label_path = labels_output / label_filename
        
        with open(label_path, 'w') as f:
            for ann in annotations_by_image[img_id]:
                # COCO bbox format: [x, y, width, height] (top-left corner)
                x, y, w, h = ann['bbox']
                
                # Convert to YOLO format (center, normalized)
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                w_norm = w / img_width
                h_norm = h / img_height
                
                # Get class ID
                class_id = cat_id_to_class.get(ann['category_id'], 0)
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
    
    print(f"Converted {len(image_info)} images to YOLO format")
    print(f"Images: {images_output}")
    print(f"Labels: {labels_output}")


def convert_voc_to_yolo(
    voc_dir: str,
    output_dir: str,
    class_names: List[str]
) -> None:
    """
    Convert Pascal VOC format annotations to YOLO format.
    
    VOC format: XML files with bounding boxes
    YOLO format: TXT files with normalized coordinates
    
    Args:
        voc_dir: Directory containing VOC XML files and images
        output_dir: Output directory for YOLO format
        class_names: List of class names (order determines class ID)
    """
    import xml.etree.ElementTree as ET
    
    output_dir = Path(output_dir)
    images_output = output_dir / 'images'
    labels_output = output_dir / 'labels'
    
    images_output.mkdir(parents=True, exist_ok=True)
    labels_output.mkdir(parents=True, exist_ok=True)
    
    voc_dir = Path(voc_dir)
    annotations_dir = voc_dir / 'Annotations'
    images_dir = voc_dir / 'JPEGImages'
    
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    
    for xml_file in annotations_dir.glob('*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Get image info
        filename = root.find('filename').text
        size = root.find('size')
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)
        
        # Copy image
        src_img = images_dir / filename
        if src_img.exists():
            shutil.copy2(src_img, images_output / filename)
        
        # Create label file
        label_filename = Path(filename).stem + '.txt'
        label_path = labels_output / label_filename
        
        with open(label_path, 'w') as f:
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                if class_name not in class_to_id:
                    continue
                
                class_id = class_to_id[class_name]
                
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                
                # Convert to YOLO format
                x_center = ((xmin + xmax) / 2) / img_width
                y_center = ((ymin + ymax) / 2) / img_height
                w_norm = (xmax - xmin) / img_width
                h_norm = (ymax - ymin) / img_height
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
    
    print(f"Converted annotations to YOLO format")


def validate_labels(
    labels_dir: str,
    images_dir: str,
    num_classes: int
) -> Dict[str, List[str]]:
    """
    Validate YOLO format labels.
    
    Checks for:
    - Missing label files
    - Invalid class IDs
    - Invalid bounding box coordinates
    - Empty label files
    
    Args:
        labels_dir: Directory containing label files
        images_dir: Directory containing images
        num_classes: Expected number of classes
    
    Returns:
        Dictionary with validation issues
    """
    labels_dir = Path(labels_dir)
    images_dir = Path(images_dir)
    
    issues = {
        'missing_labels': [],
        'invalid_class_id': [],
        'invalid_bbox': [],
        'empty_labels': [],
        'orphan_labels': []
    }
    
    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    images = {
        f.stem: f
        for f in images_dir.iterdir()
        if f.suffix.lower() in image_extensions
    }
    
    # Find all labels
    labels = {f.stem: f for f in labels_dir.glob('*.txt')}
    
    # Check for missing labels
    for img_stem in images:
        if img_stem not in labels:
            issues['missing_labels'].append(images[img_stem].name)
    
    # Check for orphan labels
    for label_stem in labels:
        if label_stem not in images:
            issues['orphan_labels'].append(labels[label_stem].name)
    
    # Validate each label file
    for label_stem, label_path in labels.items():
        content = label_path.read_text().strip()
        
        if not content:
            issues['empty_labels'].append(label_path.name)
            continue
        
        for line_num, line in enumerate(content.split('\n'), 1):
            parts = line.strip().split()
            
            if len(parts) != 5:
                issues['invalid_bbox'].append(f"{label_path.name}:{line_num}")
                continue
            
            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Check class ID
                if class_id < 0 or class_id >= num_classes:
                    issues['invalid_class_id'].append(f"{label_path.name}:{line_num}")
                
                # Check coordinates (should be normalized 0-1)
                for val in [x_center, y_center, width, height]:
                    if val < 0 or val > 1:
                        issues['invalid_bbox'].append(f"{label_path.name}:{line_num}")
                        break
                        
            except ValueError:
                issues['invalid_bbox'].append(f"{label_path.name}:{line_num}")
    
    # Print summary
    print("\nValidation Summary:")
    print(f"  Total images: {len(images)}")
    print(f"  Total labels: {len(labels)}")
    print(f"  Missing labels: {len(issues['missing_labels'])}")
    print(f"  Invalid class IDs: {len(issues['invalid_class_id'])}")
    print(f"  Invalid bboxes: {len(issues['invalid_bbox'])}")
    print(f"  Empty labels: {len(issues['empty_labels'])}")
    print(f"  Orphan labels: {len(issues['orphan_labels'])}")
    
    return issues


def split_dataset(
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Dict[str, int]:
    """
    Split dataset into train/val/test sets.
    
    Args:
        images_dir: Source images directory
        labels_dir: Source labels directory
        output_dir: Output directory
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with split counts
    """
    import random
    
    random.seed(seed)
    
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    
    # Find matching image-label pairs
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    pairs = []
    
    for img_path in images_dir.iterdir():
        if img_path.suffix.lower() not in image_extensions:
            continue
        label_path = labels_dir / (img_path.stem + '.txt')
        if label_path.exists():
            pairs.append((img_path, label_path))
    
    # Shuffle
    random.shuffle(pairs)
    
    # Split
    n = len(pairs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    splits = {
        'train': pairs[:n_train],
        'val': pairs[n_train:n_train + n_val],
        'test': pairs[n_train + n_val:]
    }
    
    # Create directories and copy files
    for split_name, split_pairs in splits.items():
        img_dir = output_dir / 'images' / split_name
        lbl_dir = output_dir / 'labels' / split_name
        
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path, lbl_path in split_pairs:
            shutil.copy2(img_path, img_dir / img_path.name)
            shutil.copy2(lbl_path, lbl_dir / lbl_path.name)
    
    counts = {name: len(pairs) for name, pairs in splits.items()}
    print(f"Dataset split: train={counts['train']}, val={counts['val']}, test={counts['test']}")
    
    return counts


def calculate_metrics(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate detection metrics (precision, recall, mAP).
    
    Args:
        predictions: List of prediction dicts with 'class_id', 'confidence', 'bbox'
        ground_truth: List of ground truth dicts with 'class_id', 'bbox'
        iou_threshold: IoU threshold for true positive
    
    Returns:
        Dictionary of metrics
    """
    if not predictions or not ground_truth:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    # Sort predictions by confidence
    predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
    
    gt_matched = [False] * len(ground_truth)
    tp = 0
    fp = 0
    
    for pred in predictions:
        best_iou = 0
        best_idx = -1
        
        for i, gt in enumerate(ground_truth):
            if gt_matched[i] or gt['class_id'] != pred['class_id']:
                continue
            
            iou = calculate_iou(pred['bbox'], gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        
        if best_iou >= iou_threshold:
            tp += 1
            gt_matched[best_idx] = True
        else:
            fp += 1
    
    fn = sum(1 for m in gt_matched if not m)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn
    }


def calculate_iou(box1: Dict, box2: Dict) -> float:
    """
    Calculate Intersection over Union between two boxes.
    
    Args:
        box1: Dict with 'x1', 'y1', 'x2', 'y2'
        box2: Dict with 'x1', 'y1', 'x2', 'y2'
    
    Returns:
        IoU value (0-1)
    """
    # Intersection coordinates
    x1 = max(box1['x1'], box2['x1'])
    y1 = max(box1['y1'], box2['y1'])
    x2 = min(box1['x2'], box2['x2'])
    y2 = min(box1['y2'], box2['y2'])
    
    # Intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Union area
    area1 = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
    area2 = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Detection utilities')
    subparsers = parser.add_subparsers(dest='command')
    
    # COCO to YOLO conversion
    coco_parser = subparsers.add_parser('coco2yolo', help='Convert COCO to YOLO format')
    coco_parser.add_argument('--json', required=True, help='COCO JSON file')
    coco_parser.add_argument('--images', required=True, help='Images directory')
    coco_parser.add_argument('--output', required=True, help='Output directory')
    
    # Validate labels
    val_parser = subparsers.add_parser('validate', help='Validate YOLO labels')
    val_parser.add_argument('--labels', required=True, help='Labels directory')
    val_parser.add_argument('--images', required=True, help='Images directory')
    val_parser.add_argument('--num-classes', type=int, required=True, help='Number of classes')
    
    # Split dataset
    split_parser = subparsers.add_parser('split', help='Split dataset')
    split_parser.add_argument('--images', required=True, help='Images directory')
    split_parser.add_argument('--labels', required=True, help='Labels directory')
    split_parser.add_argument('--output', required=True, help='Output directory')
    split_parser.add_argument('--train', type=float, default=0.8, help='Train ratio')
    split_parser.add_argument('--val', type=float, default=0.1, help='Val ratio')
    
    args = parser.parse_args()
    
    if args.command == 'coco2yolo':
        convert_coco_to_yolo(args.json, args.images, args.output)
    elif args.command == 'validate':
        validate_labels(args.labels, args.images, args.num_classes)
    elif args.command == 'split':
        split_dataset(args.images, args.labels, args.output, args.train, args.val)
