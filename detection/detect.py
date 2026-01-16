"""
Marine Life Detection Inference

Provides detection on images and videos using trained YOLOv5 models
or the Ultralytics YOLO package.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple
import json
import csv
from datetime import datetime


class MarineDetector:
    """
    Marine life detector using YOLOv5/YOLOv8.
    
    Uses the ultralytics package for inference, which provides
    a unified interface for YOLO models.
    """
    
    # Default marine life classes
    DEFAULT_CLASSES = [
        'fish', 'shark', 'jellyfish', 'starfish',
        'coral', 'diseased', 'damaged'
    ]
    
    # Colors for visualization (BGR format)
    COLORS = [
        (255, 128, 0),    # Fish - orange
        (0, 0, 255),      # Shark - red
        (255, 0, 255),    # Jellyfish - magenta
        (255, 255, 0),    # Starfish - cyan
        (0, 255, 128),    # Coral - green
        (0, 128, 255),    # Diseased - orange-red
        (128, 128, 128),  # Damaged - gray
    ]
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        classes: Optional[List[str]] = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = 'cpu'
    ):
        """
        Initialize detector.
        
        Args:
            model_path: Path to trained weights (uses pretrained COCO if None)
            classes: List of class names
            confidence_threshold: Detection confidence threshold
            iou_threshold: NMS IoU threshold
            device: Inference device ('cpu' or 'cuda:0')
        """
        import torch
        
        self.classes = classes or self.DEFAULT_CLASSES
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        # Load model using torch hub (works with YOLOv5 repo trained models)
        if model_path is None:
            print("Loading pretrained YOLOv5s model...")
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=False)
        else:
            print(f"Loading model from {model_path}...")
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
        
        # Configure model
        self.model.conf = confidence_threshold
        self.model.iou = iou_threshold
    
    def detect(
        self,
        image: Union[np.ndarray, str],
        visualize: bool = True
    ) -> Tuple[List[Dict], Optional[np.ndarray]]:
        """
        Detect marine life in a single image.
        
        Args:
            image: Input image (array or path)
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
        df = results.pandas().xyxy[0]
        
        for _, row in df.iterrows():
            detection = {
                'class_id': int(row['class']),
                'class_name': row['name'],
                'confidence': float(row['confidence']),
                'bbox': {
                    'x1': float(row['xmin']),
                    'y1': float(row['ymin']),
                    'x2': float(row['xmax']),
                    'y2': float(row['ymax']),
                }
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
        """
        Draw detection bounding boxes on image.
        
        Args:
            image: Input image (will be modified)
            detections: List of detection dictionaries
            thickness: Box line thickness
            font_scale: Text font scale
        
        Returns:
            Annotated image
        """
        for det in detections:
            x1 = int(det['bbox']['x1'])
            y1 = int(det['bbox']['y1'])
            x2 = int(det['bbox']['x2'])
            y2 = int(det['bbox']['y2'])
            
            # Get color for this class
            class_id = det['class_id'] % len(self.COLORS)
            color = self.COLORS[class_id]
            
            # Draw box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"{det['class_name']}: {det['confidence']:.2f}"
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
    
    def detect_batch(
        self,
        images: List[Union[np.ndarray, str]],
        visualize: bool = True
    ) -> List[Tuple[List[Dict], Optional[np.ndarray]]]:
        """
        Detect marine life in multiple images.
        
        Args:
            images: List of images (arrays or paths)
            visualize: Whether to draw bounding boxes
        
        Returns:
            List of (detections, annotated_image) tuples
        """
        from tqdm import tqdm
        
        results = []
        for image in tqdm(images, desc="Detecting"):
            try:
                result = self.detect(image, visualize)
                results.append(result)
            except Exception as e:
                print(f"Error processing image: {e}")
                results.append(([], None))
        
        return results
    
    def detect_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        save_json: bool = True,
        save_csv: bool = True,
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')
    ) -> Dict[str, List[Dict]]:
        """
        Detect marine life in all images in a directory.
        
        Args:
            input_dir: Input directory
            output_dir: Output directory for annotated images
            save_json: Save detections to JSON file
            save_csv: Save detections to CSV file
            extensions: Image file extensions to process
        
        Returns:
            Dictionary mapping filenames to detections
        """
        from tqdm import tqdm
        
        input_dir = Path(input_dir)
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find images
        image_files = []
        for ext in extensions:
            image_files.extend(input_dir.glob(f'*{ext}'))
            image_files.extend(input_dir.glob(f'*{ext.upper()}'))
        
        all_detections = {}
        
        for img_path in tqdm(image_files, desc="Processing"):
            detections, annotated = self.detect(str(img_path), visualize=bool(output_dir))
            
            all_detections[img_path.name] = detections
            
            if output_dir and annotated is not None:
                cv2.imwrite(str(output_dir / img_path.name), annotated)
        
        # Save results
        if output_dir:
            if save_json:
                self._save_json(all_detections, output_dir / 'detections.json')
            if save_csv:
                self._save_csv(all_detections, output_dir / 'detections.csv')
        
        return all_detections
    
    def detect_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        skip_frames: int = 0
    ) -> List[Dict]:
        """
        Detect marine life in a video.
        
        Args:
            video_path: Path to input video
            output_path: Path to save annotated video (optional)
            skip_frames: Number of frames to skip between detections
        
        Returns:
            List of frame-wise detections
        """
        from tqdm import tqdm
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        all_detections = []
        frame_idx = 0
        
        pbar = tqdm(total=total_frames, desc="Processing video")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            if frame_idx % (skip_frames + 1) == 0:
                detections, annotated = self.detect(frame, visualize=bool(writer))
                all_detections.append({
                    'frame': frame_idx,
                    'timestamp': frame_idx / fps,
                    'detections': detections
                })
                
                if writer and annotated is not None:
                    writer.write(annotated)
            elif writer:
                writer.write(frame)
            
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        if writer:
            writer.release()
        
        return all_detections
    
    def _save_json(self, detections: Dict, path: Path) -> None:
        """Save detections to JSON file."""
        output = {
            'timestamp': datetime.now().isoformat(),
            'model': str(self.model.model),
            'confidence_threshold': self.conf_threshold,
            'detections': detections
        }
        with open(path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Saved detections to {path}")
    
    def _save_csv(self, detections: Dict, path: Path) -> None:
        """Save detections to CSV file."""
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'filename', 'class_id', 'class_name', 'confidence',
                'x1', 'y1', 'x2', 'y2', 'width', 'height'
            ])
            
            for filename, dets in detections.items():
                for det in dets:
                    writer.writerow([
                        filename,
                        det['class_id'],
                        det['class_name'],
                        det['confidence'],
                        det['bbox']['x1'],
                        det['bbox']['y1'],
                        det['bbox']['x2'],
                        det['bbox']['y2'],
                        det['bbox']['width'],
                        det['bbox']['height']
                    ])
        print(f"Saved detections to {path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect marine life in images/videos')
    parser.add_argument('input', help='Input image, directory, or video')
    parser.add_argument('--output', help='Output directory or video path')
    parser.add_argument('--model', help='Path to trained model weights')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU threshold')
    
    args = parser.parse_args()
    
    detector = MarineDetector(
        model_path=args.model,
        confidence_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    input_path = Path(args.input)
    
    if input_path.is_dir():
        detector.detect_directory(input_path, args.output)
    elif input_path.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv'):
        detector.detect_video(str(input_path), args.output)
    else:
        detections, annotated = detector.detect(str(input_path))
        print(f"Found {len(detections)} detections")
        
        for det in detections:
            print(f"  {det['class_name']}: {det['confidence']:.2f}")
        
        if args.output and annotated is not None:
            cv2.imwrite(args.output, annotated)
            print(f"Saved annotated image to {args.output}")
