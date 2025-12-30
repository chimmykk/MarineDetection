"""
End-to-End Underwater Image Processing Pipeline

Combines image enhancement (Stage 1) and marine life detection (Stage 2)
into a unified pipeline with configurable options.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
from tqdm import tqdm

# Import enhancement modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from denoising.clahe import apply_clahe, adaptive_clahe
from denoising.white_balance import gray_world, combined_white_balance
from denoising.dehazing import underwater_dehaze, dark_channel_prior


@dataclass
class PipelineConfig:
    """Configuration for the underwater processing pipeline."""
    
    # Enhancement settings
    enhancement_method: str = 'combined'  # 'clahe', 'white_balance', 'dehaze', 'unet', 'combined'
    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 8
    white_balance_method: str = 'auto'
    dehaze_enabled: bool = True
    
    # Detection settings
    detection_enabled: bool = True
    detection_model: Optional[str] = None  # Path to trained model
    detection_confidence: float = 0.25
    detection_iou: float = 0.45
    
    # Output settings
    save_enhanced: bool = True
    save_annotated: bool = True
    save_json: bool = True
    output_format: str = 'jpg'
    
    # Processing settings
    target_size: Optional[Tuple[int, int]] = None
    preserve_original_size: bool = True


@dataclass
class ProcessingResult:
    """Result from processing a single image."""
    filename: str
    original_path: Optional[str]
    enhanced_path: Optional[str]
    annotated_path: Optional[str]
    original_image: Optional[np.ndarray] = None
    enhanced_image: Optional[np.ndarray] = None
    annotated_image: Optional[np.ndarray] = None
    detections: List[Dict] = field(default_factory=list)
    processing_time: float = 0.0


class UnderwaterPipeline:
    """
    Complete underwater image processing pipeline.
    
    Stage 1: Image Enhancement
        - CLAHE contrast enhancement
        - White balance correction
        - Underwater dehazing
        - (Optional) U-Net deep learning enhancement
    
    Stage 2: Object Detection
        - YOLOv5 marine life detection
        - Bounding box annotation
        - JSON/CSV export
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize pipeline.
        
        Args:
            config: Pipeline configuration (uses defaults if None)
        """
        self.config = config or PipelineConfig()
        self.detector = None
        self.unet_model = None
        
        # Initialize detector if enabled
        if self.config.detection_enabled:
            self._init_detector()
        
        # Initialize U-Net if using deep learning enhancement
        if self.config.enhancement_method == 'unet':
            self._init_unet()
    
    def _init_detector(self) -> None:
        """Initialize the marine life detector."""
        try:
            from detection.detect import MarineDetector
            self.detector = MarineDetector(
                model_path=self.config.detection_model,
                confidence_threshold=self.config.detection_confidence,
                iou_threshold=self.config.detection_iou
            )
            print("Detection model loaded")
        except Exception as e:
            print(f"Could not load detector: {e}")
            print("Detection will be disabled")
            self.config.detection_enabled = False
    
    def _init_unet(self) -> None:
        """Initialize U-Net enhancement model."""
        try:
            from denoising.unet.infer import UNetInference
            # Look for pre-trained model
            model_paths = [
                Path(__file__).parent.parent / 'checkpoints' / 'best_model.pth',
                Path(__file__).parent.parent / 'checkpoints' / 'final_model.pth',
            ]
            
            for model_path in model_paths:
                if model_path.exists():
                    self.unet_model = UNetInference(str(model_path))
                    print(f"U-Net model loaded from {model_path}")
                    return
            
            print("No U-Net model found, falling back to traditional enhancement")
            self.config.enhancement_method = 'combined'
            
        except Exception as e:
            print(f"Could not load U-Net: {e}")
            self.config.enhancement_method = 'combined'
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply enhancement pipeline to image.
        
        Args:
            image: Input BGR image
        
        Returns:
            Enhanced BGR image
        """
        method = self.config.enhancement_method
        
        if method == 'unet' and self.unet_model:
            return self.unet_model.enhance(image)
        
        elif method == 'clahe':
            return apply_clahe(
                image,
                clip_limit=self.config.clahe_clip_limit,
                tile_grid_size=(self.config.clahe_tile_size, self.config.clahe_tile_size)
            )
        
        elif method == 'white_balance':
            return combined_white_balance(image, method=self.config.white_balance_method)
        
        elif method == 'dehaze':
            return underwater_dehaze(image)
        
        elif method == 'combined':
            # Apply all traditional methods in sequence
            enhanced = image.copy()
            
            # 1. White balance correction
            enhanced = combined_white_balance(enhanced, method='auto')
            
            # 2. CLAHE contrast enhancement
            enhanced = adaptive_clahe(enhanced)
            
            # 3. Dehazing (if enabled)
            if self.config.dehaze_enabled:
                enhanced = underwater_dehaze(enhanced)
            
            return enhanced
        
        else:
            print(f"Unknown enhancement method: {method}, returning original")
            return image
    
    def detect_marine_life(
        self,
        image: np.ndarray
    ) -> Tuple[List[Dict], np.ndarray]:
        """
        Detect marine life in image.
        
        Args:
            image: Input BGR image (preferably enhanced)
        
        Returns:
            Tuple of (detections list, annotated image)
        """
        if not self.config.detection_enabled or self.detector is None:
            return [], image
        
        detections, annotated = self.detector.detect(image, visualize=True)
        return detections, annotated
    
    def process_image(
        self,
        image: Union[np.ndarray, str, Path],
        output_dir: Optional[Union[str, Path]] = None
    ) -> ProcessingResult:
        """
        Process a single image through the complete pipeline.
        
        Args:
            image: Input image (array or path)
            output_dir: Directory to save outputs
        
        Returns:
            ProcessingResult with all outputs
        """
        import time
        start_time = time.time()
        
        # Load image
        if isinstance(image, (str, Path)):
            path = Path(image)
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"Could not load image: {path}")
            filename = path.stem
        else:
            img = image
            path = None
            filename = f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        original_size = (img.shape[1], img.shape[0])
        
        # Resize if needed
        if self.config.target_size:
            img = cv2.resize(img, self.config.target_size)
        
        # Stage 1: Enhancement
        enhanced = self.enhance_image(img)
        
        # Resize back if needed
        if self.config.target_size and self.config.preserve_original_size:
            enhanced = cv2.resize(enhanced, original_size)
        
        # Stage 2: Detection
        detections, annotated = self.detect_marine_life(enhanced)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create result
        result = ProcessingResult(
            filename=filename,
            original_path=str(path) if path else None,
            enhanced_path=None,
            annotated_path=None,
            enhanced_image=enhanced,
            annotated_image=annotated,
            detections=detections,
            processing_time=processing_time
        )
        
        # Save outputs
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            ext = f".{self.config.output_format}"
            
            if self.config.save_enhanced:
                enhanced_path = output_dir / f"{filename}_enhanced{ext}"
                cv2.imwrite(str(enhanced_path), enhanced)
                result.enhanced_path = str(enhanced_path)
            
            if self.config.save_annotated and self.config.detection_enabled:
                annotated_path = output_dir / f"{filename}_detected{ext}"
                cv2.imwrite(str(annotated_path), annotated)
                result.annotated_path = str(annotated_path)
        
        return result
    
    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')
    ) -> List[ProcessingResult]:
        """
        Process all images in a directory.
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
            extensions: Image file extensions to process
        
        Returns:
            List of ProcessingResult objects
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        # Find images
        image_files = []
        for ext in extensions:
            image_files.extend(input_dir.glob(f'*{ext}'))
            image_files.extend(input_dir.glob(f'*{ext.upper()}'))
        
        results = []
        all_detections = {}
        
        for img_path in tqdm(sorted(image_files), desc="Processing"):
            try:
                result = self.process_image(img_path, output_dir)
                results.append(result)
                all_detections[result.filename] = result.detections
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        # Save combined detections JSON
        if self.config.save_json and results:
            self._save_summary(output_dir, results, all_detections)
        
        return results
    
    def process_video(
        self,
        video_path: Union[str, Path],
        output_path: Union[str, Path],
        skip_frames: int = 0
    ) -> List[Dict]:
        """
        Process a video through the pipeline.
        
        Args:
            video_path: Input video path
            output_path: Output video path
            skip_frames: Frames to skip between processing
        
        Returns:
            List of frame-wise detections
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        all_detections = []
        frame_idx = 0
        
        pbar = tqdm(total=total_frames, desc="Processing video")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % (skip_frames + 1) == 0:
                # Enhance
                enhanced = self.enhance_image(frame)
                
                # Detect
                detections, annotated = self.detect_marine_life(enhanced)
                
                all_detections.append({
                    'frame': frame_idx,
                    'timestamp': frame_idx / fps,
                    'detections': detections
                })
                
                writer.write(annotated)
            else:
                # For skipped frames, just enhance
                enhanced = self.enhance_image(frame)
                writer.write(enhanced)
            
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        writer.release()
        
        # Save detections JSON
        json_path = Path(output_path).with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump({
                'source': str(video_path),
                'fps': fps,
                'total_frames': total_frames,
                'frames': all_detections
            }, f, indent=2)
        
        print(f"Processed video saved to {output_path}")
        print(f"Detections saved to {json_path}")
        
        return all_detections
    
    def _save_summary(
        self,
        output_dir: Path,
        results: List[ProcessingResult],
        detections: Dict
    ) -> None:
        """Save processing summary."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'enhancement_method': self.config.enhancement_method,
                'detection_enabled': self.config.detection_enabled,
                'detection_confidence': self.config.detection_confidence,
            },
            'statistics': {
                'total_images': len(results),
                'total_detections': sum(len(d) for d in detections.values()),
                'avg_processing_time': np.mean([r.processing_time for r in results]),
            },
            'detections': detections
        }
        
        summary_path = output_dir / 'processing_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary saved to {summary_path}")


def run_pipeline_cli():
    """Command-line interface for the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Underwater Image Enhancement and Marine Life Detection Pipeline'
    )
    
    # Input/Output
    parser.add_argument('input', help='Input image, directory, or video')
    parser.add_argument('output', help='Output directory or video path')
    
    # Enhancement options
    parser.add_argument('--enhancement', 
                        choices=['clahe', 'white_balance', 'dehaze', 'unet', 'combined'],
                        default='combined', help='Enhancement method')
    parser.add_argument('--no-dehaze', action='store_true', help='Disable dehazing')
    
    # Detection options
    parser.add_argument('--no-detection', action='store_true', help='Disable detection')
    parser.add_argument('--model', help='Path to detection model weights')
    parser.add_argument('--conf', type=float, default=0.25, help='Detection confidence')
    
    # Processing options
    parser.add_argument('--size', type=int, nargs=2, help='Target size (width height)')
    parser.add_argument('--skip-frames', type=int, default=0, help='Frames to skip for video')
    
    args = parser.parse_args()
    
    # Create config
    config = PipelineConfig(
        enhancement_method=args.enhancement,
        dehaze_enabled=not args.no_dehaze,
        detection_enabled=not args.no_detection,
        detection_model=args.model,
        detection_confidence=args.conf,
        target_size=tuple(args.size) if args.size else None,
    )
    
    # Initialize pipeline
    pipeline = UnderwaterPipeline(config)
    
    input_path = Path(args.input)
    
    if input_path.is_dir():
        # Process directory
        pipeline.process_directory(input_path, args.output)
        
    elif input_path.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv'):
        # Process video
        pipeline.process_video(input_path, args.output, args.skip_frames)
        
    else:
        # Process single image
        result = pipeline.process_image(input_path, args.output)
        
        print(f"\nProcessed: {result.filename}")
        print(f"Processing time: {result.processing_time:.2f}s")
        print(f"Detections: {len(result.detections)}")
        
        for det in result.detections:
            print(f"  - {det['class_name']}: {det['confidence']:.2f}")


if __name__ == '__main__':
    run_pipeline_cli()
